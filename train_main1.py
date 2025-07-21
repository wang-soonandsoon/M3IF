import os

from models.MItransformer import HeterogeneousTransformerBlock

# 指定使用的GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
# from data_loader.MSRS import MSRSTrainDataset
from data_loader.MSRS_dataset_1 import MSRSTrainDataset
from models.common import gradient, clamp
# from models.fusion_modelv4 import MSGFusion
from models.fusion_modelv6 import MSGFusion
from torch.cuda.amp import GradScaler, autocast
from models.cls_model import LoraCLIP
# from pytorch_msssim import ms_ssim # 似乎未使用，可以注释掉
from models.common import RGB2YCrCb
from torch.utils.tensorboard import SummaryWriter
from models.new_loss import FocalLoss
criterion = nn.CrossEntropyLoss()
from torchvision import transforms
# from models.drop_loss import DropLoss

# CL_loss = DropLoss(vanilla_loss=criterion).cuda()
# from test_Function import main as test_main  # 重命名导入的main，避免与主函数名冲突
import logging  # 导入 logging 模块
import sys # 导入 sys 模块
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 配置 logging (可选，但比 print 更好)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_total_mi_loss(model: nn.Module):
    """
    Iterates through encoder and decoder blocks to calculate and sum MI loss.
    This also clears the MI statistics in each MoE module.
    """
    total_mi_loss = torch.tensor(0.0, device=next(model.parameters()).device) # Start with 0 on the correct device

    # Iterate through encoder blocks
    if hasattr(model, 'encoder'):
        for module in model.encoder.children():
            if isinstance(module, HeterogeneousTransformerBlock):
                if hasattr(module, 'moe_ffn') and hasattr(module.moe_ffn, 'get_mi_loss_and_clear'):
                    mi_loss_block = module.moe_ffn.get_mi_loss_and_clear()
                    total_mi_loss += mi_loss_block
                    # Optional: Log MI loss per block if needed for debugging
                    # logging.debug(f"MI Loss from encoder block {module}: {mi_loss_block.item()}")

    # Iterate through decoder blocks
    if hasattr(model, 'decode'):
        for module in model.decode.children():
            if isinstance(module, HeterogeneousTransformerBlock):
                 if hasattr(module, 'moe_ffn') and hasattr(module.moe_ffn, 'get_mi_loss_and_clear'):
                    mi_loss_block = module.moe_ffn.get_mi_loss_and_clear()
                    total_mi_loss += mi_loss_block
                    # Optional: Log MI loss per block if needed for debugging
                    # logging.debug(f"MI Loss from decoder block {module}: {mi_loss_block.item()}")

    return total_mi_loss

def setup_environment(args):
    """Initializes logging (console & file), seeds, device, and save path."""

    # ---- Logging Setup Start ----
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()  # 获取 root logger

    # 清除之前可能存在的 handlers，防止重复添加（可选，但有时有用）
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)  # 设置全局最低日志级别

    # Console Handler (StreamHandler) - 输出到控制台
    console_handler = logging.StreamHandler(sys.stdout)  # 输出到标准输出
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # File Handler (FileHandler) - 输出到文件
    # 确保 save_path 存在
    try:  # 增加一点错误处理
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
            # 此时 logger 可能还没完全设置好，或者权限问题，用 print 可能更可靠
            print(f"Created save directory for logging: {args.save_path}")

        log_file = os.path.join(args.save_path, 'training.log')
        file_handler = logging.FileHandler(log_file, mode='a')  # 'a' means append (追加模式)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # 如果创建目录或文件失败，至少在控制台打印错误
        print(f"!!! Error setting up file logging to {args.save_path}: {e}", file=sys.stderr)
        # 可以选择继续（只用控制台日志）或退出
        # exit(1)

    # ---- Logging Setup End ----

    # 现在可以用 logger 记录信息了
    logger.info("Logger configured to output to console and potentially file.")  # 修改措辞

    # --- 后面的代码保持不变 ---
    init_seeds(args.seed)  # init_seeds 内部也会调用 logging.info
    # logger.info(f"Seeds initialized with seed: {args.seed}") # 这句在 init_seeds 里有了

    if args.cuda and not torch.cuda.is_available():
        logger.warning("CUDA specified but not available. Running on CPU.")
        args.cuda = False
        device = torch.device("cpu")
    elif args.cuda:
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU.")

    # 再次检查 save_path 确保存在，因为 logging 可能失败
    try:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
            logger.info(f"Ensured save directory exists: {args.save_path}")
    except Exception as e:
        logger.error(f"Could not create save directory {args.save_path}: {e}. Check permissions.")
        # 根据情况决定是否退出
        # exit(1)

    return device

def has_abnormal_values(tensor, threshold=1e9):
    """检测张量是否包含 NaN、Inf 或绝对值超过阈值的值 (工具函数)"""
    if not isinstance(tensor, torch.Tensor):  # 确保输入是张量
        logging.warning(f"has_abnormal_values received non-tensor input: {type(tensor)}")
        return False
    return torch.isnan(tensor).any() or torch.isinf(tensor).any() or (tensor.abs() > threshold).any()


def init_seeds(seed=0):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if args.cuda: # args 可能尚未定义，但通常在主函数中会检查
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置种子
    # 根据种子确定性设置
    if seed == 0:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True  # 通常为了速度
        cudnn.deterministic = False
    logging.info(f"Seeds initialized with seed: {seed}")


# --- Checkpoint 保存函数 ---
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Saves checkpoint to disk, including model, optimizer, and scheduler state."""
    logging.info(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)

# --- Checkpoint 加载函数 ---
def load_checkpoint(model, optimizer, scheduler, filename="checkpoint.pth.tar"): # Added scheduler argument
    """Loads checkpoint from disk, restoring model, optimizer, and scheduler state."""
    start_epoch = 0
    if os.path.isfile(filename):
        logging.info(f"=> Loading checkpoint '{filename}'")
        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'

        try:
            checkpoint = torch.load(filename, map_location=map_location)
            start_epoch = checkpoint['epoch']

            # Load model state
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                logging.warning(f"Could not load model state_dict strictly: {e}. Attempting non-strict loading.")
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint and optimizer is not None:
                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                 logging.warning("Optimizer state not found in checkpoint or optimizer is None.")


            # --- Load Scheduler State --- <--- NEW
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logging.info("Loaded scheduler state from checkpoint.")
                except Exception as e:
                    logging.error(f"Error loading scheduler state: {e}. Scheduler might reset.")
            elif scheduler is not None:
                 logging.warning("Scheduler state not found in checkpoint, but a scheduler is defined. Scheduler might reset.")


            logging.info(f"=> Loaded checkpoint '{filename}' (resuming from epoch {start_epoch})")

        except Exception as e:
            logging.error(f"Error loading checkpoint {filename}: {e}")
            logging.warning("Could not load checkpoint. Starting from scratch.")
            start_epoch = 0 # Reset epoch

    else:
        logging.info(f"=> No checkpoint found at '{filename}', starting from scratch.")
        start_epoch = 0 # Reset epoch

    # Return updated objects
    return start_epoch, model, optimizer, scheduler # Return scheduler as well



# --- Configuration ---
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch PIAFusion Training (Decoupled)')
    # --- Paths and Saving ---
    parser.add_argument('--dataset_path', default='/home/mimi/桌面/sdd/moe-融合-极端环境/fusion_datasets_1',
                        help='path to dataset')
    parser.add_argument('--save_path', default='pretrained_v501/', help='Directory to save models and checkpoints')
    parser.add_argument('--resume', default=None, type=str, help='Path to latest checkpoint for resuming')
    parser.add_argument('--cls_model_path', default='pretrained/best_cls.pth',
                        help='Path to pre-trained classification model')
    parser.add_argument('--save_freq', default=10, type=int, help='Save checkpoint every N epochs')
    parser.add_argument('--test_start_epoch', default=400, type=int,
                        help='Epoch after which testing function is called')

    # --- Model and Training ---
    parser.add_argument('--arch', default='fusion_model', choices=['fusion_model'], help='Fusion model architecture')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers')  # Increased default
    parser.add_argument('--epochs', default=600, type=int, help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int, help='mini-batch size')  # Increased default
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training')

    # --- Learning Rate Scheduler --- <--- NEW SECTION
    parser.add_argument('--scheduler', default='cosine', type=str,
                        choices=['manual', 'step', 'cosine'],
                        help="Learning rate scheduler type ('manual', 'step', 'cosine')")
    parser.add_argument('--lr_step_size', default=100, type=int,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', default=0.5, type=float,
                        help='Multiplicative factor for StepLR scheduler')
    # T_max for CosineAnnealingLR usually defaults to args.epochs later
    parser.add_argument('--lr_eta_min', default=1e-7, type=float, # Adjusted default lower than initial lr
                        help='Minimum learning rate for CosineAnnealingLR scheduler')

    # --- Loss Weights ---
    parser.add_argument('--w_pix', type=float, default=30.0, help='Weight for pixel loss')
    parser.add_argument('--w_gard', type=float, default=50.0, help='Weight for gradient loss')
    parser.add_argument('--w_seg', type=float, default=40.0, help='Weight for segmentation loss')
    parser.add_argument('--w_color', type=float, default=30.0, help='Weight for color loss')
    parser.add_argument('--w_recon_vis', type=float, default=10.0, help='Weight for reconstruction loss')
    parser.add_argument('--w_recon_inf', type=float, default=10.0, help='Weight for reconstruction loss')
    # --- MoE Specific Loss Weights ---
    parser.add_argument('--w_moe_standard', type=float, default=0.0, help='Weight for Standard MoE auxiliary loss (CV/Z loss)') # Renamed and clarified
    parser.add_argument('--w_MI', type=float, default=0.01, help='Weight for Mutual Information (MI) auxiliary loss') # NEW

    # --- Environment ---
    parser.add_argument('--cuda', action='store_true', default=True, help='use GPU computation')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='do not use GPU computation')

    args = parser.parse_args()
    return args


def load_datasets(args):
    """Loads the training dataset and creates a DataLoader."""
    import torchvision.transforms as transforms
    data_transform = transforms.Compose([
        #transforms.Resize((480, 640)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    train_dataset = MSRSTrainDataset(args.dataset_path, transform = data_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=args.cuda, drop_last=True)  # drop_last can help with batch norm stability
    logging.info(f"Data loaded: {len(train_dataset)} training samples.")
    return train_loader


def build_models(args, device):
    """Builds or loads the fusion and classification models."""
    # Fusion Model
    if args.arch == 'fusion_model':
        model = MSGFusion().to(device)
    else:
        logging.error(f"Unknown architecture: {args.arch}")
        exit(1)

    # Classification Model (Frozen)
    try:
        cls_model = torch.load(args.cls_model_path, map_location=device, weights_only=False)  # Safer loading
        cls_model.eval()
        logging.info(f"Loaded classification model from {args.cls_model_path}")
    except FileNotFoundError:
        logging.error(f"Classification model not found at {args.cls_model_path}. Exiting.")
        exit(1)
    except Exception as e:
        logging.error(f"Error loading classification model: {e}")
        exit(1)

    return model, cls_model


def build_optimizer_and_criterion(model, args, device):
    """Builds the optimizer, loss criterion, and LR scheduler."""
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Loss criterion setup (remains the same)
    cl_loss = FocalLoss().to(device)
    criterion = {
        'l1': nn.L1Loss().to(device),
        'fi': cl_loss
    }

    # --- Learning Rate Scheduler Initialization --- <--- NEW
    scheduler = None # Default to no scheduler (for manual)
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
        logging.info(f"Using StepLR scheduler with step_size={args.lr_step_size}, gamma={args.lr_gamma}")
    elif args.scheduler == 'cosine':
        # T_max is typically the total number of epochs
        T_max = args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=args.lr_eta_min
        )
        logging.info(f"Using CosineAnnealingLR scheduler with T_max={T_max}, eta_min={args.lr_eta_min}")
    elif args.scheduler == 'manual':
        logging.info("Using manual learning rate schedule (defined in update_learning_rate function)")
        # Keep the manual update function if needed for this case
    else:
        logging.warning(f"Unknown scheduler type: {args.scheduler}. Falling back to no scheduler.")

    # Return optimizer, criterion, AND the scheduler
    return optimizer, criterion, scheduler


def calculate_losses(outputs, targets, criterion, loss_weights, device):
    """Calculates all loss components and the total weighted loss, including MoE aux losses."""
    # --- Unpack Model Outputs ---
    # Model now returns: fused_image, seg_res, vi_Reconstruction, ir_Reconstruction, total_aux_losses_dict
    fused_image, seg_res, vi_Reconstruction, ir_Reconstruction, total_aux_losses_dict = outputs
    vis_gt, inf_gt, seg_gt = targets

    fused_image = clamp(fused_image)

    # 1. Reconstruction Loss (Unchanged)
    loss_Reconstruction_vis = criterion['l1'](vi_Reconstruction, vis_gt)
    loss_Reconstruction_inf = criterion['l1'](ir_Reconstruction, inf_gt)

    # 2. Intensity (Pixel) and Gradient Loss (Unchanged)
    f_y, f_cr, f_cb = RGB2YCrCb(fused_image)
    gt_y, gt_cr, gt_cb = RGB2YCrCb(vis_gt)
    if inf_gt.shape[1] == 1:
        inf_y = inf_gt
    else:
        inf_y = RGB2YCrCb(inf_gt)[0]
    loss_pix = criterion['l1'](f_y, torch.max(gt_y, inf_y))
    grad_fy = gradient(f_y)
    grad_gty = gradient(gt_y)
    grad_infy = gradient(inf_y)
    loss_gard = criterion['l1'](grad_fy, torch.max(grad_gty, grad_infy))

    # 3. Segmentation Loss (Unchanged, error handling is good)
    loss_seg = torch.tensor(0.0, device=device)
    if has_abnormal_values(seg_res) or has_abnormal_values(seg_gt.float()):
        logging.warning("Abnormal values in segmentation inputs.")
    else:
        seg_gt = seg_gt.squeeze(1).long()
        if seg_res.shape[0] != seg_gt.shape[0] or seg_res.dim() != 4 or seg_gt.dim() != 3 or seg_res.shape[2:] != seg_gt.shape[1:]:
            logging.error(f"Shape mismatch for segmentation: seg_res {seg_res.shape}, seg_gt {seg_gt.shape}")
        else:
            try:
                loss_seg = criterion['fi'](seg_res, seg_gt)
            except Exception as e:
                logging.error(f"Error calculating segmentation loss: {e}")

    # 4. Color Loss (Unchanged)
    loss_color = criterion['l1'](f_cr, gt_cr) + criterion['l1'](f_cb, gt_cb)

    # --- 5. Fetch and Weight Auxiliary Losses ---
    # Use .get() for safety, providing a default 0 tensor if a key is missing
    loss_moe_standard = total_aux_losses_dict.get('standard_moe_loss', torch.tensor(0.0, device=device))
    loss_moe_mi = total_aux_losses_dict.get('mi_loss', torch.tensor(0.0, device=device))

    # Get weights from loss_weights dictionary
    w_moe_standard = loss_weights.get('moe_standard', 1.0) # Default weight 1 if not found
    w_MI = loss_weights.get('MI', 1.0)                     # Default weight 1 if not found

    # --- 6. Calculate Total Loss ---
    total_loss = (loss_weights['pix'] * loss_pix +
                  loss_weights['gard'] * loss_gard +
                  loss_weights['fi'] * loss_seg +  # Use the key from loss_weights
                  loss_weights['color'] * loss_color +
                  loss_weights['recon_vis'] * loss_Reconstruction_vis +
                  loss_weights['recon_inf'] * loss_Reconstruction_inf +
                  w_moe_standard * loss_moe_standard + # Add weighted standard MoE loss
                  w_MI * loss_moe_mi)                  # Add weighted MI loss

    # --- 7. Prepare Individual Losses for Logging (Unweighted) ---
    individual_losses = {
        'pix': loss_pix.item(),
        'gard': loss_gard.item(),
        'fi': loss_seg.item() if torch.is_tensor(loss_seg) else loss_seg, # Keep existing logic
        'color': loss_color.item(),
        'recon_vis': loss_Reconstruction_vis.item(),
        'recon_inf': loss_Reconstruction_inf.item(),
        'moe_standard': loss_moe_standard.item(), # Log unweighted standard MoE loss
        'moe_mi': loss_moe_mi.item(),             # Log unweighted MI loss
        'total': total_loss.item()                 # Log the final total loss
    }

    # Abnormal value check (Unchanged)
    if has_abnormal_values(total_loss):
        logging.error(f"Abnormal total loss detected: {total_loss.item()}. Setting loss to 0 for this batch.")
        return torch.tensor(0.0, device=device, requires_grad=True), individual_losses

    return total_loss, individual_losses




def update_learning_rate(optimizer, epoch, total_epochs, initial_lr):
    """Applies learning rate decay."""
    if epoch < total_epochs // 2:
        lr = initial_lr
    else:
        lr = initial_lr * (total_epochs - epoch) / (total_epochs - total_epochs // 2)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_one_epoch(epoch, model, cls_model, loader, optimizer, criterion, loss_weights, scaler, device, args, writer):
    """Runs the training loop for one epoch."""
    model.train()
    # Initialize epoch losses dictionary including new MoE keys
    all_loss_keys = list(loss_weights.keys())
    # Ensure the keys used in calculate_losses for aux losses are present
    if 'moe_standard' not in all_loss_keys: all_loss_keys.append('moe_standard')
    if 'MI' not in all_loss_keys: all_loss_keys.append('MI') # Use the key defined in main loss_weights
    # Also include the unweighted keys for accumulation
    if 'moe_standard' not in all_loss_keys: all_loss_keys.append('moe_standard') # For unweighted accumulation
    if 'moe_mi' not in all_loss_keys: all_loss_keys.append('moe_mi')             # For unweighted accumulation
    all_loss_keys.append('total')
    # Use set to avoid duplicates if keys overlap
    epoch_losses = {k: 0.0 for k in set(all_loss_keys + ['moe_standard', 'moe_mi'])} # Ensure unweighted keys exist


    batch_count = 0
    train_tqdm = tqdm(loader, total=len(loader), desc=f"Epoch {epoch}/{args.epochs}")

    for batch_idx, (input_vis, input_inf, vis_gt, Inf_gt, seg_gt, image_clip, name) in enumerate(train_tqdm):
        input_vis = input_vis.to(device)
        input_inf = input_inf.to(device)
        vis_gt = vis_gt.to(device)
        Inf_gt = Inf_gt.to(device)
        seg_gt = seg_gt.to(device)
        image_clip_vi = image_clip[0].to(device)
        image_clip_ir = image_clip[1].to(device)
        input_inf = torch.cat([input_inf] * 3, dim=1) if input_inf.shape[1] == 1 else input_inf

        optimizer.zero_grad()

        with autocast(enabled=args.cuda):
            with torch.no_grad():
                _, feature_vi = cls_model(image_clip_vi)
                _, feature_ir = cls_model(image_clip_ir)
                feature = feature_vi * feature_ir
            # --- Correctly unpack model outputs ---
            outputs = model(input_vis, input_inf, feature)
            # outputs = (fused_image, seg_res, vi_Reconstruction, ir_Reconstruction, total_aux_losses_dict)

            targets = (vis_gt, Inf_gt, seg_gt)
            loss, batch_losses = calculate_losses(outputs, targets, criterion, loss_weights, device)

        if loss.requires_grad:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logging.warning(f"Skipping optimizer step for batch {batch_idx} due to invalid loss.")

        if batch_losses:
            # Accumulate *unweighted* losses reported by calculate_losses
            for k in epoch_losses.keys(): # Iterate through all expected keys in epoch_losses
                # Use .get(k, 0.0) for robustness against missing keys in batch_losses
                epoch_losses[k] += batch_losses.get(k, 0.0)
            batch_count += 1

            # Update tqdm postfix with *weighted* losses for display
            # Get weights safely using .get()
            w_pix = loss_weights.get('pix', 1.0)
            w_gard = loss_weights.get('gard', 1.0)
            w_seg = loss_weights.get('fi', 1.0) # Use 'fi' key as defined in calculate_losses
            w_color = loss_weights.get('color', 1.0)
            w_recon_vis = loss_weights.get('recon_vis', 1.0)
            w_recon_inf = loss_weights.get('recon_inf', 1.0)
            w_moe_std = loss_weights.get('moe_standard', 1.0) # Use the key from main loss_weights
            w_mi = loss_weights.get('MI', 1.0)           # Use the key from main loss_weights

            train_tqdm.set_postfix(
                loss_total=f"{batch_losses.get('total', 0.0):.4f}",
                loss_pix=f"{w_pix * batch_losses.get('pix', 0.0):.4f}",
                loss_gard=f"{w_gard * batch_losses.get('gard', 0.0):.4f}",
                loss_seg=f"{w_seg * batch_losses.get('fi', 0.0):.4f}",
                loss_color=f"{w_color * batch_losses.get('color', 0.0):.4f}",
                loss_recon_vis=f"{w_recon_vis * batch_losses.get('recon_vis', 0.0):.4f}",
                loss_recon_inf=f"{w_recon_inf * batch_losses.get('recon_inf', 0.0):.4f}",
                loss_moe_std=f"{w_moe_std * batch_losses.get('moe_standard', 0.0):.4f}", # Weighted std MoE
                loss_moe_mi=f"{w_mi * batch_losses.get('moe_mi', 0.0):.4f}",        # Weighted MI MoE
                lr=f"{optimizer.param_groups[0]['lr']:.6f}"
            )

        # Image logging (remains the same)
        if batch_idx == 0 and epoch % 5 == 0:
            from torchvision.utils import make_grid
            vis_grid = make_grid(input_vis.clamp(0, 1), nrow=args.batch_size // 2)
            inf_grid = make_grid(input_inf.clamp(0, 1), nrow=args.batch_size // 2)
            # outputs[0] is fused_image from the model's return tuple
            fused_grid = make_grid(outputs[0].clamp(0, 1), nrow=args.batch_size // 2)
            gt_grid = make_grid(vis_gt.clamp(0, 1), nrow=args.batch_size // 2)
            writer.add_image('Images/Visible_Input', vis_grid, global_step=epoch)
            writer.add_image('Images/Infrared_Input', inf_grid, global_step=epoch)
            writer.add_image('Images/Fused_Output', fused_grid, global_step=epoch)
            writer.add_image('Images/Ground_Truth', gt_grid, global_step=epoch)
            logging.info(f"Logged images to TensorBoard for epoch {epoch}")

        # TensorBoard Logging (Scalars per Batch - Unweighted)
        if batch_losses and loss.requires_grad:
            global_step = epoch * len(loader) + batch_idx
            writer.add_scalar('Loss/Total_Batch', batch_losses['total'], global_step)
            writer.add_scalar('Loss/Pixel_Batch', batch_losses['pix'], global_step)
            writer.add_scalar('Loss/Gradient_Batch', batch_losses['gard'], global_step)
            writer.add_scalar('Loss/Segmentation_Batch', batch_losses['fi'], global_step)
            writer.add_scalar('Loss/Color_Batch', batch_losses['color'], global_step)
            writer.add_scalar('Loss/Reconstruction_Batch_vis', batch_losses['recon_vis'], global_step)
            writer.add_scalar('Loss/Reconstruction_Batch_inf', batch_losses['recon_inf'], global_step)
            # --- Log unweighted aux losses ---
            writer.add_scalar('Loss/MoE_Standard_Batch', batch_losses.get('moe_standard', 0.0), global_step)
            writer.add_scalar('Loss/MoE_MI_Batch', batch_losses.get('moe_mi', 0.0), global_step)


    # Calculate average epoch loss (unweighted averages)
    if batch_count > 0:
        # Calculate average *unweighted* losses first
        avg_losses = {k: v / batch_count for k, v in epoch_losses.items()}

        # Log weighted averages to console for summary
        w_pix = loss_weights.get('pix', 1.0)
        w_gard = loss_weights.get('gard', 1.0)
        w_seg = loss_weights.get('fi', 1.0)
        w_color = loss_weights.get('color', 1.0)
        w_recon_vis = loss_weights.get('recon_vis', 1.0)
        w_recon_inf = loss_weights.get('recon_inf', 1.0)
        w_moe_std = loss_weights.get('moe_standard', 1.0)
        w_mi = loss_weights.get('MI', 1.0)

        logging.info(f"Epoch {epoch} Summary | Avg Total Loss: {avg_losses.get('total', 0.0):.4f} | "
                     f"Pix: {w_pix * avg_losses.get('pix', 0.0):.4f}, "
                     f"Gard: {w_gard * avg_losses.get('gard', 0.0):.4f}, "
                     f"Seg: {w_seg * avg_losses.get('fi', 0.0):.4f}, "
                     f"Color: {w_color * avg_losses.get('color', 0.0):.4f}, "
                     f"Recon_vis: {w_recon_vis * avg_losses.get('recon_vis', 0.0):.4f}, "
                     f"Recon_inf: {w_recon_inf * avg_losses.get('recon_inf', 0.0):.4f}, "
                     f"MoE_Std: {w_moe_std * avg_losses.get('moe_standard', 0.0):.4f}, " # Weighted Std MoE Avg
                     f"MoE_MI: {w_mi * avg_losses.get('moe_mi', 0.0):.4f}")            # Weighted MI MoE Avg

        # TensorBoard Logging (Scalars per Epoch - Unweighted Averages)
        writer.add_scalar('Loss/Total_Epoch_Avg', avg_losses.get('total', 0.0), epoch)
        writer.add_scalar('Loss/Pixel_Epoch_Avg', avg_losses.get('pix', 0.0), epoch)
        writer.add_scalar('Loss/Gradient_Epoch_Avg', avg_losses.get('gard', 0.0), epoch)
        writer.add_scalar('Loss/Segmentation_Epoch_Avg', avg_losses.get('fi', 0.0), epoch)
        writer.add_scalar('Loss/Color_Epoch_Avg', avg_losses.get('color', 0.0), epoch)
        writer.add_scalar('Loss/Reconstruction_vis_Epoch_Avg', avg_losses.get('recon_vis', 0.0), epoch)
        writer.add_scalar('Loss/Reconstruction_inf_Epoch_Avg', avg_losses.get('recon_inf', 0.0), epoch) # Corrected key
        # --- Log unweighted average aux losses ---
        writer.add_scalar('Loss/MoE_Standard_Epoch_Avg', avg_losses.get('moe_standard', 0.0), epoch)
        writer.add_scalar('Loss/MoE_MI_Epoch_Avg', avg_losses.get('moe_mi', 0.0), epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

    else:
        logging.warning(f"Epoch {epoch} completed without processing any valid batches.")
        return None # Indicate failure or no data

    # --- REMOVE the epoch-end MI calculation ---
    # total_mi_loss_epoch = get_total_mi_loss(model) # REMOVED - Not needed/functional anymore
    # writer.add_scalar('Loss_Epoch/Mutual_Information', total_mi_loss_epoch.item(), epoch) # REMOVED
    # logging.info(f"Epoch {epoch} | 计算得到的总互信息损失: {total_mi_loss_epoch.item():.4f}") # REMOVED
    # =========================================================== #

    return avg_losses # Return the dictionary of average unweighted losses

# --- Main Execution ---
# --- 主执行函数 ---
# --- 主执行函数 ---
def main():
    args = parse_args()
    device = setup_environment(args)

    # --- Tensorboard Setup (remains the same) ---
    tb_log_dir = os.path.join(args.save_path, 'logs')
    if not os.path.exists(tb_log_dir):
        try:
            os.makedirs(tb_log_dir)
            logging.info(f"已创建 TensorBoard 日志目录: {tb_log_dir}")
        except Exception as e:
            logging.error(f"无法创建日志目录 {tb_log_dir}: {e}")
    writer = SummaryWriter(log_dir=tb_log_dir)
    logging.info(f"TensorBoard 日志将保存到: {tb_log_dir}")

    train_loader = load_datasets(args)
    model, cls_model = build_models(args, device)
    optimizer, criterion, scheduler = build_optimizer_and_criterion(model, args, device)

    # --- Define Loss Weights Dictionary ---
    # Use keys that match those expected by calculate_losses
    loss_weights = {
        'pix': args.w_pix,
        'gard': args.w_gard,
        'fi': args.w_seg,       # Matches the key used for segmentation criterion
        'color': args.w_color,
        'recon_vis': args.w_recon_vis,
        'recon_inf': args.w_recon_inf,
        'moe_standard': args.w_moe_standard, # Weight for standard MoE loss
        'MI': args.w_MI                      # Weight for MI loss
    }
    logging.info(f"Loss weights: {loss_weights}") # Log the weights being used
    # -------------------------------------------

    scaler = GradScaler(enabled=args.cuda)

    # --- Checkpoint Loading (remains the same) ---
    start_epoch = 0
    checkpoint_path = args.resume if args.resume else os.path.join(args.save_path, 'latest_checkpoint.pth.tar')
    if os.path.exists(checkpoint_path):
        start_epoch, model, optimizer, scheduler = load_checkpoint(model, optimizer, scheduler, checkpoint_path)

    logging.info(f"从 epoch {start_epoch} 开始训练")

    # --- Epoch Loop (remains the same, train_one_epoch call is correct) ---
    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch} | 开始，学习率: {current_lr:.7f}")

        avg_epoch_losses = train_one_epoch(epoch, model, cls_model, train_loader, optimizer, criterion, loss_weights,
                                           scaler, device, args, writer)

        # Scheduler Step (remains the same)
        if scheduler is not None and args.scheduler != 'manual':
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch} | Scheduler 步进完成。下一 Epoch 的新学习率: {new_lr:.7f}")

        # Checkpoint Saving (remains the same)
        if avg_epoch_losses is not None:
            latest_checkpoint_filename = os.path.join(args.save_path, 'latest_checkpoint.pth.tar')
            checkpoint_state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if args.cuda else None,
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            }
            save_checkpoint(checkpoint_state, latest_checkpoint_filename)

            if (epoch + 1) % args.save_freq == 0:
                epoch_checkpoint_filename = os.path.join(args.save_path, f'checkpoint_epoch_{epoch + 1}.pth.tar')
                save_checkpoint(checkpoint_state, epoch_checkpoint_filename)
    # --- Epoch Loop End ---

    writer.close()
    logging.info("TensorBoard writer 已关闭。")
    logging.info("训练完成。")


if __name__ == '__main__':
    main()



# if __name__ == '__main__':
#     # Set CUDA_VISIBLE_DEVICES *before* anything else if needed, or rely on system environment
#     # os.environ['CUDA_VISIBLE_DEVICES'] = "0" # Better set outside the script
#     main()

