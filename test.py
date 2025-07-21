"""测试融合网络"""
import argparse
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data_loader.msrs_data import MSRS_data
from models.common import clamp
from models.fusion_model import FM
from models.cls_model import LoraCLIP
from models.drop_loss import convert_to_pixel_mask
def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

# --- Checkpoint 加载函数 ---
def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    """加载检查点"""
    if os.path.isfile(filename):
        logging.info(f"=> Loading checkpoint '{filename}'")
        # checkpoint = torch.load(filename) # 加载到CPU，避免GPU OOM
        # 更加安全的加载方式，指定 map_location
        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'

        try:
            checkpoint = torch.load(filename, map_location=map_location)
            start_epoch = checkpoint['epoch']
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                logging.warning(f"Could not load model state_dict strictly: {e}. Attempting non-strict loading.")
                # 尝试非严格加载（如果模型结构有微小变化）
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 如果学习率是动态调整的，可能也需要加载调度器状态或直接使用epoch计算
            # current_lr = checkpoint.get('lr', args.lr) # 获取保存的学习率（如果保存了）
            logging.info(f"=> Loaded checkpoint '{filename}' (epoch {start_epoch})")
            return start_epoch, model, optimizer
        except Exception as e:
            logging.error(f"Error loading checkpoint {filename}: {e}")
            logging.warning("Could not load checkpoint. Starting from scratch.")
            return 0, model, optimizer  # 返回初始 epoch

    else:
        logging.info(f"=> No checkpoint found at '{filename}', starting from scratch.")
        return 0, model, optimizer  # 返回初始 epoch

if __name__ == '__main__':
    dataset = 'MSRS'
    parser = argparse.ArgumentParser(description='PyTorch PIAFusion')
    parser.add_argument('--dataset_path', metavar='DIR', default=f'/home/gpu/wzd/code/Fusion/Dataset/msrsda/MSRStest/test',
                        help='path to dataset (default: imagenet)')  # 测试数据存放位置
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
                        choices=['fusion_model'])
    parser.add_argument('--save_path', default=f'/home/gpu/wzd/code/Fusion/Dataset/msrsda/testset/{dataset}/fi')  # 融合结果存放位置
    parser.add_argument('--save_path_seg', default=f'/home/gpu/wzd/code/Fusion/Dataset/msrsda/testset/{dataset}/seg')  # 融合结果存放位置
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()

    init_seeds(args.seed)

    test_dataset = MSRS_data(args.dataset_path, phase='test')
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.save_path_seg):
        os.makedirs(args.save_path_seg)

    # 如果是融合网络
    if args.arch == 'fusion_model':
        model = FM().cuda()
        # model = model.cuda()
        checkpoint_path = '/home/gpu/wzd/code/Fusion/code/pretrained_decoupled4/checkpoint_epoch_105.pth.tar'
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # model.load_state_dict(torch.load('/home/gpu/wzd/code/Fusion/code/fusion.pth',weights_only=False))
        model.eval()
        #cls_model = LoraCLIP(num_classes=4)
        cls_model = torch.load('/home/gpu/wzd/code/Fusion/code/best_cls.pth',weights_only=False)
        cls_model.cuda()
        cls_model.eval()
        test_tqdm = tqdm(test_loader, total=len(test_loader))
        with torch.no_grad():
            for vis_image, inf_image, vis_image_clip, name in test_tqdm:
                vis_image = vis_image.cuda()
                inf_image = inf_image.cuda()
                inf_image = torch.cat([inf_image] * 3, dim=1)
                vis_image_clip = vis_image_clip.cuda()
                _, feature = cls_model(vis_image_clip)

                fused_image, seg_res, _ = model(vis_image, inf_image, feature)
                fused_image = clamp(fused_image)
                # print(fused_image.shape) # torch.Size([1, 3, 480, 640])
                # print('seg_res.shape',seg_res.shape) # torch.Size([1, 3, 480, 640])
                # print("fused_image Range: [{:.4f}, {:.4f}]".format(fused_image.min().item(), fused_image.max().item()))
                # fused_image Range: [0.0000, 0.4953]
                # print("seg_res Range: [{:.4f}, {:.4f}]".format(seg_res.min(), seg_res.max())) # seg_res Range: [-2.7985, 5.1082]
                seg_res = convert_to_pixel_mask(seg_res)
                # print('aaaa',seg_res.shape)
                # print(seg_res.unique())
                # print("seg_res Range: [{:.4f}, {:.4f}]".format(seg_res.min(), seg_res.max()))
                rgb_fused_image = transforms.ToPILImage()(fused_image[0])
                # print("\nPIL方法:")
                # for channel in range(3):  # RGB 三个通道
                #     channel_min, channel_max = rgb_fused_image.split()[channel].getextrema()
                #     print(f"通道 {channel} 的最小值: {channel_min}, 最大值: {channel_max}")
                rgb_fused_image.save(f'{args.save_path}/{name[0]}')

                seg_res = transforms.ToPILImage()(seg_res.to(torch.uint8))
                seg_res.save(f'{args.save_path_seg}/{name[0]}')