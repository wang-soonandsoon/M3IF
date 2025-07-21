import torch
from models.transformer import MTransformerBlock, MoEFeedForward
# from common import SimpleChannelAttentionFusion
import torch.nn as nn
import torch.nn.functional as F

class IndependentSpatialGatedFusionBlock(nn.Module):
    def __init__(self, channels):
        super(IndependentSpatialGatedFusionBlock, self).__init__()
        reduction = max(4, channels // 8)
        self.channels = channels
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            # --- Change Start ---
            # 输出 2*C 通道
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False),
            # --- Change End ---
            nn.Sigmoid()
        )
        # 可选: 增加一个最终的融合卷积
        # self.final_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, v, i):
        combined = torch.cat([v, i], dim=1)
        gates = self.attention(combined) # (B, 2*C, H, W)
        # --- Change Start ---
        gate_v, gate_i = torch.split(gates,self.channels, dim=1) # (B, C, H, W) each
        gated_fusion = gate_v * v + gate_i * i
        # gated_fusion = self.final_conv(gated_fusion) # 可选
        # --- Change End ---
        return gated_fusion

class GatedFusionBlock(nn.Module):
    """
    优化版本：移除有问题的熵计算，仅使用通道注意力。
    VLGI = Very Light General Innovation (Simplified)
    """
    def __init__(self, channels):
        super(GatedFusionBlock, self).__init__()
        reduction = max(4, channels // 8)  # 动态缩减比例
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # 输入通道数为 channels * 2
            nn.Conv2d(channels * 2, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            # 输出通道数为 channels，匹配 gate * v 的需求
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, v, i):
        # 1. 计算通道注意力门控
        combined = torch.cat([v, i], dim=1)
        gate = self.attention(combined) # shape: (B, C, 1, 1)
        # # 动态均值加权 (使用 Softmax) (创新点)
        # v_mean = torch.mean(v, dim=[2, 3], keepdim=True) # Shape: (B, C, 1, 1)
        # i_mean = torch.mean(i, dim=[2, 3], keepdim=True) # Shape: (B, C, 1, 1)
        #
        # # 在每个通道内，对 v_mean 和 i_mean 应用 Softmax
        # # 我们需要的是 v 的权重，所以取 Softmax 输出的第一个元素
        # # 为了应用 softmax，我们暂时将它们看作是 logits
        # # (B, C, 1, 1) -> (B, C, 2, 1) where dim=2 holds [v_mean, i_mean]
        # means_for_softmax = torch.stack([v_mean, i_mean], dim=2).squeeze(-1) # Shape: (B, C, 2, 1) -> (B, C, 2)
        #
        # # 应用 Softmax (可选温度缩放)
        # # dim=2 表示在 [v_mean, i_mean] 之间进行归一化
        # softmax_weights = F.softmax(means_for_softmax / self.temperature, dim=2) # Shape: (B, C, 2)
        #
        # # 取出 v 对应的权重，并恢复维度 (B, C, 1, 1)
        # dynamic_weight = softmax_weights[:, :, 0].unsqueeze(-1).unsqueeze(-1) # Shape: (B, C, 1, 1)
        #
        # # 权重组合: 注意力权重 * 基于 Softmax 均值的动态权重
        # gate = gate * dynamic_weight
        # 2. 直接使用通道注意力进行门控融合
        # gate 会自动广播到 (B, C, H, W)
        gated_fusion = gate * v + (1 - gate) * i
        return gated_fusion



# 下采样 分辨率2倍，通道4倍
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)  # 降采样
        )

    def forward(self, x):
        return self.body(x)

# 上采样 分辨率2， 通道4
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
# 这个版本模型使用了1次编码器，两次解码器完成重建
from .MItransformer import HeterogeneousTransformerBlock
class encoder(nn.Module):
    def __init__(self, d_text=512):
        super(encoder, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, stride=1, padding=1, kernel_size=3)
        self.moe_vit_block1 = HeterogeneousTransformerBlock(dim=16, d_text=d_text, num_experts=8, top_k=2)
        self.down1 = Downsample(n_feat=16)
        self.moe_vit_block2 = HeterogeneousTransformerBlock(dim=32, d_text=d_text, num_experts=8, top_k=2)
        self.down2 = Downsample(n_feat=32)
        self.moe_vit_block3 = HeterogeneousTransformerBlock(dim=64, d_text=d_text, num_experts=8, top_k=2)
        self.down3 = Downsample(n_feat=64)
        self.moe_vit_block4 = HeterogeneousTransformerBlock(dim=128, d_text=d_text, num_experts=8, top_k=2)
        self.down4 = Downsample(n_feat=128)

    def forward(self, x, route_feature, task_id):
        # Initialize aggregated aux losses dictionary
        device = x.device
        total_aux_losses = {'standard_moe_loss': torch.tensor(0.0, device=device),
                            'mi_loss': torch.tensor(0.0, device=device)}

        x1_ = self.conv(x)
        x1_moe, aux_losses1 = self.moe_vit_block1(x1_, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses1.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses1.get('mi_loss', 0.0)
        x2_ = self.down1(x1_moe)

        x2_moe, aux_losses2 = self.moe_vit_block2(x2_, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses2.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses2.get('mi_loss', 0.0)
        x3_ = self.down2(x2_moe)

        x3_moe, aux_losses3 = self.moe_vit_block3(x3_, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses3.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses3.get('mi_loss', 0.0)
        x4_ = self.down3(x3_moe)

        x4_moe, aux_losses4 = self.moe_vit_block4(x4_, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses4.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses4.get('mi_loss', 0.0)
        x5_ = self.down4(x4_moe) # Bottleneck

        # Return features and the dictionary of aggregated aux losses
        return x1_, x2_, x3_, x4_, x5_, total_aux_losses


class decoder(nn.Module):
    def __init__(self, d_text=512):
        super(decoder, self).__init__()
        self.moe_vit_block1 = HeterogeneousTransformerBlock(dim=256, d_text=d_text, num_experts=8, top_k=2)
        self.up1 = Upsample(n_feat=256)
        self.moe_vit_block2 = HeterogeneousTransformerBlock(dim=128, d_text=d_text, num_experts=8, top_k=2)
        self.up2 = Upsample(n_feat=128)
        self.moe_vit_block3 = HeterogeneousTransformerBlock(dim=64, d_text=d_text, num_experts=8, top_k=2)
        self.up3 = Upsample(n_feat=64)
        self.moe_vit_block4 = HeterogeneousTransformerBlock(dim=32, d_text=d_text, num_experts=8, top_k=2)
        self.up4 = Upsample(n_feat=32)

    def forward(self, x1, x2, x3, x4, x5, route_feature, task_id):
        # Initialize aggregated aux losses dictionary
        device = x5.device # Use device from an input tensor
        total_aux_losses = {'standard_moe_loss': torch.tensor(0.0, device=device),
                            'mi_loss': torch.tensor(0.0, device=device)}

        y5_moe, aux_losses1 = self.moe_vit_block1(x5, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses1.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses1.get('mi_loss', 0.0)
        y4_up = self.up1(y5_moe)
        y4 = y4_up + x4

        y4_moe, aux_losses2 = self.moe_vit_block2(y4, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses2.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses2.get('mi_loss', 0.0)
        y3_up = self.up2(y4_moe)
        y3 = y3_up + x3

        y3_moe, aux_losses3 = self.moe_vit_block3(y3, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses3.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses3.get('mi_loss', 0.0)
        y2_up = self.up3(y3_moe)
        y2 = y2_up + x2

        y2_moe, aux_losses4 = self.moe_vit_block4(y2, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses4.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses4.get('mi_loss', 0.0)
        y1_up = self.up4(y2_moe)
        y1 = y1_up + x1

        # Return final feature map and the dictionary of aggregated aux losses
        return y1, total_aux_losses


class MSGFusion(nn.Module):
    def __init__(self, n_class=9, d_text=512):
        super(MSGFusion, self).__init__()
        self.n_class = n_class
        self.encoder = encoder(d_text=d_text)
        self.decode = decoder(d_text=d_text)

        # Gated Fusion Blocks (remain the same)
        self.fusion_block1 = IndependentSpatialGatedFusionBlock(channels=16)
        self.fusion_block2 = IndependentSpatialGatedFusionBlock(channels=32)
        self.fusion_block3 = IndependentSpatialGatedFusionBlock(channels=64)
        self.fusion_block4 = IndependentSpatialGatedFusionBlock(channels=128)
        self.fusion_block5 = IndependentSpatialGatedFusionBlock(channels=256)

        # Task-Specific Heads (remain the same)
        self.fusion_head = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.seg_attention_conv = nn.Conv2d(in_channels=16 * 2, out_channels= 2, kernel_size=1, stride=1, padding=0)
        intermediate_channels_seg = 32
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=intermediate_channels_seg, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels_seg, self.n_class, kernel_size=1)
        )
        self.recon_head_vis = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.recon_head_inf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)

    def _reconstruct_and_segment(self, fused_image, route_feature):
        # Initialize aggregated aux losses dictionary for this path
        device = fused_image.device
        recon_seg_aux_losses = {'standard_moe_loss': torch.tensor(0.0, device=device),
                                'mi_loss': torch.tensor(0.0, device=device)}

        # 1. Encode the fused image (task_id=1 for auxiliary path gating)
        # Encoder output features: rf1, rf2, rf3, rf4, rf5
        # Encoder aux losses: aux_losses_enc (dictionary potentially containing 'standard_moe_loss', 'mi_loss')
        rf1, rf2, rf3, rf4, rf5, aux_losses_enc = self.encoder(fused_image, route_feature, task_id=1)

        # Aggregate losses from encoder
        recon_seg_aux_losses['standard_moe_loss'] += aux_losses_enc.get('standard_moe_loss', torch.tensor(0.0, device=device))
        recon_seg_aux_losses['mi_loss'] += aux_losses_enc.get('mi_loss', torch.tensor(0.0, device=device))

        # 2. Decode features ONCE (task_id=1 for auxiliary path gating)
        # Decoder output features: y1_decoded_features (e.g., shape [B, C, H, W])
        # Decoder aux losses: aux_losses_dec (dictionary potentially containing 'standard_moe_loss', 'mi_loss')
        y1_decoded_features, aux_losses_dec = self.decode(rf1, rf2, rf3, rf4, rf5, route_feature, task_id=1)

        # Aggregate losses from decoder
        recon_seg_aux_losses['standard_moe_loss'] += aux_losses_dec.get('standard_moe_loss', torch.tensor(0.0, device=device))
        recon_seg_aux_losses['mi_loss'] += aux_losses_dec.get('mi_loss', torch.tensor(0.0, device=device))

        # 3. Apply Reconstruction Heads to the single decoded feature map
        bw_vi = self.recon_head_vis(y1_decoded_features) # e.g., Output: [B, 3, H, W]
        bw_ir = self.recon_head_inf(y1_decoded_features) # e.g., Output: [B, 3, H, W]

        # 4. Apply Segmentation Head DIRECTLY to the single decoded feature map
        # This assumes seg_head is designed to take the output of the decoder directly.
        # If the attention mechanism from version 1 is essential, adjustments are needed here
        # or in the definition of seg_head / seg_attention_conv.
        seg_res = self.seg_head(y1_decoded_features) # e.g., Output: [B, n_class, H, W]

        # Return the reconstruction/segmentation outputs and the aggregated aux losses dictionary
        return bw_vi, bw_ir, seg_res, recon_seg_aux_losses

    def forward(self, vi, ir, route_feature):
        device = vi.device
        # Initialize aggregated aux losses dictionary for the main fusion path
        fusion_aux_losses = {'standard_moe_loss': torch.tensor(0.0, device=device),
                             'mi_loss': torch.tensor(0.0, device=device)}

        # --- 1. Encoding ---
        v1, v2, v3, v4, v5, aux_losses_enc_v = self.encoder(vi, route_feature, task_id=0)
        fusion_aux_losses['standard_moe_loss'] += aux_losses_enc_v.get('standard_moe_loss', 0.0)
        fusion_aux_losses['mi_loss'] += aux_losses_enc_v.get('mi_loss', 0.0)

        i1, i2, i3, i4, i5, aux_losses_enc_i = self.encoder(ir, route_feature, task_id=0)
        fusion_aux_losses['standard_moe_loss'] += aux_losses_enc_i.get('standard_moe_loss', 0.0)
        fusion_aux_losses['mi_loss'] += aux_losses_enc_i.get('mi_loss', 0.0)

        # --- 2. Feature Fusion ---
        f1 = self.fusion_block1(v1, i1)
        f2 = self.fusion_block2(v2, i2)
        f3 = self.fusion_block3(v3, i3)
        f4 = self.fusion_block4(v4, i4)
        f5 = self.fusion_block5(v5, i5)

        # --- 3. Fusion Feature Decoding ---
        y1_fused, aux_losses_dec_fus = self.decode(f1, f2, f3, f4, f5, route_feature, task_id=0)
        fusion_aux_losses['standard_moe_loss'] += aux_losses_dec_fus.get('standard_moe_loss', 0.0)
        fusion_aux_losses['mi_loss'] += aux_losses_dec_fus.get('mi_loss', 0.0)

        # --- 4. Generate Fusion Output ---
        fusion_res = self.fusion_head(y1_fused)

        # --- 5. Reconstruction & Segmentation (Training Only) ---
        if self.training:
            # Call the helper function which now returns aux losses for its path
            bw_vi, bw_ir, seg_res, recon_seg_aux_losses = self._reconstruct_and_segment(
                fusion_res, # Keep attached, gradients can flow
                route_feature
            )

            # Combine aux losses from both paths
            total_aux_losses = {}
            total_aux_losses['standard_moe_loss'] = fusion_aux_losses.get('standard_moe_loss', 0.0) + \
                                                    recon_seg_aux_losses.get('standard_moe_loss', 0.0)
            total_aux_losses['mi_loss'] = fusion_aux_losses.get('mi_loss', 0.0) + \
                                          recon_seg_aux_losses.get('mi_loss', 0.0)

            # Return all outputs and the final combined aux losses dictionary
            return fusion_res, seg_res, bw_vi, bw_ir, total_aux_losses
        else:
            # --- Evaluation Mode ---
            # Only return fusion result and the aux losses from the fusion path
            # Other outputs are None
            return fusion_res, None, None, None, fusion_aux_losses



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dummy input tensors on the target device
    image_vis = torch.rand(1, 3, 256, 256, device=device)  # Visible image
    image_ir = torch.rand(1, 3, 256, 256, device=device)  # Infrared image
    route_feature = torch.rand(1, 512, device=device)  # Route feature

    # Instantiate the model and move it to the device
    # Ensure n_class and d_text match your model's initialization if needed
    model = MSGFusion(n_class=9, d_text=512).to(device)  # Replace Dummy with actual MSGFusion

    # --- Test in Training Mode ---
    print("\n--- Testing in Training Mode (model.train()) ---")
    model.train()  # Set the model to training mode

    # Perform the forward pass
    outputs_train = model(image_vis, image_ir, route_feature)

    # Unpack the outputs explicitly
    fusion_res_train, seg_res_train, bw_vi_train, bw_ir_train, total_aux_losses_train = outputs_train

    # Print shapes of the tensor outputs
    print("Output Shapes:")
    print(f"  Fusion Result (fusion_res):      {fusion_res_train.shape}")
    print(f"  Segmentation Result (seg_res):   {seg_res_train.shape}")
    print(f"  Recon Visible (bw_vi):         {bw_vi_train.shape}")
    print(f"  Recon Infrared (bw_ir):        {bw_ir_train.shape}")

    # Print the auxiliary losses dictionary and its contents
    print("\nAuxiliary Losses Dictionary (total_aux_losses):")
    print(f"  Dictionary: {total_aux_losses_train}")
    # Access and print individual loss values (use .item() to get scalar)
    if isinstance(total_aux_losses_train, dict):
        std_loss = total_aux_losses_train.get('standard_moe_loss', torch.tensor(float('nan'))).item()
        mi_loss = total_aux_losses_train.get('mi_loss', torch.tensor(float('nan'))).item()
        print(f"  Standard MoE Loss value: {std_loss:.4f}")
        print(f"  MI Loss value:           {mi_loss:.4f}")
    else:
        print("  Warning: Expected aux_losses to be a dict, but got:", type(total_aux_losses_train))

    # --- Test in Evaluation Mode ---
    print("\n--- Testing in Evaluation Mode (model.eval()) ---")
    model.eval()  # Set the model to evaluation mode

    # Use torch.no_grad() context manager for evaluation
    with torch.no_grad():
        outputs_eval = model(image_vis, image_ir, route_feature)

    # Unpack the outputs explicitly for eval mode
    fusion_res_eval, seg_res_eval, bw_vi_eval, bw_ir_eval, fusion_aux_losses_eval = outputs_eval

    # Print shapes and values
    print("Output Values/Shapes:")
    print(f"  Fusion Result (fusion_res):      {fusion_res_eval.shape if fusion_res_eval is not None else 'None'}")
    print(f"  Segmentation Result (seg_res):   {seg_res_eval}")  # Expected: None
    print(f"  Recon Visible (bw_vi):         {bw_vi_eval}")  # Expected: None
    print(f"  Recon Infrared (bw_ir):        {bw_ir_eval}")  # Expected: None

    # Print the auxiliary losses dictionary (only fusion path losses expected)
    print("\nAuxiliary Losses Dictionary (fusion_aux_losses):")
    print(f"  Dictionary: {fusion_aux_losses_eval}")
    if isinstance(fusion_aux_losses_eval, dict):
        std_loss_eval = fusion_aux_losses_eval.get('standard_moe_loss', torch.tensor(float('nan'))).item()
        mi_loss_eval = fusion_aux_losses_eval.get('mi_loss', torch.tensor(float('nan'))).item()
        print(f"  Standard MoE Loss value: {std_loss_eval:.4f}")
        print(f"  MI Loss value:           {mi_loss_eval:.4f}")
    else:
        print("  Warning: Expected aux_losses to be a dict, but got:", type(fusion_aux_losses_eval))

    print("\nTest Completed.")


