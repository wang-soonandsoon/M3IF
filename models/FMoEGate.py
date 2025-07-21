import logging
import warnings

import torch
import torch.nn as nn
import math
from torch.distributions.normal import Normal
import numpy as np

# --- 假设你已经复制粘贴了 MoEGate 类的完整代码在这里 ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal # 从 PyTorch 的概率分布库中导入正态分布
import math
import numpy as np
from typing import Dict, Optional, Tuple
class GlobalAvgPool(nn.Module):
    def forward(self, x):
        # 假设 x 是 (B, C, H, W)
        return F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1) # (B, C)

class GatingFusionMoEGate(nn.Module):
    """
    MoE 门控网络，使用门控融合 (Gating Fusion / 后期融合) 策略。
    分别计算基于输入 x 和外部 Degraded_feature 的门控 logits，然后融合它们。
    """
    def __init__(self, d_x: int, d_text: int, M: int = 4, K: int = 1, noisy_gating: bool = True):
        """
        Args:
            d_x (int): 输入 x 的通道维度 C。
            d_text (int): Degraded_feature 的维度。
            M (int): 专家总数。
            K (int): Top-K 选择。
            noisy_gating (bool): 是否使用噪声门控。
        """
        super().__init__()
        self.M = M
        self.k = K
        self.noisy_gating = noisy_gating

        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # 处理 x

        # --- 两组独立的门控权重和噪声权重 ---
        # 基于 x_pooled (维度 d_x)
        self.w_gate_x = nn.Parameter(torch.zeros(d_x, M), requires_grad=True)
        self.w_noise_x = nn.Parameter(torch.zeros(d_x, M), requires_grad=True)
        # 基于 Degraded_feature (维度 d_text)
        self.w_gate_text = nn.Parameter(torch.zeros(d_text, M), requires_grad=True)
        self.w_noise_text = nn.Parameter(torch.zeros(d_text, M), requires_grad=True)
        # 喵呜~ 现在有两套小本本记权重啦！一套给 x，一套给 Degraded_feature！

        # --- 可学习的融合权重 (推荐！) ---
        # 使用一个标量参数，通过 sigmoid 映射到 (0, 1) 作为 x 的权重 alpha。
        # text 的权重可以是 1 - alpha，或者另一个独立的学习参数。
        # 这里我们用一个参数控制两者比例。
        # 初始值设为 0.0，这样 sigmoid(0.0) = 0.5，初始时两者权重相等。
        self.logit_weight_param = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # 这个小参数决定了 x 的信号和 Degraded_feature 的信号哪个更重要一点点~

        # --- MoEGate 的其他组件 ---
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.M

    def forward(self, x: torch.Tensor, Degraded_feature: torch.Tensor, loss_coef=1e-2, noise_epsilon=1e-2):
        """
        接收 x 和 Degraded_feature 进行门控计算。
        """
        batch_size = x.shape[0]

        # 1. 提取 x 的全局特征
        x_pooled = self.gap(x).view(batch_size, -1) # 形状: [B, d_x]

        # 2. 分别计算两组 logits 和噪声标准差
        clean_logits_x = x_pooled @ self.w_gate_x           # [B, M]
        clean_logits_text = Degraded_feature @ self.w_gate_text # [B, M]

        if self.noisy_gating and self.training:
            # 计算 x 的噪声部分
            raw_noise_stddev_x = x_pooled @ self.w_noise_x
            noise_stddev_x = self.softplus(raw_noise_stddev_x) + noise_epsilon
            noisy_logits_x = clean_logits_x + (torch.randn_like(clean_logits_x) * noise_stddev_x)

            # 计算 Degraded_feature 的噪声部分
            raw_noise_stddev_text = Degraded_feature @ self.w_noise_text
            noise_stddev_text = self.softplus(raw_noise_stddev_text) + noise_epsilon
            noisy_logits_text = clean_logits_text + (torch.randn_like(clean_logits_text) * noise_stddev_text)

            # --- 融合噪声标准差 (近似处理) ---
            # 精确融合噪声标准差比较复杂。一种近似方法是假设噪声独立，
            # 加权后的总方差 = w_x^2 * var_x + w_t^2 * var_t
            # 总标准差 = sqrt(w_x^2 * std_x^2 + w_t^2 * std_t^2)
            alpha = torch.sigmoid(self.logit_weight_param) # x 的权重 (0, 1)
            beta = 1.0 - alpha                          # text 的权重 (假设和为 1)
            # 或者用另一个独立参数: beta = torch.sigmoid(another_param)

            # 注意：这里的 alpha 和 beta 是标量，会广播
            final_noise_stddev_squared = alpha.pow(2) * noise_stddev_x.pow(2) + beta.pow(2) * noise_stddev_text.pow(2)
            final_noise_stddev = torch.sqrt(final_noise_stddev_squared + 1e-8) # 加一点 epsilon 防 sqrt(0)

        else:
            # 推理或不使用噪声时
            noisy_logits_x = clean_logits_x
            noisy_logits_text = clean_logits_text
            # 在计算 _prob_in_top_k 时，如果 noisy_gating=True 但处于 eval 模式，
            # 理论上也需要 stddev，但此时 stddev 应为 0 或 epsilon。
            # 为了简化且在推理时不需要，这里直接设为 None 或 0。
            # 如果 _prob_in_top_k 中需要非 None 的 stddev，则设为 epsilon。
            final_noise_stddev = torch.full_like(clean_logits_x, noise_epsilon) # 或者 None，取决于 _prob_in_top_k 实现

        # 3. 融合 Logits (加权求和)
        if self.training: # 只在训练时计算 alpha, beta 用于融合 noisy logits
             alpha = torch.sigmoid(self.logit_weight_param)
             beta = 1.0 - alpha
        else: # 推理时可以用固定的权重，或者也用学习到的权重
             with torch.no_grad(): # 确保推理时不计算梯度
                 alpha = torch.sigmoid(self.logit_weight_param)
                 beta = 1.0 - alpha

        # 融合 clean logits (用于 _prob_in_top_k)
        final_clean_logits = alpha * clean_logits_x + beta * clean_logits_text
        # 融合 noisy logits (用于 TopK 选择)
        final_noisy_logits = alpha * noisy_logits_x + beta * noisy_logits_text

        logits = final_noisy_logits # 使用融合后的 noisy logits 进行 TopK 选择

        # --- 4. 后续计算与原 MoEGate 相同 ---
        # Top-K 选择、创建稀疏 gates
        num_experts_to_consider = min(self.k + 1, self.M)
        # 使用融合后的 noisy logits 进行 TopK
        top_logits, top_indices = logits.topk(num_experts_to_consider, dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits) # 对融合后的 topk noisy logits 做 softmax
        zeros = torch.zeros_like(logits, requires_grad=False)
        gates = zeros.scatter(dim=1, index=top_k_indices, src=top_k_gates)

        # 计算 moe_loss
        importance = gates.sum(dim=0)
        if self.noisy_gating and self.k < self.M and self.training:
            # --- 关键：使用融合后的 clean_logits, noisy_logits, noise_stddev ---
            load = (self._prob_in_top_k(final_clean_logits, final_noisy_logits, final_noise_stddev, top_logits)).sum(dim=0)
        else:
            load = self._gates_to_load(gates)

        loss = self.cv_squared(importance) + self.cv_squared(load)
        moe_loss = loss * loss_coef

        return gates, moe_loss

    # --- 需要包含 _gates_to_load, cv_squared, _prob_in_top_k 方法 ---
    # --- 这些辅助方法不需要修改，可以直接复制粘贴过来 ---
    def _gates_to_load(self, gates: torch.Tensor):
        # ... (代码同上) ...
        return (gates > 0).sum(dim=0)

    def cv_squared(self, x: torch.Tensor):
        # ... (代码同上) ...
        eps = 1e-10
        if x.shape[0] <= 1:
            return torch.tensor(0.0, device=x.device, dtype=torch.float32)
        variance = x.float().var()
        mean_squared = x.float().mean() ** 2
        return variance / (mean_squared + eps)

    def _prob_in_top_k(self, clean_values: torch.Tensor, noisy_values: torch.Tensor, noise_stddev: torch.Tensor, noisy_top_values: torch.Tensor):
        # ... (代码同上，注意输入的 clean_values, noisy_values, noise_stddev 都是融合后的) ...
        # --- 检查 noise_stddev 是否有效 ---
        if noise_stddev is None or (noise_stddev == 0).all():
             logging.warning("`_prob_in_top_k` received zero or None noise_stddev. Load balancing might be inaccurate in eval.")
             # 可以返回一个近似值，例如基于 clean_values 的 topk 结果
             # 这里简单返回 0，因为在推理时通常不关心这个 loss
             return torch.zeros_like(clean_values)

        batch_size = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch_size, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)

        # --- 处理可能的除零错误 ---
        # 确保 noise_stddev 不为零，添加一个很小的 epsilon
        noise_stddev_safe = noise_stddev + 1e-8

        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev_safe)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev_safe)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

# class FusedMoEGate(nn.Module):
#     """
#     融合输入 x 特征和外部 Degraded_feature 的 MoE 门控网络。
#     """
#
#     def __init__(self, d_x: int, d_text: int, M: int = 4, K: int = 1, noisy_gating: bool = True, fusion_dim: int = None):
#         """
#         Args:
#             d_x (int): 输入 x 的通道维度 C。
#             d_text (int): Degraded_feature 的维度。
#             M (int): 专家总数。
#             K (int): Top-K 选择。
#             noisy_gating (bool): 是否使用噪声门控。
#             fusion_dim (int, optional): 融合后特征的维度。如果为 None，则为 d_x + d_text。
#                                          也可以先将两者投影到同一维度再融合。
#         """
#         super().__init__()
#         self.M = M
#         self.k = K
#         self.noisy_gating = noisy_gating
#
#         self.gap = nn.AdaptiveAvgPool2d((1, 1)) # 处理 x
#
#         # --- 融合层 (示例：简单拼接 + 线性层) ---
#         # 1. 获取 x 的全局特征 (维度 d_x)
#         # 2. 获取 Degraded_feature (维度 d_text)
#         # 3. 将它们拼接起来 (维度 d_x + d_text)
#         # 4. (可选) 通过一个线性层进行融合和降维
#         self.input_fusion_dim = d_x + d_text
#         if fusion_dim is None:
#             fusion_dim = self.input_fusion_dim # 如果不指定，就用拼接后的维度
#
#         # 如果需要进行特征空间的对齐或降维
#         self.fusion_layer = nn.Linear(self.input_fusion_dim, fusion_dim)
#         self.fusion_activation = nn.GELU() # 或者其他激活函数
#
#         # --- 门控参数现在基于融合后的维度 fusion_dim ---
#         self.w_gate = nn.Parameter(torch.zeros(fusion_dim, M), requires_grad=True)
#         self.w_noise = nn.Parameter(torch.zeros(fusion_dim, M), requires_grad=True)
#
#         # ... MoEGate 的其他部分 (softplus, softmax, buffers, 断言) ...
#         self.softplus = nn.Softplus()
#         self.softmax = nn.Softmax(dim=1)
#         self.register_buffer("mean", torch.tensor([0.0]))
#         self.register_buffer("std", torch.tensor([1.0]))
#         assert self.k <= self.M
#
#     # --- 修改 forward 方法以接收 Degraded_feature ---
#     def forward(self, x: torch.Tensor, Degraded_feature: torch.Tensor, loss_coef=1e-2, noise_epsilon=1e-2):
#         """
#         接收 x 和 Degraded_feature 进行门控计算。
#         """
#         batch_size = x.shape[0]
#
#         # 1. 提取 x 的全局特征
#         x_pooled = self.gap(x).view(batch_size, -1) # 形状: [B, d_x]
#
#         # 2. (确保 Degraded_feature 形状是 [B, d_text])
#         #    可能需要检查和调整 Degraded_feature 的形状
#
#         # 3. 融合特征 (示例：简单拼接)
#         # print('x_pooled.shape',x_pooled.shape)
#         # print('Degraded_feature.shape',Degraded_feature.shape)
#         fused_feature = torch.cat([x_pooled, Degraded_feature], dim=1) # 形状: [B, d_x + d_text]
#
#         # 4. (可选) 通过融合层处理
#         if hasattr(self, 'fusion_layer'):
#             fused_feature = self.fusion_layer(fused_feature)
#             fused_feature = self.fusion_activation(fused_feature)
#         # 此时 fused_feature 形状为 [B, fusion_dim]
#
#         # --- 后续计算与 MoEGate 类似，但使用 fused_feature ---
#         clean_logits = fused_feature @ self.w_gate # 使用融合特征计算 logits
#         probs_for_mi = self.softmax(clean_logits)  # <--- USE CLEAN LOGITS HERE
#         if self.noisy_gating and self.training:
#             raw_noise_stddev = fused_feature @ self.w_noise
#             noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
#             noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
#             logits = noisy_logits
#         else:
#             logits = clean_logits
#
#         # ... Top-K 选择、创建稀疏 gates、计算 moe_loss 的逻辑与原 MoEGate 相同 ...
#         num_experts_to_consider = min(self.k + 1, self.M)
#         top_logits, top_indices = logits.topk(num_experts_to_consider, dim=1)
#         top_k_logits = top_logits[:, :self.k]
#         top_k_indices = top_indices[:, :self.k]
#         top_k_gates = self.softmax(top_k_logits)
#         zeros = torch.zeros_like(logits, requires_grad=False)
#         gates = zeros.scatter(dim=1, index=top_k_indices, src=top_k_gates)
#         # print('gates.shape',gates.shape)
#         importance = gates.sum(dim=0)
#         if self.noisy_gating and self.k < self.M and self.training:
#             # 注意：_prob_in_top_k 也需要使用融合后的 clean_logits, noisy_logits, noise_stddev
#             load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(dim=0)
#         else:
#             load = self._gates_to_load(gates)
#
#         loss = self.cv_squared(importance) + self.cv_squared(load)
#         moe_loss = loss * loss_coef
#
#         return gates, moe_loss, probs_for_mi, top_k_indices # <--- 加入 top_k_indices# <--- Return probs from clean_logits
#     def _gates_to_load(self, gates: torch.Tensor):
#         """
#         辅助函数：根据门控值计算每个专家的实际负载（被分配的样本数）。
#
#         对于一个专家，其负载定义为在这个批次中，门控值大于 0 的样本数量。
#
#         Args:
#             gates (torch.Tensor): 稀疏门控张量，形状 [Batch, M]。
#
#         Returns:
#             torch.Tensor: 形状为 [M] 的张量，包含每个专家的负载计数。
#         """
#         # `gates > 0` 产生一个布尔张量。
#         # `.sum(dim=0)` 沿着批次维度求和，True 被视为 1，False 被视为 0。
#         # 结果即为每个专家（每列）被选中的次数。
#         return (gates > 0).sum(dim=0)
#
#     def cv_squared(self, x: torch.Tensor):
#         """
#         辅助函数：计算平方变异系数 (CV^2)。
#
#         CV^2 = variance(x) / (mean(x)^2)
#         它是一个衡量相对变异性的指标，常用于量化分布（如专家负载或重要性）的均匀度。
#         CV^2 值越小，表示分布越均匀。
#         对于空张量或只有一个元素的张量，返回 0。
#         添加了一个小的 epsilon 以保证数值稳定性（避免除以接近零的均值）。
#
#         Args:
#             x (torch.Tensor): 需要计算 CV^2 的张量（通常是一维的，如负载或重要性列表）。
#
#         Returns:
#             torch.Tensor: 包含 CV^2 值的标量张量。
#         """
#         eps = 1e-10 # 防止除以零的小常数
#
#         # 如果张量元素少于等于1个，无法计算方差，返回 0。
#         if x.shape[0] <= 1:
#             return torch.tensor(0.0, device=x.device, dtype=torch.float32) # 返回浮点型的 0
#
#         # 计算方差 (variance) 和 均值的平方 (mean squared)。
#         # 使用 .float() 确保计算在浮点数上进行。
#         variance = x.float().var()
#         mean_squared = x.float().mean() ** 2
#         # 返回 CV^2 = variance / (mean^2 + eps)
#         return variance / (mean_squared + eps)
#
#     def _prob_in_top_k(self, clean_values: torch.Tensor, noisy_values: torch.Tensor, noise_stddev: torch.Tensor, noisy_top_values: torch.Tensor):
#         """
#         辅助函数：用于计算带噪声 Top-K 门控的负载均衡损失。
#
#         它计算的是，给定当前的干净 logits、噪声 logits 和噪声标准差，如果重新采样噪声，
#         每个元素（专家 logit）进入 Top-K 的概率是多少。这提供了一个可微分的负载度量。
#
#         这个方法的逻辑基于 "Outrageously Large Neural Networks" 论文 (Shazeer et al., 2017)
#         及其相关实现。
#
#         Args:
#             clean_values (torch.Tensor): 未加噪声的 logits。形状 [Batch, M].
#             noisy_values (torch.Tensor): 添加噪声后的 logits。形状 [Batch, M].
#             noise_stddev (torch.Tensor): 添加的噪声的标准差。形状 [Batch, M].
#             noisy_top_values (torch.Tensor): 带噪声的 logits 中，排名前 k+1 的值。形状 [Batch, k+1].
#
#         Returns:
#             torch.Tensor: 形状为 [Batch, M] 的张量，其中元素 [b, m] 表示对于样本 b，
#                           专家 m 进入 Top-K 的概率。
#         """
#         batch_size = clean_values.size(0) # 批次大小 B
#         # k = self.k (选择的专家数)
#         # m = noisy_top_values.size(1) (实际是 k+1)
#         m = noisy_top_values.size(1) # m = k+1
#         # 将 top values 展平成一维，方便索引。形状: [B * m]
#         top_values_flat = noisy_top_values.flatten()
#
#         # --- 确定进入 Top-K 的阈值 logit ---
#         # 对于每个样本 b，其进入 Top-K 的阈值是第 k 高的 *带噪声* logit 值。
#         # 在 `top_values_flat` (长度 B*m) 中找到每个样本的第 k 个元素的位置索引。
#         # `torch.arange(batch_size) * m` 定位每个样本的起始位置，`+ self.k` 定位第 k 个元素 (0-based index k)。
#         threshold_positions_if_in = torch.arange(batch_size, device=clean_values.device) * m + self.k
#         # 使用 `torch.gather` 根据计算出的索引从 `top_values_flat` 中提取实际的阈值。
#         # `unsqueeze(1)` 将形状从 [B] 变为 [B, 1] 以便后续广播。
#         # `threshold_if_in`: 如果某个 noisy_value 确实在 Top-K 中，它需要超过的阈值。
#         threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
#
#         # 同样地，找到第 k-1 个元素的位置索引（作为不在 Top-K 时的阈值）。
#         threshold_positions_if_out = threshold_positions_if_in - 1
#         # `threshold_if_out`: 如果某个 noisy_value 不在 Top-K 中，它需要超过的阈值（才能进入）。
#         threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
#
#         # --- 使用正态分布累积分布函数 (CDF) 计算概率 ---
#         # `torch.gt(a, b)` 比较 a > b 是否成立。
#         # `is_in` 判断每个 `noisy_values` 是否大于它对应的 `threshold_if_in`。
#         # 即，在这次噪声采样中，该值是否确实进入了 Top-K。形状 [B, M]
#         is_in = torch.gt(noisy_values, threshold_if_in)
#         normal = Normal(self.mean, self.std)
#         prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
#         # 计算如果当前值不在 Top-K 内，重新采样噪声后进入 Top-K 内的概率。
#         # 使用 `threshold_if_out` 作为阈值。
#         prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
#
#         # `torch.where(condition, x, y)`: 如果 condition 为 True，选择 x，否则选择 y。
#         # 根据 `is_in` 的结果，选择对应的概率。
#         prob = torch.where(is_in, prob_if_in, prob_if_out) # 形状: [B, M]
#
#         return prob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal # 需要导入 Normal

class FusedMoEGate(nn.Module):
    """
    融合输入 x 特征和外部 Degraded_feature 的 MoE 门控网络。
    增加了选项以控制是否计算标准的辅助（负载均衡）损失。
    """

    def __init__(self,
                 d_x: int,
                 d_text: int,
                 M: int = 4,
                 K: int = 1,
                 noisy_gating: bool = True,
                 fusion_dim: int = None,
                 calculate_standard_aux_loss: bool = False, # <-- 新增控制标志
                 loss_coef: float = 1e-2,                 # <-- 将 loss_coef 移到 init
                 noise_epsilon: float = 1e-2              # <-- 将 noise_epsilon 移到 init
                ):
        """
        Args:
            d_x (int): 输入 x 的通道维度 C。
            d_text (int): Degraded_feature 的维度。
            M (int): 专家总数。
            K (int): Top-K 选择。
            noisy_gating (bool): 是否使用噪声门控。
            fusion_dim (int, optional): 融合后特征的维度。如果为 None，则为 d_x + d_text。
            calculate_standard_aux_loss (bool): 如果为 True，则计算并返回标准的 MoE 辅助损失（基于 CV^2）。默认为 True。
            loss_coef (float): 标准辅助损失的系数。
            noise_epsilon (float): 添加到噪声标准差的 epsilon，以确保稳定性。
        """
        super().__init__()
        self.M = M
        self.k = K
        self.noisy_gating = noisy_gating
        self.calculate_standard_aux_loss = calculate_standard_aux_loss # <-- 存储控制标志
        self.loss_coef = loss_coef
        self.noise_epsilon = noise_epsilon

        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # 处理 x

        # --- 融合层 ---
        self.input_fusion_dim = d_x + d_text
        if fusion_dim is None:
            fusion_dim = self.input_fusion_dim
        self.fusion_layer = nn.Linear(self.input_fusion_dim, fusion_dim)
        self.fusion_activation = nn.GELU()

        # --- 门控参数 ---
        self.w_gate = nn.Parameter(torch.zeros(fusion_dim, M), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(fusion_dim, M), requires_grad=True)

        # --- 其他 ---
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.M

    def forward(self, x: torch.Tensor, Degraded_feature: torch.Tensor): # 移除 forward 中的 loss_coef 和 noise_epsilon
        """
        接收 x 和 Degraded_feature 进行门控计算。
        根据 self.calculate_standard_aux_loss 决定是否计算 moe_loss。
        """
        batch_size = x.shape[0]
        device = x.device # 获取设备

        # 1. 特征融合 (保持不变)
        x_pooled = self.gap(x).view(batch_size, -1)
        fused_feature = torch.cat([x_pooled, Degraded_feature], dim=1)
        if hasattr(self, 'fusion_layer'):
            fused_feature = self.fusion_layer(fused_feature)
            fused_feature = self.fusion_activation(fused_feature)

        # 2. 计算 Logits 和 Probabilities (保持不变)
        clean_logits = fused_feature @ self.w_gate
        probs_for_mi = self.softmax(clean_logits) # 用于 MI Loss

        if self.noisy_gating and self.training:
            raw_noise_stddev = fused_feature @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + self.noise_epsilon # 使用 self.noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
            noise_stddev = None # 确保在非训练或非噪声门控时为 None

        # 3. Top-K 选择和稀疏门控 (保持不变)
        num_experts_to_consider = min(self.k + 1, self.M)
        top_logits, top_indices = logits.topk(num_experts_to_consider, dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=False)
        gates = zeros.scatter(dim=1, index=top_k_indices, src=top_k_gates) # 稀疏 gates 用于 dispatcher

        # 4. 条件性计算标准辅助损失 (moe_loss)
        moe_loss = torch.tensor(0.0, device=device) # 默认损失为 0
        if self.calculate_standard_aux_loss: # <-- 检查标志
            importance = gates.sum(dim=0)
            if self.noisy_gating and self.k < self.M and self.training and noise_stddev is not None:
                # 只有在 noisy gating 且需要时才调用 _prob_in_top_k
                load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(dim=0)
            else:
                load = self._gates_to_load(gates)

            loss = self.cv_squared(importance) + self.cv_squared(load)
            moe_loss = loss * self.loss_coef # 使用 self.loss_coef

        # 5. 返回结果
        # moe_loss 要么是计算得到的损失，要么是 0
        return gates, moe_loss, probs_for_mi, top_k_indices

    # --- 辅助函数 (_gates_to_load, cv_squared, _prob_in_top_k) 保持不变 ---
    def _gates_to_load(self, gates: torch.Tensor):
        return (gates > 0).sum(dim=0)

    def cv_squared(self, x: torch.Tensor):
        eps = 1e-10
        if x.shape[0] <= 1:
            return torch.tensor(0.0, device=x.device, dtype=torch.float32)
        variance = x.float().var()
        mean_squared = x.float().mean() ** 2
        return variance / (mean_squared + eps)

    def _prob_in_top_k(self, clean_values: torch.Tensor, noisy_values: torch.Tensor, noise_stddev: torch.Tensor, noisy_top_values: torch.Tensor):
        # --- 需要导入 Normal ---
        # from torch.distributions.normal import Normal
        batch_size = clean_values.size(0)
        m = noisy_top_values.size(1) # k+1
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch_size, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        is_in = torch.gt(noisy_values, threshold_if_in)
        normal = Normal(self.mean, self.std) # 使用 self.mean, self.std (已注册为 buffer)
        # 确保 noise_stddev 不为零，添加小的 epsilon
        noise_stddev_safe = noise_stddev + 1e-6
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev_safe)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev_safe)

        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob



# class SparseDispatcher(object):
#     """
#     稀疏调度器：用于在 MoE 层中高效地将输入分发给专家，并组合它们的输出。
#
#     此类接收一个稀疏的门控张量 `gates`（来自 MoEGate），并预先计算索引，
#     以便能够高效地将输入数据仅仅路由到每个样本被选中的那些专家那里。
#     它避免了对未被选中的专家进行不必要的计算和数据传输。
#
#     主要提供两个方法：
#     1. `dispatch`: 接收完整的输入批次，返回一个列表，列表中每个张量包含发送给特定专家的输入样本。
#     2. `combine`: 接收来自专家的输出张量列表，并将它们组合回对应原始批次的单个输出张量，
#                  可以选择性地根据门控值进行加权。
#
#     假设门控张量 `gates` 表明了哪些批次元素（样本）需要发送给哪些专家
#     （即 `gates[b, e] > 0` 表示样本 `b` 需要发送给专家 `e`）。
#
#     注意：原始实现的注释提到输入/输出是二维的 [batch, depth]，但这里提供的 `combine` 方法
#           能够处理四维张量 [batch, C, H, W]。调用者需要确保数据格式的一致性或进行适配。
#
#     参考文献:
#         - Shazeer et al., "Outrageously Large Neural Networks" (https://arxiv.org/abs/1701.06538)
#         - 实现灵感来源: https://github.com/davidmrau/mixture-of-experts/blob/master/moe.py
#     """
#
#     def __init__(self, num_experts: int, gates: torch.Tensor):
#         """
#         初始化 SparseDispatcher。根据输入的门控值 `gates` 预先计算路由所需的索引。
#
#         Args:
#             num_experts (int): 专家总数 (M)。
#             gates (torch.Tensor): 来自 MoEGate 的稀疏门控张量，形状为 [Batch, num_experts]。
#         """
#         self._gates = gates           # 保存原始的门控张量 [B, M]
#         self._num_experts = num_experts # 保存专家总数 M
#
#         # --- 预计算用于高效路由的索引 ---
#
#         # 1. 找出所有非零门控值的位置，并按专家索引排序。
#         # `torch.nonzero(gates)` 返回 `gates` 中非零元素的索引，形状为 [NonZeroElements, 2]，
#         # 其中每一行是 `[batch_idx, expert_idx]`。
#         # `.sort(0)` 按列（维度0）对这些索引对进行排序。我们主要关心按 `expert_idx` (第1列) 排序的结果。
#         # `sorted_experts`: 排序后的 `[batch_idx, expert_idx]` 对。
#         # `index_sorted_experts`: 从排序后位置映射回原始非零位置的索引。
#         sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
#
#         # 2. 提取每个非零门控对应的专家索引，按专家顺序排列。
#         # `split(1, dim=1)` 将 `sorted_experts` ([N, 2]) 沿着维度1（列）拆分成两个 [N, 1] 的张量。
#         # 我们只需要第二个张量，即专家索引 `_expert_index`。形状: [NonZeroElements, 1]
#         _, self._expert_index = sorted_experts.split(1, dim=1)
#
#         # 3. 提取与排序后的专家索引对应的原始批次索引。
#         # `torch.nonzero(gates)[:, 0]` 获取所有非零元素的原始批次索引。
#         # `index_sorted_experts[:, 1]` 提供了按专家排序后的顺序。
#         # 通过这个索引来重新排列原始批次索引，得到 `_batch_index`。
#         # `_batch_index[i]` 表示当所有非零条目按专家分组排序后，第 i 个条目对应的原始批次索引。
#         # 形状: [NonZeroElements]
#         self._batch_index = torch.nonzero(gates)[:, 0][index_sorted_experts[:, 1]]
#
#         # 4. 计算每个专家接收到的样本数量。
#         # `(gates > 0)` 创建一个布尔张量。
#         # `.sum(0)` 沿着批次维度（维度0）求和，得到每个专家（每列）被选中的次数。
#         # `_part_sizes[e]` 就是专家 `e` 将接收的样本数。转换为列表。形状: [M] (列表)
#         self._part_sizes = (gates > 0).sum(0).tolist()
#
#         # 5. 收集与排序后的非零条目对应的实际门控值。
#         # `gates[self._batch_index.flatten()]`: 使用重新排序的批次索引 `_batch_index` 从原始 `gates` 中提取行。
#         # `gates_exp` 的形状是 `[NonZeroElements, M]`。
#         gates_exp = gates[self._batch_index.flatten()]
#         # `torch.gather(input, dim, index)` 从 `input` 中根据 `index` 收集值。
#         # `dim=1`: 沿着专家维度收集。
#         # `index=self._expert_index`: 指定要收集的专家索引。
#         # `_nonzero_gates[i]` 就是第 i 个非零条目（按专家排序后）对应的门控值。
#         # 形状: [NonZeroElements, 1]
#         self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
#
#     def dispatch(self, inp: torch.Tensor) -> list[torch.Tensor]:
#         """
#         根据预计算的路由信息，将输入张量 `inp` 分发给各个专家。
#
#         接收完整的输入批次 `inp`，并将其分割成一个列表，列表中第 i 个张量
#         包含了所有需要由第 i 个专家处理的样本数据。
#
#         Args:
#             inp (torch.Tensor): 整个批次的输入张量。
#                                 形状: `[Batch, <extra_input_dims>]` (例如, [B, C, H, W]).
#
#         Returns:
#             list[torch.Tensor]: 一个包含 `num_experts` 个张量的列表。列表中的第 i 个张量
#                                 形状为 `[expert_batch_size_i, <extra_input_dims>]`，
#                                 包含了发送给第 i 个专家的输入样本。
#                                 `expert_batch_size_i` 是路由到专家 i 的样本数量（可能为 0）。
#         """
#         # 使用预计算的 `_batch_index` 从输入 `inp` 中选取所有需要被处理的样本。
#         # 如果一个样本被路由到 K 个专家，它将在 `inp_exp` 中出现 K 次。
#         # 结果 `inp_exp` 中的样本是按专家顺序排列好的。
#         # 形状: `[NonZeroElements, <extra_input_dims>]`
#         inp_exp = inp[self._batch_index]
#
#         # `torch.split(tensor, split_size_or_sections, dim=0)`
#         # 根据 `_part_sizes` (一个包含每个块大小的列表) 将 `inp_exp` 沿着维度 0 (批次/样本维度)
#         # 分割成多个张量块。每个块对应一个专家的输入。
#         return torch.split(inp_exp, self._part_sizes, dim=0)
#
#     def combine(self, expert_out, multiply_by_gates) -> torch.Tensor:
#         """
#         将来自各个专家的输出组合回对应原始批次的单个张量。
#
#         接收一个包含所有专家输出的列表 `expert_out`，将它们拼接起来，
#         然后使用 `index_add` 操作将这些输出（可以选择性地乘以门控值）
#         添加/放置回一个与原始批次布局对应的结果张量中。
#
#         Args:
#             expert_out (list[torch.Tensor]): 包含 `num_experts` 个张量的列表，其中第 i 个张量
#                                              是第 i 个专家的输出。
#                                              第 i 个张量的形状: `[expert_batch_size_i, <extra_output_dims>]`。
#                                              例如 `<extra_output_dims>` 可以是 C, H, W。
#             multiply_by_gates (bool): 如果为 True，在组合前将每个专家的输出乘以其对应的门控值。
#                                       如果为 False，则简单地将处理同一原始样本的多个专家的输出相加。
#                                       默认为 True。
#
#         Returns:
#             torch.Tensor: 组合后的输出张量。
#                           形状: `[Batch, <extra_output_dims>]`。
#         """
#         # `torch.cat(expert_out, 0)` 将专家输出列表中的所有张量沿着维度 0 (批次/样本维度) 拼接起来。
#         # `stitched` 包含了所有专家处理的结果，按专家顺序排列。
#         # 形状: `[NonZeroElements, <output_dims>]`
#         # 注意: 原始代码中可能包含 `.exp()` / `.log()` 操作，暗示专家可能在对数空间操作。
#         # 这里暂时省略，如果需要，应取消注释。
#         stitched = torch.cat(expert_out, 0) # .exp()
#
#         if multiply_by_gates:
#             # 如果需要，将每个专家输出乘以对应的门控值进行加权。
#             # `self._nonzero_gates` 的形状是 [NonZeroElements, 1]。
#             # 为了与 `stitched` (例如 [NonZeroElements, C, H, W]) 进行逐元素乘法，
#             # 需要使用 `unsqueeze` 将 `_nonzero_gates` 的维度扩展到与 `stitched` 匹配，
#             # 以便进行广播 (broadcasting)。
#             if stitched.dim() == 4: # 假设 stitched 是 4D: [N, C, H, W]
#                  # 增加 C, H, W 三个维度
#                  broadcast_gates = self._nonzero_gates.unsqueeze(-1).unsqueeze(-1) # 形状变为 [N, 1, 1, 1]
#             # 可以添加对其他维度的处理
#             # elif stitched.dim() == 2: # 假设 stitched 是 2D: [N, D]
#             #     broadcast_gates = self._nonzero_gates # 形状已经是 [N, 1]，可以直接广播或只需 unsqueeze 一次
#             else: # 通用处理，根据需要增加维度
#                  broadcast_gates = self._nonzero_gates
#                  # 增加 `stitched` 维度数 - `_nonzero_gates` 维度数 个维度
#                  for _ in range(stitched.dim() - self._nonzero_gates.dim()):
#                      broadcast_gates = broadcast_gates.unsqueeze(-1)
#
#             # 执行逐元素乘法
#             stitched = stitched.mul(broadcast_gates)
#
#         # 创建一个最终输出形状的全零张量，作为组合结果的容器。
#         # 批次大小从 `self._gates` 获取 (原始批次大小 B)。
#         # 其他维度 (<output_dims>) 从 `expert_out` 中的某个张量推断 (假设所有专家输出维度一致)。
#         output_dims = expert_out[-1].size()[1:] # 获取除批次维度外的其他维度形状
#         zeros = torch.zeros(
#             self._gates.size(0), # 原始批次大小 B
#             *output_dims,        # 输出维度，例如 C, H, W
#             requires_grad=True,  # 确保梯度可以反向传播
#             device=stitched.device, # 与 stitched 在同一设备上
#             dtype=stitched.dtype    # 与 stitched 数据类型一致
#         )
#
#         # `zeros.index_add_(dim, index, source)` 是原地操作。
#         # 这里使用非原地版本 `zeros.index_add(dim, index, source)`。
#         # `dim=0`: 沿着维度 0 (批次维度) 进行添加。
#         # `index=self._batch_index`: 指定 `source` (即 `stitched`) 中的每个元素应该添加到 `zeros` 的哪一行（哪个原始样本）。
#         # `source=stitched`: 要添加的数据。
#         # 这个操作会将所有对应同一个原始样本的专家输出（可能已加权）累加到 `zeros` 张量的相应行中。
#         combined = zeros.index_add(0, self._batch_index, stitched)
#
#         # 如果之前使用了 exp()，这里可能需要 log()，并添加 eps 防止 log(0)。
#         # combined[combined == 0] = np.finfo(float).eps
#         # return combined.log()
#         return combined # 如果没有 log/exp 操作，直接返回 combined
#
#     def expert_to_gates(self) -> list[torch.Tensor]:
#         """
#         获取每个专家处理的样本对应的门控值。
#
#         Returns:
#             list[torch.Tensor]: 一个包含 `num_experts` 个张量的列表。第 i 个张量包含了
#                                 所有被分发给第 i 个专家的样本对应的原始门控值。
#                                 第 i 个张量的形状: `[expert_batch_size_i, 1]`。
#         """
#         # 使用 `torch.split` 将预计算好的 `_nonzero_gates` (形状 [NonZeroElements, 1])
#         # 按照每个专家处理的样本数 `_part_sizes` 分割成列表。
#         return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

import torch
import torch.nn as nn
# Assuming Normal and F are not directly used in SparseDispatcher itself
# from torch.distributions.normal import Normal
# import torch.nn.functional as F

class SparseDispatcher(object):
    """
    稀疏调度器：用于在 MoE 层中高效地将输入分发给专家，并组合它们的输出。

    此类接收一个稀疏的门控张量 `gates`（来自 MoEGate），并预先计算索引，
    以便能够高效地将输入数据仅仅路由到每个样本被选中的那些专家那里。
    它避免了对未被选中的专家进行不必要的计算和数据传输。

    主要提供两个方法：
    1. `dispatch`: 接收完整的输入批次，返回一个列表，列表中每个张量包含发送给特定专家的输入样本。
    2. `combine`: 接收来自专家的输出张量列表，并将它们组合回对应原始批次的单个输出张量，
                 可以选择性地根据门控值进行加权。

    假设门控张量 `gates` 表明了哪些批次元素（样本）需要发送给哪些专家
    （即 `gates[b, e] > 0` 表示样本 `b` 需要发送给专家 `e`）。

    注意：原始实现的注释提到输入/输出是二维的 [batch, depth]，但这里提供的 `combine` 方法
          能够处理四维张量 [batch, C, H, W]。调用者需要确保数据格式的一致性或进行适配。

    参考文献:
        - Shazeer et al., "Outrageously Large Neural Networks" (https://arxiv.org/abs/1701.06538)
        - 实现灵感来源: https://github.com/davidmrau/mixture-of-experts/blob/master/moe.py
    """

    def __init__(self, num_experts: int, gates: torch.Tensor):
        """
        初始化 SparseDispatcher。根据输入的门控值 `gates` 预先计算路由所需的索引。

        Args:
            num_experts (int): 专家总数 (M)。
            gates (torch.Tensor): 来自 MoEGate 的稀疏门控张量，形状为 [Batch, num_experts]。
        """
        self._gates = gates           # 保存原始的门控张量 [B, M]
        self._num_experts = num_experts # 保存专家总数 M

        # --- 预计算用于高效路由的索引 ---

        # 1. 找出所有非零门控值的位置，并按专家索引排序。
        # `torch.nonzero(gates)` 返回 `gates` 中非零元素的索引，形状为 [NonZeroElements, 2]，
        # 其中每一行是 `[batch_idx, expert_idx]`。
        # `.sort(0)` 按列（维度0）对这些索引对进行排序。我们主要关心按 `expert_idx` (第1列) 排序的结果。
        # `sorted_experts`: 排序后的 `[batch_idx, expert_idx]` 对。
        # `index_sorted_experts`: 从排序后位置映射回原始非零位置的索引。
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)

        # 2. 提取每个非零门控对应的专家索引，按专家顺序排列。
        # `split(1, dim=1)` 将 `sorted_experts` ([N, 2]) 沿着维度1（列）拆分成两个 [N, 1] 的张量。
        # 我们只需要第二个张量，即专家索引 `_expert_index`。形状: [NonZeroElements, 1]
        _, self._expert_index = sorted_experts.split(1, dim=1)

        # 3. 提取与排序后的专家索引对应的原始批次索引。
        # `torch.nonzero(gates)[:, 0]` 获取所有非零元素的原始批次索引。
        # `index_sorted_experts[:, 1]` 提供了按专家排序后的顺序。
        # 通过这个索引来重新排列原始批次索引，得到 `_batch_index`。
        # `_batch_index[i]` 表示当所有非零条目按专家分组排序后，第 i 个条目对应的原始批次索引。
        # 形状: [NonZeroElements]
        self._batch_index = torch.nonzero(gates)[:, 0][index_sorted_experts[:, 1]]

        # 4. 计算每个专家接收到的样本数量。
        # `(gates > 0)` 创建一个布尔张量。
        # `.sum(0)` 沿着批次维度（维度0）求和，得到每个专家（每列）被选中的次数。
        # `_part_sizes[e]` 就是专家 `e` 将接收的样本数。转换为列表。形状: [M] (列表)
        self._part_sizes = (gates > 0).sum(0).tolist()

        # 5. 收集与排序后的非零条目对应的实际门控值。
        # `gates[self._batch_index.flatten()]`: 使用重新排序的批次索引 `_batch_index` 从原始 `gates` 中提取行。
        # `gates_exp` 的形状是 `[NonZeroElements, M]`。
        gates_exp = gates[self._batch_index.flatten()]
        # `torch.gather(input, dim, index)` 从 `input` 中根据 `index` 收集值。
        # `dim=1`: 沿着专家维度收集。
        # `index=self._expert_index`: 指定要收集的专家索引。
        # `_nonzero_gates[i]` 就是第 i 个非零条目（按专家排序后）对应的门控值。
        # 形状: [NonZeroElements, 1]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """
        根据预计算的路由信息，将输入张量 `inp` 分发给各个专家。

        接收完整的输入批次 `inp`，并将其分割成一个列表，列表中第 i 个张量
        包含了所有需要由第 i 个专家处理的样本数据。

        Args:
            inp (torch.Tensor): 整个批次的输入张量。
                                形状: `[Batch, <extra_input_dims>]` (例如, [B, C, H, W]).

        Returns:
            list[torch.Tensor]: 一个包含 `num_experts` 个张量的列表。列表中的第 i 个张量
                                形状为 `[expert_batch_size_i, <extra_input_dims>]`，
                                包含了发送给第 i 个专家的输入样本。
                                `expert_batch_size_i` 是路由到专家 i 的样本数量（可能为 0）。
        """
        # 使用预计算的 `_batch_index` 从输入 `inp` 中选取所有需要被处理的样本。
        # 如果一个样本被路由到 K 个专家，它将在 `inp_exp` 中出现 K 次。
        # 结果 `inp_exp` 中的样本是按专家顺序排列好的。
        # 形状: `[NonZeroElements, <extra_input_dims>]`
        inp_exp = inp[self._batch_index]

        # `torch.split(tensor, split_size_or_sections, dim=0)`
        # 根据 `_part_sizes` (一个包含每个块大小的列表) 将 `inp_exp` 沿着维度 0 (批次/样本维度)
        # 分割成多个张量块。每个块对应一个专家的输入。
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates) -> torch.Tensor:
        """
        将来自各个专家的输出组合回对应原始批次的单个张量。

        接收一个包含所有专家输出的列表 `expert_out`，将它们拼接起来，
        然后使用 `index_add` 操作将这些输出（可以选择性地乘以门控值）
        添加/放置回一个与原始批次布局对应的结果张量中。

        Args:
            expert_out (list[torch.Tensor]): 包含 `num_experts` 个张量的列表，其中第 i 个张量
                                             是第 i 个专家的输出。
                                             第 i 个张量的形状: `[expert_batch_size_i, <extra_output_dims>]`。
                                             例如 `<extra_output_dims>` 可以是 C, H, W。
            multiply_by_gates (bool): 如果为 True，在组合前将每个专家的输出乘以其对应的门控值。
                                      如果为 False，则简单地将处理同一原始样本的多个专家的输出相加。
                                      默认为 True。

        Returns:
            torch.Tensor: 组合后的输出张量。
                          形状: `[Batch, <extra_output_dims>]`。
        """
        # `torch.cat(expert_out, 0)` 将专家输出列表中的所有张量沿着维度 0 (批次/样本维度) 拼接起来。
        # `stitched` 包含了所有专家处理的结果，按专家顺序排列。
        # 形状: `[NonZeroElements, <output_dims>]`
        # 注意: 原始代码中可能包含 `.exp()` / `.log()` 操作，暗示专家可能在对数空间操作。
        # 这里暂时省略，如果需要，应取消注释。
        stitched = torch.cat(expert_out, 0) # .exp()

        if multiply_by_gates:
            # 如果需要，将每个专家输出乘以对应的门控值进行加权。
            # `self._nonzero_gates` 的形状是 [NonZeroElements, 1]。
            # 为了与 `stitched` (例如 [NonZeroElements, C, H, W]) 进行逐元素乘法，
            # 需要使用 `unsqueeze` 将 `_nonzero_gates` 的维度扩展到与 `stitched` 匹配，
            # 以便进行广播 (broadcasting)。
            if stitched.dim() == 4: # 假设 stitched 是 4D: [N, C, H, W]
                 # 增加 C, H, W 三个维度
                 broadcast_gates = self._nonzero_gates.unsqueeze(-1).unsqueeze(-1) # 形状变为 [N, 1, 1, 1]
            # 可以添加对其他维度的处理
            # elif stitched.dim() == 2: # 假设 stitched 是 2D: [N, D]
            #     broadcast_gates = self._nonzero_gates # 形状已经是 [N, 1]，可以直接广播或只需 unsqueeze 一次
            else: # 通用处理，根据需要增加维度
                 broadcast_gates = self._nonzero_gates
                 # 增加 `stitched` 维度数 - `_nonzero_gates` 维度数 个维度
                 for _ in range(stitched.dim() - self._nonzero_gates.dim()):
                     broadcast_gates = broadcast_gates.unsqueeze(-1)

            # 执行逐元素乘法
            stitched = stitched.mul(broadcast_gates)

        # 创建一个最终输出形状的全零张量，作为组合结果的容器。
        # 批次大小从 `self._gates` 获取 (原始批次大小 B)。
        # 其他维度 (<output_dims>) 从 `expert_out` 中的某个张量推断 (假设所有专家输出维度一致)。
        output_dims = expert_out[-1].size()[1:] # 获取除批次维度外的其他维度形状
        zeros = torch.zeros(
            self._gates.size(0), # 原始批次大小 B
            *output_dims,        # 输出维度，例如 C, H, W
            requires_grad=True,  # 确保梯度可以反向传播
            device=stitched.device, # 与 stitched 在同一设备上
            dtype=stitched.dtype    # 与 stitched 数据类型一致
        )

        # `zeros.index_add_(dim, index, source)` 是原地操作。
        # 这里使用非原地版本 `zeros.index_add(dim, index, source)`。
        # `dim=0`: 沿着维度 0 (批次维度) 进行添加。
        # `index=self._batch_index`: 指定 `source` (即 `stitched`) 中的每个元素应该添加到 `zeros` 的哪一行（哪个原始样本）。
        # `source=stitched`: 要添加的数据。
        # 这个操作会将所有对应同一个原始样本的专家输出（可能已加权）累加到 `zeros` 张量的相应行中。
        combined = zeros.index_add(0, self._batch_index, stitched)

        # 如果之前使用了 exp()，这里可能需要 log()，并添加 eps 防止 log(0)。
        # combined[combined == 0] = np.finfo(float).eps
        # return combined.log()
        return combined # 如果没有 log/exp 操作，直接返回 combined

    def expert_to_gates(self):
        """
        获取每个专家处理的样本对应的门控值。
        """
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class SimpleMoELayer(nn.Module):
    """
    一个简单的混合专家（MoE）层示例，演示 MoEGate 和 SparseDispatcher 的配合使用。
    假设输入是 4D 张量 (B, C, H, W)，专家是简单的线性层作用在展平后的特征上。
    """
    def __init__(self, input_channels: int, input_height: int, input_width: int, num_experts: int, top_k: int):
        """
        初始化简单的 MoE 层。

        Args:
            input_channels (int): 输入特征的通道数 C。
            input_height (int): 输入特征的高度 H。
            input_width (int): 输入特征的宽度 W。
            num_experts (int): 专家的总数 M。
            top_k (int): 每个样本选择的专家数量 K。
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width

        # 计算展平后的特征维度，专家将在这个维度上操作
        # 呜喵~ 要把图片特征压扁扁才能送给线性专家师傅...
        self.flattened_dim = input_channels * input_height * input_width

        # 实例化门控网络 MoEGate
        # 这里的 'd' 对应输入给门控网络的通道数 C
        # 本喵站在这里守门！(≧ω≦)／
        self.gate = FusedMoEGate(d_x=input_channels, M=num_experts, K=top_k)

        # 实例化专家列表
        # nn.ModuleList 用于存储一系列的 nn.Module
        # 请各位专家师傅就位~ 排排坐好！
        self.experts = nn.ModuleList([
            nn.Linear(self.flattened_dim, self.flattened_dim) for i in range(num_experts)
        ])

    def forward(self, x: torch.Tensor):
        """
        MoE 层的前向传播过程。

        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C, H, W]。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - output (torch.Tensor): MoE 层的输出张量，形状与输入相同 [B, C, H, W]。
            - moe_aux_loss (torch.Tensor): 来自 MoEGate 的负载均衡损失。
        """
        # 检查输入形状是否符合预期，喵呜~
        assert x.dim() == 4 and x.shape[1] == self.input_channels and \
               x.shape[2] == self.input_height and x.shape[3] == self.input_width, \
               f"输入的形状不对哦~ 需要 [B={x.shape[0]}, C={self.input_channels}, H={self.input_height}, W={self.input_width}]，但得到的是 {x.shape}"

        # --- 1. 通过门控网络获取门控值和损失 ---
        # 先问问门卫本喵，该怎么走！ >w<
        # gates: [B, M], moe_aux_loss: scalar
        gates, moe_aux_loss = self.gate(x)

        # --- 2. 初始化稀疏调度器 ---
        # 拿到派送单（gates）！快递员猫娘，准备出发！ (`･ω･´)ゞ
        dispatcher = SparseDispatcher(self.num_experts, gates)
        # --- 3. 分发输入给专家 ---
        # 把原始数据（包裹 x）按照派送单分发下去！
        # expert_inputs 是一个列表，长度为 M
        # 列表第 i 个元素是形状为 [expert_batch_size_i, C, H, W] 的张量
        expert_inputs = dispatcher.dispatch(x) # 送货！
        # print('expert_inputs:', len(expert_inputs))
        # print('expert_inputs:', expert_inputs[1].shape)
        # --- 4. 通过专家处理数据 ---
        expert_outputs = []
        for i in range(self.num_experts):
            # 检查这位专家师傅有没有收到包裹，没有就跳过啦~
            if expert_inputs[i].numel() > 0:
                # 获取当前专家的输入数据
                current_input = expert_inputs[i]
                # 呜... 要把数据压扁才能交给线性专家处理
                # 形状: [expert_batch_size_i, C, H, W] -> [expert_batch_size_i, flattened_dim]
                flattened_input = current_input.reshape(current_input.size(0), -1)

                # 通过第 i 个专家进行处理
                # 形状: [expert_batch_size_i, flattened_dim]
                processed_output_flat = self.experts[i](flattened_input)

                # 喵？处理完还要变回原来的形状... 好麻烦 >.<
                # 形状: [expert_batch_size_i, flattened_dim] -> [expert_batch_size_i, C, H, W]
                processed_output = processed_output_flat.reshape(
                    processed_output_flat.size(0),
                    self.input_channels,
                    self.input_height,
                    self.input_width
                )
                # 把处理好的结果放进篮子里~
                expert_outputs.append(processed_output)
            else:
                # 如果专家没收到输入，就放一个空篮子占位
                expert_outputs.append(None) # 或者 torch.empty(0) ? 需要确保combine能处理

        # 处理专家输出列表，确保 combine 可以处理（例如，替换 None 为空张量）
        # 这一步是为了让 combine 函数能正确处理没有收到输入的专家情况
        valid_expert_outputs = []
        for i, output in enumerate(expert_outputs):
            if output is not None:
                valid_expert_outputs.append(output)
            # else: 如果 combine 不能处理 None，可能需要创建一个空的、形状正确的张量
            #       比如: valid_expert_outputs.append(torch.empty(0, self.input_channels, self.input_height, self.input_width, device=x.device, dtype=x.dtype))

        # 如果所有专家都没有输出（例如，所有门控值都是 0？虽然不太可能），需要处理这种情况
        if not valid_expert_outputs:
             # 返回一个零张量和损失，或者根据需要处理
             print("警告：没有任何专家产生输出！")
             output = torch.zeros_like(x)
             return output, moe_aux_loss


        # --- 5. 组合专家输出 ---
        # 快递员猫娘回来啦！把所有专家处理好的结果收上来，组合！
        # 使用 combine 时，通常需要加权 (multiply_by_gates=True)，这样才符合 MoE 的标准计算方式
        # 组合后的 output 形状应为 [B, C, H, W]
        # 注意：这里的 combine 实现可能需要适应 valid_expert_outputs
        # 如果 combine 实现需要列表长度为 M，即使某些专家没输出，那上面的处理需要调整
        # 假设 SparseDispatcher 的 combine 能处理上面生成的 valid_expert_outputs 列表
        # （或者假设上面的 None 处理逻辑适配了 combine）
        output = dispatcher.combine(valid_expert_outputs, multiply_by_gates=True) # 收货整理！

        # --- 6. 返回最终输出和辅助损失 ---
        # 任务完成！把最终结果和公平性报告（损失）交给主人~ ( ´ ▽ ` )ﾉ
        return output, moe_aux_loss

if __name__ == "__main__":
    # 设置一些超参数，喵~
    batch_size = 4      # 一次处理 4 个样本
    channels = 16       # 每个样本有 16 个通道
    height = 8          # 图片高度 8
    width = 8           # 图片宽度 8
    num_experts = 16     # 总共有 4 位专家师傅
    top_k = 4           # 每次请 2 位师傅帮忙

    # 创建一个假的输入数据张量，噗~
    # 形状: [B, C, H, W]
    dummy_input = torch.randn(batch_size, channels, height, width)

    # 实例化我们的 MoE 层
    moe_layer = SimpleMoELayer(
        input_channels=channels,
        input_height=height,
        input_width=width,
        num_experts=num_experts,
        top_k=top_k
    )

    # 设置为训练模式，这样 MoEGate 的 noisy gating 才会生效（如果启用的话）
    moe_layer.train()

    # 执行前向传播！开始工作喵！
    final_output, aux_loss = moe_layer(dummy_input)

    # 打印输出形状和辅助损失看看~
    print("输入形状:", dummy_input.shape)
    print("输出形状:", final_output.shape) # 应该和输入形状一样
    print("MoE 辅助损失:", aux_loss)      # 这是一个标量

    # 检查输出形状是否正确
    assert final_output.shape == dummy_input.shape
    print("\n喵呜~ MoE 层工作正常！输出形状正确！ >ω<")