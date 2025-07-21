from typing import Tuple

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import numbers
from models.FMoEGate import FusedMoEGate,SparseDispatcher,GatingFusionMoEGate

class DepthwiseSeparableConv(nn.Module):
    # ... (之前的定义) ...
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False): # 通常 DSC 不用偏置
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
                    torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class FFTAttentionLite(nn.Module):
    """轻量级频域注意力（仅1层Conv+频域调制）"""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 1)  # 复用原有维度

    def forward(self, x):
        # 1. 将输入转换为 float32 以增强数值稳定性
        x = x.float()

        # 2. 频域变换
        x_fft = torch.fft.rfft2(x, norm='ortho')

        # 3. 动态滤波（仅用振幅谱）
        x_mag = torch.abs(x_fft)
        gate = torch.tanh(self.proj(x_mag))  # 使用 tanh 扩大输出范围

        # 4. 调制频域信号
        x_fft = x_fft * gate # 保持复数特性

        # 5. 逆变换并转换回原始精度
        x_out = torch.fft.irfft2(x_fft, s=x.shape[-2:], norm='ortho')
        return x_out.to(x.dtype)  # 转换回原始精度

# class SobelExpert(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.sobel_x = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
#         self.sobel_y = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
#         # 初始化 Sobel 核
#         self._init_sobel_kernel()
#
#     def _init_sobel_kernel(self):
#         # Sobel 核
#         sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
#         sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
#         # 扩展为多通道卷积核
#         sobel_kernel_x = sobel_kernel_x.view(1, 1, 3, 3).repeat(self.sobel_x.out_channels, 1, 1, 1)
#         sobel_kernel_y = sobel_kernel_y.view(1, 1, 3, 3).repeat(self.sobel_y.out_channels, 1, 1, 1)
#         self.sobel_x.weight.data = sobel_kernel_x
#         self.sobel_y.weight.data = sobel_kernel_y
#         self.sobel_x.weight.requires_grad = False
#         self.sobel_y.weight.requires_grad = False
#
#     def forward(self, x):
#         grad_x = self.sobel_x(x)
#         grad_y = self.sobel_y(x)
#         edge_map = torch.sqrt(grad_x**2 + grad_y**2)  # 计算边缘强度
#         return edge_map

class ExpertDilatedConv(nn.Module):
    """
    Expert using Conv2d with a specific dilation rate.
    Adds residual connection.
    """
    def __init__(self, dim, hidden_dim_factor=4, dilation=1):
        super().__init__()
        hidden_dim = int(dim * hidden_dim_factor)
        # Best practice: Expand channels, process, then contract back
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1), # Pointwise expansion
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation, groups=hidden_dim if hidden_dim % dim == 0 and hidden_dim >= dim else 1), # Depthwise or Grouped Conv with dilation
            # Use standard conv if grouping is not straightforward
            # nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1)  # Pointwise contraction
        )

    def forward(self, x):
        return self.net(x) + x # Residual connection
#-----------------------基于MoE的特征提取，提高参数利用率，保证整体计算量不提升过多-------------#
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEFeedForward(nn.Module):
    def __init__(self, dim, num_experts=4, top_k=1, noise_std=0.1):
        super(MoEFeedForward, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std  # 噪声标准差

        # 定义多个专家，每个专家是 (B, C, W, H) -> (B, C, W, H)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1)
            ) for _ in range(num_experts)
        ])

        # 门控网络（根据 text_feature 选择专家）
        self.gating_network = nn.Linear(512, num_experts, bias=False)

    def forward(self, x, text_feature, training=True):
        B, C, H, W = x.shape  # 保持 (B, C, H, W) 格式

        # 计算 gating 权重 (B, num_experts)，根据 text_feature 计算专家选择概率
        gate_logits = self.gating_network(text_feature)  # (B, num_experts)

        if training:
            # 添加噪声扰动 (Gumbel Noise)
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise  # (B, num_experts)

        gate_weights = F.softmax(gate_logits, dim=-1)  # (B, num_experts)

        # 选择 top-k 专家
        topk_values, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)  # (B, top_k)

        # 初始化 MoE 输出
        moe_output = torch.zeros_like(x)  # (B, C, H, W)

        # 仅计算 Top-k 专家
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]  # (B,)
            weight = topk_values[:, i].view(B, 1, 1, 1)  # (B, 1, 1, 1) 用于加权

            # 计算当前专家输出，仅在选中的 batch 进行计算
            for j in range(self.num_experts):
                mask = (expert_idx == j).view(B, 1, 1, 1)  # 选中的 batch
                if mask.any():
                    moe_output += weight * mask * self.experts[j](x)  # (B, C, H, W)

        return moe_output

class MTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, LayerNorm_type='WithBias'):
        super(MTransformerBlock, self).__init__()

        self.attn = Attention(dim=dim, bias=bias, num_heads=num_heads)

        # 替换 FFN 为 MoEFFN
        self.moe_ffn = MoEFeedForward(dim)

    def forward(self, x, text_feature):
        # print("MTransformerBlock")
        # print("x",x.shape)
        x = x + self.attn(x)
        x = x + self.moe_ffn(x, text_feature)

        return x

class SparseMoEFeedForward(nn.Module): # 改个名字区分一下
    # 在 __init__ 中，传入 text_feature 的维度 d_text
    def __init__(self, dim, d_text=512, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim

        # --- 使用新的 FusedMoEGate ---
        self.gate = FusedMoEGate(d_x=dim, d_text=d_text, M=num_experts, K=top_k)
        # self.gate = GatingFusionMoEGate(d_x=dim, d_text=d_text, M=num_experts, K=top_k)
        # self.gate = TextOnlyMoEGate(d_text=d_text, M=num_experts, K=top_k)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1)
            ) for _ in range(num_experts)
        ])

    # --- forward 方法需要接收并传递 text_feature ---
    def forward(self, x: torch.Tensor, text_feature: torch.Tensor):
        batch_size, channels, height, width = x.shape
        assert channels == self.dim

        # --- 调用 FusedMoEGate ---
        gates, moe_aux_loss = self.gate(x, text_feature) # 把 text_feature 传进去
        # gates, moe_aux_loss = self.gate(text_feature) # 把 text_feature 传进去

        # --- 后续的 dispatcher, dispatch, expert 计算, combine 逻辑不变 ---
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [None] * self.num_experts
        for i in range(self.num_experts):
            if expert_inputs[i] is not None and expert_inputs[i].shape[0] > 0:
                # print("i:",i)
                # print("expert_inputs[i]",expert_inputs[i].shape)
                expert_outputs[i] = self.experts[i](expert_inputs[i])

        valid_expert_outputs = [output for output in expert_outputs if output is not None]
        if not valid_expert_outputs:
            print("警告喵！没有任何专家产生输出！返回零张量！")
            output = torch.zeros_like(x)
        else:
            output = dispatcher.combine(valid_expert_outputs, multiply_by_gates=True)

        # 返回组合后的输出和辅助损失
        return output, moe_aux_loss

class HomogeneousSparseMoEFeedForward(nn.Module): # 改个名字区分一下
    # 在 __init__ 中，传入 text_feature 的维度 d_text
    def __init__(self, dim, d_text=512, num_experts=16, top_k=2,expert_use_fft = True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim

        # --- 使用新的 FusedMoEGate ---
        self.gate = FusedMoEGate(d_x=dim, d_text=d_text, M=num_experts, K=top_k)
        # self.gate = GatingFusionMoEGate(d_x=dim, d_text=d_text, M=num_experts, K=top_k)
        # self.gate = TextOnlyMoEGate(d_text=d_text, M=num_experts, K=top_k)
        # self.experts = nn.ModuleList([
        #     ResidualExpert(dim, use_fft=expert_use_fft) # 使用新的 ResidualExpert 类
        #     for _ in range(num_experts)
        # ])
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1)
            ) for _ in range(num_experts)
        ])
        # 这里可能可以修改为异构专家
        # self.experts = nn.ModuleList()
        # for i in range(num_experts):
        #     expert = ExpertWithConvFusion(dim)
        #     self.experts.append(expert)
    # --- forward 方法需要接收并传递 text_feature ---
    def forward(self, x: torch.Tensor, text_feature: torch.Tensor):
        batch_size, channels, height, width = x.shape
        assert channels == self.dim

        gates, moe_aux_loss = self.gate(x, text_feature)
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x) # 得到长度为 num_experts 的列表
        # --- 正确的修复逻辑 ---
        expert_outputs = [] # 初始化一个空列表来收集所有专家的输出

        # 确定专家输出的形状（除了批次维度）
        # 因为你的专家是 Conv2d(padding=1) -> GELU -> Conv2d(padding=1)，
        # 它们应该保持通道数 (self.dim) 和空间维度 (height, width) 不变。
        output_dims = (self.dim, height, width)

        for i in range(self.num_experts):
            expert_input_i = expert_inputs[i]
            if expert_input_i is not None and expert_input_i.shape[0] > 0:
                # 如果专家有输入，计算输出
                expert_output_i = self.experts[i](expert_input_i)
                expert_outputs.append(expert_output_i)
            else:
                # 如果专家没有输入，添加一个形状正确的空张量作为占位符
                # [0, C, H, W]
                empty_output = torch.zeros(0, *output_dims, device=x.device, dtype=x.dtype)
                expert_outputs.append(empty_output)

        # 现在 expert_outputs 保证包含 self.num_experts 个张量
        # （有些是实际输出，有些是空张量）
        # 直接将完整的列表传递给 combine
        output = dispatcher.combine(expert_outputs, multiply_by_gates=True)
        # combine 内部的 torch.cat 可以处理空张量，index_add 也可以处理空 source

        return output, moe_aux_loss

class SparseTransformerBlock(nn.Module):
    """
    改进的 Transformer 块，增加了 Pre-LayerNorm (使用 GroupNorm 实现)。
    """
    # 在 __init__ 中，需要知道 d_text 以初始化 SparseMoEFeedForwardWithText
    def __init__(self, dim, d_text=512, num_heads=4, num_experts=16, top_k=2, bias=False, LayerNorm_type='WithBias', eps=1e-6): # 添加 eps 参数
        super().__init__()
        # --- 添加 LayerNorm (使用 GroupNorm 模拟) ---
        # self.norm1 = LayerNormFunction(dim, eps=eps) # 使用自定义的或直接用下面的
        # self.norm2 = LayerNormFunction(dim, eps=eps)
        self.norm1 = nn.GroupNorm(1, dim, eps=eps) # 替换为 GroupNorm
        self.norm2 = nn.GroupNorm(1, dim, eps=eps) # 替换为 GroupNorm

        self.attn = Attention(dim=dim, bias=bias, num_heads=num_heads) # 假设 Attention 不需要 text_feature
        self.moe_ffn = HomogeneousSparseMoEFeedForward(dim, d_text, num_experts=num_experts, top_k=top_k)

    # --- forward 方法需要接收并传递 text_feature ---
    def forward(self, x: torch.Tensor, text_feature: torch.Tensor):
        # --- Pre-LN 结构 ---
        # 1. Attention 部分
        residual1 = x
        x_norm1 = self.norm1(x)
        attn_output = self.attn(x_norm1)
        x = residual1 + attn_output # 第一个残差连接

        # 2. MoE FFN 部分
        residual2 = x
        x_norm2 = self.norm2(x)
        moe_output, moe_loss = self.moe_ffn(x_norm2, text_feature) # MoE 输入归一化后的特征
        x = residual2 + moe_output # 第二个残差连接

        return x, moe_loss # 返回输出和损失
# Define the learnable convolutional block function
def create_learnable_block(dim):
    """Creates the standard learnable convolutional block."""
    return nn.Sequential(
        nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv2d(dim, dim, kernel_size=3, padding=1)
    )

# --- Modified Heterogeneous Experts with Learnable Block ---

class NoRescaleExpert(nn.Module):
    """
    类型1: 不进行上下采样，保持原尺度。
    包含一个非参数化操作（AvgPool）和一个可学习的卷积块。
    """
    def __init__(self, dim): # Requires dim for Conv2d
        super().__init__()
        # Non-parameterized operation
        # Learnable convolutional block
        self.learnable_block = create_learnable_block(dim)

    def forward(self, x):
        # Apply non-parameterized op first, then the learnable block
        x = self.learnable_block(x)
        return x


class UpThenDownExpert(nn.Module):
    """
    类型2: 先上采样后下采样，使用固定因子 2。
    在尺度变换之间插入可学习的卷积块。
    """
    def __init__(self, dim, downsample_mode='max'): # Requires dim
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # Learnable block applied at the upsampled resolution
        self.learnable_block = create_learnable_block(dim)
        if downsample_mode == 'max':
            self.downsample_net = nn.MaxPool2d(kernel_size=2, stride=2)
        elif downsample_mode == 'avg':
            self.downsample_net = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError("downsample_mode must be 'max' or 'avg'")

    def forward(self, x):
        x = self.upsample(x)         # 1. Up-sample
        x = self.learnable_block(x)  # 2. Apply learnable block
        x = self.downsample_net(x)   # 3. Down-sample
        return x


class DownThenUpExpert(nn.Module):
    """
    类型3: 先下采样后上采样，使用固定因子 2。
    在尺度变换之间插入可学习的卷积块。
    """
    def __init__(self, dim, downsample_mode='max'): # Requires dim
        super().__init__()
        if downsample_mode == 'max':
            self.downsample_net = nn.MaxPool2d(kernel_size=2, stride=2)
        elif downsample_mode == 'avg':
            self.downsample_net = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError("downsample_mode must be 'max' or 'avg'")
        # Learnable block applied at the downsampled resolution
        self.learnable_block = create_learnable_block(dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.downsample_net(x)   # 1. Down-sample
        x = self.learnable_block(x)  # 2. Apply learnable block
        x = self.upsample(x)         # 3. Up-sample
        return x


class MultiScaleHeterogeneousMoEFeedForward(nn.Module):
    """
    改造后的混合专家，实现了异构的多尺度处理。
    专家类型是固定的异构：不进行上下采样、先上采样后下采样、先下采样后上采样。
    所有操作通过非参数化方式完成（使用 Upsample 和 Pooling）。
    """

    def __init__(self, dim, d_text=512, num_experts=16, top_k=2, expert_use_fft=True, downsample_mode='max'):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim  # 通道数，用于输出形状参考

        # 使用 FusedMoEGate 门控模块
        self.gate = FusedMoEGate(d_x=dim, d_text=d_text, M=num_experts, K=top_k)

        # 创建异构专家列表：按索引循环分配不同类型（i % 3）
        # - i % 3 == 0: 类型1 (不进行上下采样)
        # - i % 3 == 1: 类型2 (先上采样后下采样)
        # - i % 3 == 2: 类型3 (先下采样后上采样)
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            if i % 3 == 0:
                # 类型1: 不进行上下采样 + 可学习块
                expert = NoRescaleExpert(dim=dim)
            elif i % 3 == 1:
                # 类型2: 先上采样 -> 可学习块 -> 后下采样
                expert = UpThenDownExpert(dim=dim, downsample_mode=downsample_mode)
            else:
                # 类型3: 先下采样 -> 可学习块 -> 后上采样
                expert = DownThenUpExpert(dim=dim, downsample_mode=downsample_mode)
            self.experts.append(expert)

        # 注意：expert_use_fft 参数保留但未使用，因为专家是非参数化的。如果需要，可以移除。
        # downsample_mode 参数允许选择 'max' 或 'avg' 池化，默认为 'max'。

    def forward(self, x: torch.Tensor, text_feature: torch.Tensor):
        batch_size, channels, height, width = x.shape
        assert channels == self.dim

        # 计算门控和辅助损失
        print('x.shape',x.shape)
        gates, moe_aux_loss = self.gate(x, text_feature)
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)  # 分派输入到专家

        # 计算专家输出，确保列表长度为 num_experts
        expert_outputs = []
        output_dims = (self.dim, height, width)  # 输出形状参考（虽然专家是非参数化的，但形状保持不变）

        for i in range(self.num_experts):
            expert_input_i = expert_inputs[i]
            if expert_input_i is not None and expert_input_i.shape[0] > 0:
                # 如果有输入，计算输出
                expert_output_i = self.experts[i](expert_input_i)
                expert_outputs.append(expert_output_i)
            else:
                # 如果没有输入，添加空张量作为占位符
                empty_output = torch.zeros(0, *output_dims, device=x.device, dtype=x.dtype)
                expert_outputs.append(empty_output)

        # 组合输出
        output = dispatcher.combine(expert_outputs, multiply_by_gates=True)

        return output, moe_aux_loss


# 其余部分保持不变，例如 HeterogeneousTransformerBlock
class HeterogeneousTransformerBlock(nn.Module):
    """
    改进的 Transformer 块，增加了 Pre-LayerNorm (使用 GroupNorm 实现)。
    """

    def __init__(self, dim, d_text=512, num_heads=4, num_experts=16, top_k=2, bias=False, LayerNorm_type='WithBias',
                 eps=1e-6):
        super().__init__()
        # 使用 GroupNorm 模拟 LayerNorm
        self.norm1 = nn.GroupNorm(1, dim, eps=eps)
        self.norm2 = nn.GroupNorm(1, dim, eps=eps)

        self.attn = Attention(dim=dim, bias=bias, num_heads=num_heads)  # 假设 Attention 不需要 text_feature
        # 使用改造后的 MoE 模块
        self.moe_ffn = MultiScaleHeterogeneousMoEFeedForward(dim, d_text, num_experts=num_experts, top_k=top_k)

    def forward(self, x: torch.Tensor, text_feature: torch.Tensor):
        # Pre-LN 结构
        residual1 = x
        x_norm1 = self.norm1(x)
        attn_output = self.attn(x_norm1)
        x = residual1 + attn_output  # 第一个残差连接

        residual2 = x
        x_norm2 = self.norm2(x)
        moe_output, moe_loss = self.moe_ffn(x_norm2, text_feature)  # MoE 输入归一化后的特征
        x = residual2 + moe_output  # 第二个残差连接

        return x, moe_loss  # 返回输出和损失

if __name__ == '__main__':
    def count_parameters(model):
        """计算模型的总参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    image = torch.rand(1,16,64,64).cuda()
    text_feature = torch.rand(1,512).cuda()
    model = SMTransformerBlock(dim=16).cuda()
    total_params = count_parameters(model)
    output, moe_aux_loss = model(image, text_feature)
    print(output.shape)
    print(moe_aux_loss)
    print(f"Total number of parameters: {total_params}")



# if __name__ == '__main__':
#     image = torch.rand(1,16,64,64).cuda()
#     text_feature = torch.rand(1,512).cuda()
#     model = MTransformerBlock(dim=16).cuda()
#     print(model(image, text_feature).shape)

