import torch
from torch import nn

class SimpleChannelAttentionFusion(nn.Module):
    """
    简单的通道注意力融合模块：根据拼接特征生成权重，加权融合。
    """
    def __init__(self, in_channels):
        super().__init__()
        # 使用 AdaptiveAvgPool2d 获取全局通道信息
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 使用 1x1 卷积学习通道重要性，并预测一个模态的权重（另一个是 1-它）
        # 输入通道是 2 * in_channels (因为拼接了 vi 和 ii)
        # 输出通道是 in_channels (为每个通道生成一个权重)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, 1, bias=False), # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 1, bias=False),    # 升维回原通道数
            nn.Sigmoid() # 使用 Sigmoid 得到 0-1 之间的权重
        )

    def forward(self, feat_vis, feat_ir):
        # feat_vis, feat_ir 形状都是 (N, C, H, W)
        combined = torch.cat((feat_vis, feat_ir), dim=1) # (N, 2C, H, W)
        weights = self.fc(self.avg_pool(combined)) # (N, C, 1, 1)
        # weights 代表了 feat_vis 的权重
        fused_feat = weights * feat_vis + (1 - weights) * feat_ir
        return fused_feat

class FiLMLayer(nn.Module):
    # ... (同之前的 FiLM 实现) ...
    def __init__(self, text_dim, feature_channels):
        super().__init__()
        self.gamma_mlp = nn.Linear(text_dim, feature_channels)
        self.beta_mlp = nn.Linear(text_dim, feature_channels)

    def forward(self, feature_map, text_feature):
        gamma = self.gamma_mlp(text_feature).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_mlp(text_feature).unsqueeze(-1).unsqueeze(-1)
        return gamma * feature_map + beta


class TextEnhancedSegHead(nn.Module):
    def __init__(self, in_channels, intermediate_channels, n_class, text_dim):
        super().__init__()
        # 1. 基础卷积块
        self.base_conv = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU()
        )
        # 2. FiLM 层
        self.film = FiLMLayer(text_dim, intermediate_channels)
        # 3. 最终输出卷积
        self.final_conv = nn.Conv2d(intermediate_channels, n_class, 1)

    # --- forward 需要接收 y1 和 text_feature ---
    def forward(self, y1, text_feature):
        seg_feat_base = self.base_conv(y1)
        modulated_feat = self.film(seg_feat_base, text_feature)
        seg_res = self.final_conv(modulated_feat)
        return seg_res


class reflect_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, pad=1):
        super(reflect_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)
        )

    def forward(self, x):
        out = self.conv(x)
        return out



class Mamba_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, pad=1):
        super(Mamba_block, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)
        )
        self.Mamba = SingleMambaBlock(dim=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.Mamba(x) + x
        return x


def gradient(input):
    """
    求图像梯度, sobel算子
    :param input:
    :return:
    """

    filter1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter2 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter1.weight.data = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ]).reshape(1, 1, 3, 3).cuda()
    filter2.weight.data = torch.tensor([
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]
    ]).reshape(1, 1, 3, 3).cuda()

    g1 = filter1(input)
    g2 = filter2(input)
    image_gradient = torch.abs(g1) + torch.abs(g2)
    return image_gradient


def clamp(value, min=0., max=1.0):
    """
    将像素值强制约束在[0,1], 以免出现异常斑点
    :param value:
    :param min:
    :param max:
    :return:
    """
    return torch.clamp(value, min=min, max=max)


def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式

    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[:,0:1,:,:]
    G = rgb_image[:,1:2,:,:]
    B = rgb_image[:,2:3,:,:]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = clamp(Y)
    Cr = clamp(Cr)
    Cb = clamp(Cb)
    return Y, Cb, Cr


def YCrCb2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式

    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = clamp(out)
    return out


def batch_YCrCb2RGB(Y_batch, Cb_batch, Cr_batch):
    """
    对 batch 的 YCrCb 图像进行批量转换，向量化处理
    :param Y_batch: Y 通道，形状为 (b, 1, w, h)
    :param Cb_batch: Cb 通道，形状为 (b, 1, w, h)
    :param Cr_batch: Cr 通道，形状为 (b, 1, w, h)
    :return: RGB 图像，形状为 (b, 3, w, h)
    """
    # 将 Y, Cb, Cr 拼接在通道维度上 -> (b, 3, w, h)
    ycrcb_batch = torch.cat([Y_batch, Cr_batch, Cb_batch], dim=1)

    # 展平每个图片为 (b, 3, w*h)，然后转置 -> (b, w*h, 3)
    im_flat = ycrcb_batch.view(ycrcb_batch.shape[0], 3, -1).transpose(1, 2)

    # 定义转换矩阵和偏置
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]).to(Y_batch.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y_batch.device)

    # 执行矩阵运算，批量转换 YCrCb -> RGB
    temp = (im_flat + bias).matmul(mat)

    # 恢复图像形状 (b, 3, w, h)
    out = temp.transpose(1, 2).view(Y_batch.shape[0], 3, Y_batch.shape[2], Y_batch.shape[3])

    # 裁剪到有效范围
    out = clamp(out)

    return out

