import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------
# Channel Attention (CA) 模块
# ----------------------------------------

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # 全局平均池化: C x H x W -> C x 1 x 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 通道注意力机制: 两层全连接 (MLP) + Sigmoid
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)  # C x 1 x 1
        y = self.conv_du(y)   # C x 1 x 1，缩放系数
        return x * y          # 逐通道缩放
'''

# 替换为 CBAM 注意力机制，既保留通道，又注意空间
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CALayer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x
'''

# ----------------------------------------
# 基本卷积
# ----------------------------------------
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


# ----------------------------------------
# Residual Channel Attention Block (RCAB)
# ----------------------------------------
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(True)):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# ----------------------------------------
# Dense RCAB
# ----------------------------------------
'''
class DenseRCABBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size=3, reduction=16, n_resblocks=20):
        super(DenseRCABBlock, self).__init__()
        self.n_resblocks = n_resblocks
        self.rcabs = nn.ModuleList()
        self.compress_layers = nn.ModuleList()

        for i in range(n_resblocks):
            in_ch = n_feat * (i + 1)
            # 压缩到固定通道数
            self.compress_layers.append(nn.Conv2d(in_ch, n_feat, 1))
            self.rcabs.append(RCAB(conv, n_feat, kernel_size, reduction))

    def forward(self, x):
        feats = [x]
        for i in range(self.n_resblocks):
            inp = torch.cat(feats, dim=1)
            inp_compressed = self.compress_layers[i](inp)
            out = self.rcabs[i](inp_compressed)
            feats.append(out)
        return feats[-1]  # 返回最后一个RCAB的输出
'''

# ----------------------------------------
# Residual Group (RG)
# ----------------------------------------

class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True))
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res



# ----------------------------------------
# Dense RG
# ----------------------------------------
'''
class DenseResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, n_resblocks):
        super(DenseResidualGroup, self).__init__()
        self.dense_block = DenseRCABBlock(conv, n_feat, kernel_size, reduction, n_resblocks)
        self.conv_last = conv(n_feat, n_feat, kernel_size)

    def forward(self, x):
        res = self.dense_block(x)
        res = self.conv_last(res)
        res += x
        return res
'''

# ----------------------------------------
# RCAN 主结构
# ----------------------------------------

class RCAN(nn.Module):
    def __init__(self, n_resgroups=10, n_resblocks=20, n_feats=64, reduction=16, scale=2):
        super(RCAN, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)

        def default_conv(in_channels, out_channels, kernel_size, bias=True):
            return nn.Conv2d(
                in_channels, out_channels, kernel_size,
                padding=(kernel_size // 2), bias=bias)

        # 1. 浅层特征提取
        self.head = default_conv(3, n_feats, kernel_size)

        # 2. 深层特征提取（Residual in Residual）
        modules_body = [
            ResidualGroup(default_conv, n_feats, kernel_size, reduction, act, n_resblocks)
            for _ in range(n_resgroups)
        ]
        modules_body.append(default_conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # 3. 上采样层
        modules_tail = [
            nn.Conv2d(n_feats, n_feats * (scale ** 2), kernel_size, padding=kernel_size // 2),
            nn.PixelShuffle(scale),
            default_conv(n_feats, 3, kernel_size)
        ]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        # Shallow feature
        x_head = self.head(x)

        # Deep feature: RIR
        res = self.body(x_head)
        res += x_head

        # Reconstruction
        x_out = self.tail(res)

        return x_out



# ----------------------------------------
# Dense RCAN
# ----------------------------------------
'''
class DenseRCAN(nn.Module):
    def __init__(self, n_resgroups=10, n_resblocks=20, n_feats=64, reduction=16, scale=2):
        super(DenseRCAN, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)

        # 1. 浅层特征提取
        self.head = default_conv(3, n_feats, kernel_size)

        # 2. 深层特征提取（Dense Residual in Dense Residual）
        self.groups = nn.ModuleList()
        self.group_compress = nn.ModuleList()

        for i in range(n_resgroups):
            in_ch = n_feats * (i + 1)
            self.group_compress.append(nn.Conv2d(in_ch, n_feats, 1))
            self.groups.append(DenseResidualGroup(default_conv, n_feats, kernel_size, reduction, act, n_resblocks))

        # 最后一个卷积
        self.body_last = default_conv(n_feats, n_feats, kernel_size)

        # 3. 上采样层
        modules_tail = [
            nn.Conv2d(n_feats, n_feats * (scale ** 2), kernel_size, padding=kernel_size // 2),
            nn.PixelShuffle(scale),
            default_conv(n_feats, 3, kernel_size)
        ]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x_head = self.head(x)
        feats = [x_head]

        for i, group in enumerate(self.groups):
            inp = torch.cat(feats, dim=1)
            inp_compressed = self.group_compress[i](inp)
            out = group(inp_compressed)
            feats.append(out)

        res = feats[-1]
        res = self.body_last(res)
        res += x_head
        x_out = self.tail(res)
        return x_out
'''

# ----------------------------------------
# 测试模型结构
# ----------------------------------------
if __name__ == '__main__':

    model = RCAN(n_resgroups=10, n_resblocks=20, n_feats=64, scale=2)  # 小规模配置方便测试
    print(model)
    x = torch.randn(1, 3, 24, 24)
    y = model(x)
    print(y.shape)  # 应该是 1 x 3 x (24*scale) x (24*scale)
    '''
    model = DenseRCAN(n_resgroups=3, n_resblocks=4, n_feats=64, scale=2)  # 小规模测试
    print(model)
    inp = torch.randn(1, 3, 24, 24)
    out = model(inp)
    print(out.shape)
    '''