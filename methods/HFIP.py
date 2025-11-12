import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, relu, bias = True):
        super(BasicConv, self).__init__()

        padding = kernel_size // 2
        layers = list()
        layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias = bias))
        layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BasicUnit(nn.Module):
    def __init__(self, in_channel, out_channel, scale_factor=2, upsample=False, kernel_size = 3, stride = 1, downsample=False, method='nearest'):
        super(BasicUnit, self).__init__()
        self.upsample = upsample
        self.downsample = downsample

        if upsample or downsample:
            self.scale_factor = scale_factor
            self.method = method

        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=kernel_size, stride=stride, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=kernel_size, stride=stride, relu=True)
        self.conv3 = BasicConv(out_channel, out_channel, kernel_size=kernel_size, stride=stride, relu=True)
    def forward(self, x):
        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.method)
        if self.downsample:
            x = nn.functional.interpolate(x, scale_factor=1/self.scale_factor, mode=self.method)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x

class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth

        self.V = nn.Conv2d(dim, dim, 1)
        self.Q = nn.Conv2d(dim, dim, 1)
        self.K = nn.Conv2d(dim, dim, 1)
        self.attn = WindowAttention(dim, window_size, num_heads)
            
        self.proj = nn.Conv2d(dim, dim, 1)
        self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape
            
            if w_shape[0] == self.dim * 2:	# QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)		
            else:
                gain = (8 * self.network_depth) ** (-1/4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size-self.shift_size+mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size-self.shift_size+mod_pad_h) % self.window_size), mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X, prompt):
        B, C, H, W = X.shape
        V = self.V(X)

        Q = self.Q(X) + prompt
        # Q = self.Q(X)
        K = self.K(X)
        QKV = torch.cat([Q, K, V], dim=1)

        # shift
        shifted_QKV = self.check_size(QKV, self.shift_size > 0)
        Ht, Wt = shifted_QKV.shape[2:]

        # partition windows
        shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
        qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

        attn_windows = self.attn(qkv)

        # merge windows
        shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

        # reverse cyclic shift
        out = shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]
        attn_out = out.permute(0, 3, 1, 2)

        conv_out = self.conv(V)
        out = self.proj(conv_out + attn_out)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_sizes=[3, 5, 7]):
        super(SpatialAttention, self).__init__()
        assert all(k in (3, 5, 7) for k in kernel_sizes), 'kernel sizes must be 3, 5, or 7'
        self.convs = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False) for k in kernel_sizes
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        
        conv_outs = [conv(x) for conv in self.convs]
        x = torch.sum(torch.stack(conv_outs), dim=0)
        
        return self.sigmoid(x)
class GaussianFilter(nn.Module):
    def __init__(self, filter_type='lowpass', cutoff_ratio=0.1):
        super(GaussianFilter, self).__init__()
        self.filter_type = filter_type
        self.cutoff_ratio = cutoff_ratio
        self.gaussian = None

    def create_filter(self, H, W, C):
        cutoff = min(H, W) * self.cutoff_ratio
        x = torch.linspace(-W // 2, W // 2, W)
        y = torch.linspace(-H // 2, H // 2, H)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        gaussian = torch.exp(-((X**2 + Y**2) / (2 * (cutoff ** 2))))

        if self.filter_type == 'lowpass':
            self.gaussian = gaussian.unsqueeze(0).unsqueeze(0).expand(1, C, -1, -1).clone().detach()
        elif self.filter_type == 'highpass':
            self.gaussian = (1 - gaussian).unsqueeze(0).unsqueeze(0).expand(1, C, -1, -1).clone().detach()

    def forward(self, x):
        _, C, H, W = x.shape
        if self.gaussian is None or self.gaussian.shape[-2:] != (H, W):
            self.create_filter(H, W, C)
        return x * self.gaussian.to(x.device)
    
class FrequencyAttention(nn.Module):
    def __init__(self, in_planes):
        super(FrequencyAttention, self).__init__()
        self.highpass_filter = GaussianFilter(filter_type='highpass')
        self.lowpass_filter = GaussianFilter(filter_type='lowpass')
        self.channel_attention = ChannelAttention(in_planes)
        self.spatial_attention = SpatialAttention(kernel_sizes=[3, 5, 7])

    def forward(self, x):

        # 对图像进行傅里叶变换
        x_freq = torch.fft.fft2(x, norm='backward')
        x_freq = torch.fft.fftshift(x_freq)

        x_freq_highpass = self.highpass_filter(x_freq)
        x_freq_lowpass = self.lowpass_filter(x_freq)

        # 进行频域处理
        x_freq_highpass = torch.fft.ifftshift(x_freq_highpass)
        x_freq_lowpass = torch.fft.ifftshift(x_freq_lowpass)
        x_freq_highpass = torch.fft.ifft2(x_freq_highpass, norm='backward').abs()
        x_freq_lowpass = torch.fft.ifft2(x_freq_lowpass, norm='backward').abs()

        # 计算通道和空间注意力权重
        w1 = self.channel_attention(x_freq_lowpass)
        w2 = self.spatial_attention(x_freq_highpass)

        x = x * w1
        x = x * w2
        prompt = x_freq_highpass

        return x, prompt



class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads,
                 window_size=8):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(network_depth, dim, num_heads=num_heads, shift_size = 0, window_size=window_size)
        self.freq_attention = FrequencyAttention(dim)

    def forward(self, x):
        x, prompt = self.freq_attention(x)
        B, C, H, W = x.shape
        identity = x
        x = x.view(B, C, H * W)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = x.view(B, C, H, W)
        x = self.attn(x, prompt)
        x = identity + x

        return x

class BasicBlock(nn.Module):
    def __init__(self, dim, Block_Number):
        super(BasicBlock, self).__init__()
        self.dim = dim
        self.Block_Number = Block_Number
        layers = [TransformerBlock(network_depth=6, 
                                    dim=dim, 
                                    num_heads=2, 
                                    window_size=8)
            for _ in range(Block_Number)]
        self.layers = nn.Sequential(*layers)
        self.ca = ChannelAttention(dim * Block_Number)
        self.sa = SpatialAttention(kernel_sizes=[3,5,7])
    def forward(self, x):
        res1 = self.layers[0](x)
        res2 = self.layers[1](res1)
        res3 = self.layers[2](res2)
        w = self.ca(torch.cat([res1,res2,res3], dim=1))
        w = w.view(-1, self.Block_Number, self.dim)[:, :, :, None, None]
        out = w[:,0,::] * res1 + w[:,1,::] * res2 + w[:,2,::] * res3
        w = self.sa(out)
        out = w * out
        return out + x

class HFIP(nn.Module):
    def __init__(self, Block_Number=3):
        super(HFIP, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            BasicBlock(base_channel, Block_Number),
            BasicBlock(base_channel * 2, Block_Number),
            BasicBlock(base_channel * 4, Block_Number),
        ])

        self.feat_extract = nn.ModuleList([
            BasicUnit(base_channel, base_channel * 2, scale_factor=2, downsample=True),
            BasicUnit(base_channel * 2, base_channel * 4, scale_factor=2, downsample=True),
            BasicUnit(base_channel * 4, base_channel * 2, scale_factor=2, upsample=True),
            BasicUnit(base_channel * 2, base_channel, scale_factor=2, upsample=True),
        ])
        self.Decoder = nn.ModuleList([
            BasicBlock(base_channel * 4, Block_Number),
            BasicBlock(base_channel * 2, Block_Number),
            BasicBlock(base_channel, Block_Number)
        ])
        pre_process = [BasicConv(3, base_channel, kernel_size=3, stride=1, relu=True)]
        post_precess = [
            BasicConv(base_channel, base_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(base_channel, 3, kernel_size=3, stride=1, relu=True)]
        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)
        self.fc = nn.ModuleList([
            BasicUnit(base_channel * 4, base_channel * 2, kernel_size=1),
            BasicUnit(base_channel * 2, base_channel, kernel_size=1),
        ])

    def forward(self, x):
        # 处理输入图像
        x_ = self.pre(x)
        res1 = self.Encoder[0](x_)
        # 128*128
        z = self.feat_extract[0](res1)
        res2 = self.Encoder[1](z)
        # 64*64
        z = self.feat_extract[1](res2)
        z = self.Encoder[2](z)

        z = self.Decoder[0](z)
        # 128*128
        z = self.feat_extract[2](z)
        z = torch.cat([z, res2], dim=1)
        z = self.fc[0](z)
        z = self.Decoder[1](z)
        # 256*256
        z = self.feat_extract[3](z)

        z = torch.cat([z, res1], dim=1)
        z = self.fc[1](z)
        z = self.Decoder[2](z)
        z = self.post(z)

        return z  # 只返回最后的输出

def build_net():
    return HFIP()

if __name__ == "__main__":

    # x = torch.randn(size=(4, 64, 256, 256)).cuda()
    # block = ProcessBlock(in_nc=64).cuda()
    # for i in range(100000):
    #     print(block(x).size())

    x = torch.randn(size=(1, 3, 128, 128)).cuda()
    net = build_net(Block_Number=6).cuda()
    # net.eval()
    # print(net)
    # exit()
    for i in range(10000):
        print(net(x)[0].size())
