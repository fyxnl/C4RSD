import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerBlock
from mamba import Vim


class GaussianFilter(nn.Module):
    def __init__(self, cutoff_ratio=0.1):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio

    def create_filter(self, H, W, C, filter_type):
        cutoff = min(H, W) * self.cutoff_ratio
        x = torch.linspace(-W // 2, W // 2, W)
        y = torch.linspace(-H // 2, H // 2, H)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        gaussian = torch.exp(-((X**2 + Y**2) / (2 * (cutoff ** 2))))

        if filter_type == 'lowpass':
            return gaussian.unsqueeze(0).unsqueeze(0).expand(1, C, -1, -1).clone().detach()
        elif filter_type == 'highpass':
            return (1 - gaussian).unsqueeze(0).unsqueeze(0).expand(1, C, -1, -1).clone().detach()

    def forward(self, x):
        _, C, H, W = x.shape
        device = x.device
        x_freq = torch.fft.fft2(x, norm='backward')
        x_freq = torch.fft.fftshift(x_freq)

        lowpass_filter = self.create_filter(H, W, C, 'lowpass').to(device)
        x_freq_lowpass = x_freq * lowpass_filter
        x_freq_lowpass = torch.fft.ifftshift(x_freq_lowpass)
        LFI = torch.fft.ifft2(x_freq_lowpass, norm='backward').real

        highpass_filter = self.create_filter(H, W, C, 'highpass').to(device)
        x_freq_highpass = x_freq * highpass_filter
        x_freq_highpass = torch.fft.ifftshift(x_freq_highpass)
        HFI = torch.fft.ifft2(x_freq_highpass, norm='backward').real
        return LFI, HFI

class InverseGaussianFilter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, HFI, LFI):
        HFI_freq = torch.fft.fft2(HFI, norm='backward')
        HFI_freq = torch.fft.fftshift(HFI_freq)

        LFI_freq = torch.fft.fft2(LFI, norm='backward')
        LFI_freq = torch.fft.fftshift(LFI_freq)

        x_freq_combined = HFI_freq + LFI_freq

        x_freq_combined = torch.fft.ifftshift(x_freq_combined)
        x_reconstructed = torch.fft.ifft2(x_freq_combined, norm='backward').real

        return x_reconstructed


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
        x = x.contiguous()
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        
        conv_outs = [conv(x) for conv in self.convs]
        x = torch.sum(torch.stack(conv_outs), dim=0)
        
        return self.sigmoid(x)


class PromptAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention(kernel_sizes=[3, 5, 7])

    def forward(self, x, prompt):

        # 计算通道和空间注意力权重
        w1 = self.ca(prompt)
        w2 = self.sa(prompt)

        x = x * w1
        x = x * w2

        return x

class TrambawaveBlock(nn.Module):
    def __init__(self, dim, cA_scan_mode):
        super().__init__()
        self.cA_scan_mode = cA_scan_mode
        self.gaussian_filter = GaussianFilter()
        self.pa = PromptAttention(dim=dim)
        self.mamba1 = Vim(dim=dim,  # embeding dim, expand channel = input_channel*patch_size**2
                        dim_inner=dim,  # =dim
                        dt_rank=16,  # Rank of the dynamic routing matrix#不占显存
                        d_state=16,  # Dimension of the state vector#超级占显存
                        patch_size=2,  # seq shrink patch_size**2
                        dropout=0.1,  # Dropout rate
                        depth=1)  # Depth of the transformer model
        self.mamba2 = Vim(dim=dim,dim_inner=dim, dt_rank=16,d_state=16, patch_size=2, dropout=0.1, depth=1)  
        self.mamba3 = Vim(dim=dim,dim_inner=dim, dt_rank=16,d_state=16, patch_size=2, dropout=0.1, depth=1)
        self.mamba4 = Vim(dim=dim,dim_inner=dim, dt_rank=16,d_state=16, patch_size=2, dropout=0.1, depth=1)
        self.transformer = TransformerBlock(network_depth=6, dim=dim, num_heads=2, window_size=8)
    def forward(self, x):
        """
        x:(B C H W)
        """
        # 执行 DWT
        cA, cH, cV, cD, original_size= HaarWavelet.wavelet_transform_2d(x)

        # 然后调用 mamba
        cH_out = self.mamba1(cH, scan_mode='V')
        cV_out = self.mamba2(cV, scan_mode='H')
        cD_out = self.mamba3(cD, scan_mode='D')
        cA_out = self.mamba4(cA, scan_mode=self.cA_scan_mode)
        x_idwt = HaarWavelet.inverse_wavelet_transform_2d(cA_out, cH_out, cV_out, cD_out, original_size)
        LFI, HFI = self.gaussian_filter(x_idwt)
        x_out = self.pa(x_idwt, LFI)
        x_out = self.transformer(x_out, HFI)

        return x_out + x

class BasicBlock(nn.Module):
    def __init__(self, dim):
        super(BasicBlock, self).__init__()
        self.dim = dim
        layers = [TrambawaveBlock(dim=dim, cA_scan_mode='H'),
                  TrambawaveBlock(dim=dim, cA_scan_mode='V'),
                  TrambawaveBlock(dim=dim, cA_scan_mode='D'),]
        self.layers = nn.Sequential(*layers)
        self.ca = ChannelAttention(dim * 3)
        self.sa = SpatialAttention(kernel_sizes=[3,5,7])
    def forward(self, x):
        res1 = self.layers[0](x)
        res2 = self.layers[1](res1)
        res3 = self.layers[2](res2)
        w = self.ca(torch.cat([res1,res2,res3], dim=1))
        w = w.view(-1, 3, self.dim)[:, :, :, None, None]
        out = w[:,0,::] * res1 + w[:,1,::] * res2 + w[:,2,::] * res3
        w = self.sa(out)
        out = w * out
        return out + x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super().__init__()
        
        # self.gaussian_filter = GaussianFilter()
        # self.ca = ChannelAttention(dim)
        # self.sa = SpatialAttention(kernel_sizes=[3, 5, 7])

        self.height = height
        d = max(int(dim/reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False), 
            nn.ReLU(),
            nn.Conv2d(d, dim*height, 1, bias=False)
        )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        
        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)
        
        feats_sum = torch.sum(in_feats, dim=1)
        # lfi, hfi = self.gaussian_filter(feats_sum)
        # w1 = self.ca(lfi)
        # w2 = self.sa(hfi)
        # feats_sum = feats_sum * w1
        # feats_sum = feats_sum * w2
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats*attn, dim=1)
        return out      


class Trambawave(nn.Module):
    def __init__(self, window_size=8,
        embed_dims=[24, 48, 96, 48, 24]):
        super().__init__()

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=3, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        self.layer1 = BasicBlock(dim=embed_dims[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicBlock(dim=embed_dims[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicBlock(dim=embed_dims[2])

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicBlock(dim=embed_dims[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])			

        self.layer5 = BasicBlock(dim=embed_dims[4])

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=3, embed_dim=embed_dims[4], kernel_size=3)

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        base_number = 16 # 2级编解码，一级小波，一级mamba压缩
        mod_pad_h = (base_number - h % base_number) % base_number
        mod_pad_w = (base_number - w % base_number) % base_number
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        # print(f'1{x.shape}')
        x = self.patch_embed(x)# B 3 H W -> B 24 H W
        x = self.layer1(x)# B 24 H W
        skip1 = x

        x = self.patch_merge1(x) # B 24 H W -> B 48 H/2 W/2
        x = self.layer2(x)# B 48 H/2 W/2
        skip2 = x

        x = self.patch_merge2(x)# B 48 H/2 W/2 -> B 96 H/4 W/4
        x = self.layer3(x) # B 96 H/4 W/4
        x = self.patch_split1(x) # B 96 H/4 W/4 -> B 48 H/2 W/2

        x = self.fusion1([x, self.skip2(skip2)]) + x #B 48 H/2 W/2 + B 48 H/2 W/2 -> B 48 H/2 W/2
        x = self.layer4(x)# B 48 H/2 W/2
        x = self.patch_split2(x) # B 48 H/2 W/2 -> B 24 H W

        x = self.fusion2([x, self.skip1(skip1)]) + x # B 24 H W + B 24 H W -> B 24 H W
        x = self.layer5(x) # B 24 H W
        x = self.patch_unembed(x) # B 24 H W ->  B 3 H W
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        x = self.forward_features(x)
        x = x[:, :, :H, :W]
        return x


def build_net():
    return Trambawave(
        embed_dims=[24, 48, 96, 48, 24])

if __name__ == "__main__":

    x = torch.randn(size=(1, 3, 128, 128)).cuda()
    net = build_net().cuda()
    print(net(x).shape)
