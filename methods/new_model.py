import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import math


class Pre_Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Pre_Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # out_channel=512
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        print(f'input={x.size()}')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(f'after maxpool: {x.size()}')

        x = self.layer1(x)
        print(f'after layer1: {x.size}')
        x = self.layer2(x)
        print(f'after layer2: {x.size}')
        x = self.layer3(x)
        print(f'after layer3: {x.size}')
        x = self.layer4(x)
        print(f'after layer4: {x.size}')

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        print(f'x: {x.size}')
        x = self.fc(x)

        print(f'after fc output: {x.size}')

        return x


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):

        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x_layer0 = x
        x = self.maxpool(x)

        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)  # x16
        # x_layer3: torch.Size([1, 1024, 16, 16])
        # x_layer2: torch.Size([1, 512, 32, 32])
        # x_layer1: torch.Size([1, 256, 64, 64])
        # x_layer0: torch.Size([1, 64, 128, 128])
        return x_layer3, x_layer2, x_layer1, x_layer0


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class Decoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None,
                 mode='iter1'):
        super(Decoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.down_convs.append(
                ConvBlock(num_filter * (2 ** i), num_filter * (2 ** (i + 1)), kernel_size, stride, padding, bias,
                          activation, norm=None)
            )
            self.up_convs.append(
                DeconvBlock(num_filter * (2 ** (i + 1)), num_filter * (2 ** i), kernel_size, stride, padding, bias,
                            activation, norm=None)
            )

    def forward(self, ft_h, ft_l_list):
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_h_list = []
            for i in range(len(ft_l_list)):
                ft_h_list.append(ft_h)
                ft_h = self.down_convs[self.num_ft - len(ft_l_list) + i](ft_h)

            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft_fusion = self.up_convs[self.num_ft - i - 1](ft_fusion - ft_l_list[i]) + ft_h_list[
                    len(ft_l_list) - i - 1]

        if self.mode == 'iter2':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)

                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter3':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(i + 1):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[len(ft_l_list) - i - 1]
                for j in range(i + 1):
                    # print(j)
                    ft = self.up_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_h
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, scale=1.0):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        self.scale = scale
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out) * self.scale
        out = out + x
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        h = (x.shape[2] - 1) * self.stride + self.kernel_size
        w = (x.shape[3] - 1) * self.stride + self.kernel_size
        x = F.interpolate(x, size=(h, w), mode="nearest-exact")
        out = self.conv2d(x)
        return out

class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=2, out_chans=3, in_chans=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = in_chans

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, out_chans*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out
from .transformer import TransformerBlock
from .mamba import Vim
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
class TambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        layers = [TransformerBlock(dim=dim, network_depth=30, num_heads=2, window_size=8),
                  TransformerBlock(dim=dim, network_depth=30, num_heads=2, window_size=8),
                  TransformerBlock(dim=dim, network_depth=30, num_heads=2, window_size=8),]
        self.layers = nn.Sequential(*layers)
        self.ca = ChannelAttention(dim * 3)
        self.sa = SpatialAttention(kernel_sizes=[3,5,7])
    def forward(self, x, mamba16H,mamba16V,mamba16D):

        res1 = self.layers[0](x,mamba16H)
        res2 = self.layers[1](res1,mamba16V)
        res3 = self.layers[2](res2,mamba16D)
        w = self.ca(torch.cat([res1,res2,res3], dim=1))
        w = w.view(-1, 3, self.dim)[:, :, :, None, None]
        out = w[:,0,::] * res1 + w[:,1,::] * res2 + w[:,2,::] * res3
        w = self.sa(out)
        out = w * out
        return out + x
class BasicBlock(nn.Module):
    def __init__(self, dim, num_blocks=10):
        super().__init__()
        self.blocks = nn.Sequential(*[TambaBlock(dim) for _ in range(num_blocks)])

    def forward(self, x, mamba16H, mamba16V, mamba16D):
        for block in self.blocks:
            x = block(x, mamba16H, mamba16V, mamba16D)
        return x
    
class PatchUnEmbed(nn.Module):
    def __init__(self, in_chans=96, out_chans=24, patch_size=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=out_chans*patch_size**2, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, out_chans=96, patch_size=4, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x

class mamba_Branches(nn.Module):
    def __init__(self, scan_mode='H'):
        super().__init__()

        self.up1 = PatchUnEmbed(in_chans=256, out_chans=64, patch_size=2)
        self.up2 = PatchUnEmbed(in_chans=64, out_chans=16, patch_size=4)
        self.down1 = PatchEmbed(in_chans=16, out_chans=64, patch_size=4)
        self.down2 = PatchEmbed(in_chans=64, out_chans=256, patch_size=2)
        self.mamba64down = Vim(dim=64, dt_rank=32,d_state=16, dropout=0.1, depth=2, scan_mode=scan_mode)  
        self.mamba256 = Vim(dim=256, dt_rank=32,d_state=64, dropout=0.1, depth=2, scan_mode=scan_mode)  
        self.mamba64up = Vim(dim=64, dt_rank=32,d_state=16, dropout=0.1, depth=2, scan_mode=scan_mode)  
        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)
        self.conv_input = ConvLayer(3, 16, kernel_size=3, stride=1)

    def forward(self,c):
        c = self.conv_input(c)
        c = self.down1(c)
        c = self.mamba64down(c)
        c = self.down2(c)
        prompt = self.mamba256(c)
        c = self.up1(prompt)
        c = self.mamba64up(c)
        c = self.up2(c)
        c = self.conv_output(c)

        return prompt,c

        


class HFIR(nn.Module):
    def __init__(self):
        super().__init__()
        self.mambaH = mamba_Branches(scan_mode='H')
        self.mambaV = mamba_Branches(scan_mode='V')
        self.mambaD = mamba_Branches(scan_mode='D')

    def forward(self, cH, cV, cD):
        
        # 分别对 cH, cV, cD 处理
        pH, outH = self.mambaH(cH)
        pV, outV = self.mambaV(cV)
        pD, outD = self.mambaD(cD)
        
        # 返回 cA，同时包括处理后的高频分量
        return (pH, pV, pD), (outH, outV, outD)

class HFIR_gf(nn.Module):
    def __init__(self, scan_mode='H'):
        super().__init__()

        self.up1 = PatchUnEmbed(in_chans=256, out_chans=64, patch_size=4)
        self.up2 = PatchUnEmbed(in_chans=64, out_chans=16, patch_size=4)
        self.down1 = PatchEmbed(in_chans=16, out_chans=64, patch_size=4)
        self.down2 = PatchEmbed(in_chans=64, out_chans=256, patch_size=4)
        self.mamba64down = Vim(dim=64, dt_rank=32,d_state=16, dropout=0.1, depth=2, scan_mode=scan_mode)  
        self.mamba256 = Vim(dim=256, dt_rank=32,d_state=64, dropout=0.1, depth=2, scan_mode=scan_mode)  
        self.mamba64up = Vim(dim=64, dt_rank=32,d_state=16, dropout=0.1, depth=2, scan_mode=scan_mode)  
        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)
        self.conv_input = ConvLayer(3, 16, kernel_size=3, stride=1)

    def forward(self,c):
        c = self.conv_input(c)
        c = self.down1(c)
        c = self.mamba64down(c)
        c = self.down2(c)
        prompt = self.mamba256(c)
        c = self.up1(prompt)
        c = self.mamba64up(c)
        c = self.up2(c)
        c = self.conv_output(c)

        return prompt,c
    
class LFIR(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
        res2net101 = Pre_Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
        res2net101.load_state_dict(torch.load('res2net101_v1b_26w_4s-0812c246.pth'))
        pretrained_dict = res2net101.state_dict()
        model_dict = self.encoder.state_dict()
        key_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(key_dict)
        self.encoder.load_state_dict(model_dict)
        self.ca1 = ChannelAttention(1024)
        self.ca2 = ChannelAttention(512)
        self.ca3 = ChannelAttention(256)
        self.ca4 = ChannelAttention(64)
        self.layer = BasicBlock(dim=256, num_blocks=10)
        self.dense_5 = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )
        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.dense_4 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.conv_4 = RDB(64, 4, 64)
        self.fusion_4 = Decoder_MDCBlock1(64, 2, mode='iter2')

        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)

        self.dense_3 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.conv_3 = RDB(32, 4, 32)
        self.fusion_3 = Decoder_MDCBlock1(32, 3, mode='iter2')

        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)

        self.dense_2 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )
        self.conv_2 = RDB(16, 4, 16)
        self.fusion_2 = Decoder_MDCBlock1(16, 4, mode='iter2')

        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)

        self.dense_1 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )
        self.conv_1 = RDB(8, 4, 8)
        self.fusion_1 = Decoder_MDCBlock1(8, 5, mode='iter2')

        self.conv_outputx = ConvLayer(16, 3, kernel_size=3, stride=1)


        self.CRA1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.CRA2 = nn.Conv2d(512,128, kernel_size=1)
        self.CRA3 = nn.Conv2d(256, 64, kernel_size=1)
        self.CRA4 = nn.Conv2d(64, 32, kernel_size=1)

    def forward(self, x, pH,pV,pD):
        ini = x
        res16x, res8x, res4x, res2x = self.encoder(x)
        res16x_w = self.ca1(res16x)
        res16x = self.CRA1(res16x*res16x_w)
        res8x_w = self.ca2(res8x)
        res8x = self.CRA2(res8x*res8x_w)
        res4x_w = self.ca3(res4x)
        res4x = self.CRA3(res4x*res4x_w)
        res2x_w = self.ca4(res2x)
        res2x = self.CRA4(res2x*res2x_w)

        res16x = self.layer(res16x, pH, pV, pD) #dehaze

        res_dehaze = res16x
        in_ft = res16x * 2
        res16x = self.dense_5(in_ft) + in_ft - res_dehaze
        res16x_1, res16x_2 = res16x.split([(res16x.size()[1] // 2), (res16x.size()[1] // 2)], dim=1)
        feature_mem_up = [res16x_1]

        res16x = self.convd16x(res16x)

        res16x = F.interpolate(res16x, res8x.size()[2:], mode='bilinear')
        res8x = torch.add(res16x, res8x)
        res8x = self.dense_4(res8x) + res8x - res16x
        res8x_1, res8x_2 = res8x.split([(res8x.size()[1] // 2), (res8x.size()[1] // 2)], dim=1)
        res8x_1 = self.fusion_4(res8x_1, feature_mem_up)
        res8x_2 = self.conv_4(res8x_2)
        feature_mem_up.append(res8x_1)
        res8x = torch.cat((res8x_1, res8x_2), dim=1)

        res8x = self.convd8x(res8x)

        res8x = F.interpolate(res8x, res4x.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, res4x)
        res4x = self.dense_3(res4x) + res4x - res8x
        res4x_1, res4x_2 = res4x.split([(res4x.size()[1] // 2), (res4x.size()[1] // 2)], dim=1)
        res4x_1 = self.fusion_3(res4x_1, feature_mem_up)
        res4x_2 = self.conv_3(res4x_2)
        feature_mem_up.append(res4x_1)
        res4x = torch.cat((res4x_1, res4x_2), dim=1)

        res4x = self.convd4x(res4x)

        res4x = F.interpolate(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)

        res2x = self.dense_2(res2x) + res2x - res4x
        res2x_1, res2x_2 = res2x.split([(res2x.size()[1] // 2), (res2x.size()[1] // 2)], dim=1)
        res2x_1 = self.fusion_2(res2x_1, feature_mem_up)
        res2x_2 = self.conv_2(res2x_2)

        feature_mem_up.append(res2x_1)
        res2x = torch.cat((res2x_1, res2x_2), dim=1)
        res2x = self.convd2x(res2x)

        res2x = F.interpolate(res2x, ini.size()[2:], mode='bilinear')
        x = torch.add(res2x, res2x)
        x = self.dense_1(x) + x - res2x
        x_1, x_2 = x.split([(x.size()[1] // 2), (x.size()[1] // 2)], dim=1)
        x_1 = self.fusion_1(x_1, feature_mem_up)

        x_2 = self.conv_1(x_2)
        x = torch.cat((x_1, x_2), dim=1)

        outx = self.conv_outputx(x)

        return outx
from methods.HaarWavelet import HaarWavelet
class FIR_wavelet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hfi = HFIR()
        self.lfi = LFIR()
    def min_max_normalize(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-6)  # 防止除零错误
        return normalized_tensor

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        base_number = 32 #
        mod_pad_h = (base_number - h % base_number) % base_number
        mod_pad_w = (base_number - w % base_number) % base_number
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


    def forward_features(self, x):
        # 获取小波分解系数
        cA, cH, cV, cD,original_size = HaarWavelet.wavelet_transform_2d(x)
        cA = self.min_max_normalize(cA)
        cH = self.min_max_normalize(cH)
        cV = self.min_max_normalize(cV)
        cD = self.min_max_normalize(cD)
        (pH, pV, pD), (outH, outV, outD) = self.hfi(cH, cV, cD)
        
        # 使用低频信息和增强的高频信息
        outA = self.lfi(cA, pH, pV, pD)
        outx = HaarWavelet.inverse_wavelet_transform_2d(outA,outH,outV,outD,original_size)
        coeffs = (outA,outH,outV,outD)
        return outx,coeffs


    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        x,coeffs = self.forward_features(x)
        x = x[:, :, :H, :W]
        return x,coeffs
class FIR(nn.Module):
    def __init__(self):
        super().__init__()
        self.hfi = HFIR()
        self.lfi = LFIR()
    def min_max_normalize(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-6)  # 防止除零错误
        return normalized_tensor

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        base_number = 32 #
        mod_pad_h = (base_number - h % base_number) % base_number
        mod_pad_w = (base_number - w % base_number) % base_number
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


    def forward_features(self, x):
        # 获取小波分解系数
        cA, cH, cV, cD,original_size = HaarWavelet.wavelet_transform_2d(x)
        cH = self.min_max_normalize(cH)
        cV = self.min_max_normalize(cV)
        cD = self.min_max_normalize(cD)
        (pH, pV, pD), (outH, outV, outD) = self.hfi(cH, cV, cD)
        
        # 使用低频信息和增强的高频信息
        outx = self.lfi(x, pH, pV, pD)
        coeffs = (outH,outV,outD)
        return outx,coeffs


    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        x,coeffs = self.forward_features(x)
        x = x[:, :, :H, :W]
        return x,coeffs
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

class FIR_gf(nn.Module):
    def __init__(self):
        super().__init__()
        self.gf= GaussianFilter()
        self.igf = InverseGaussianFilter()
        self.hfi = HFIR_gf()
        self.lfi = LFIR()


    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        base_number = 32 #
        mod_pad_h = (base_number - h % base_number) % base_number
        mod_pad_w = (base_number - w % base_number) % base_number
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


    def forward_features(self, x):
        # 获取小波分解系数
        LFI,HFI = self.gf(x)
        prompt,outHFI = self.hfi(HFI)
        
        # 使用低频信息和增强的高频信息
        outLFI = self.lfi(LFI, prompt, prompt, prompt)
        outx = self.igf(LFI=outLFI,HFI=outHFI)
        coeffs = (outLFI,outHFI)
        return outx,coeffs


    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        x,coeffs = self.forward_features(x)
        x = x[:, :, :H, :W]
        return x,coeffs
    
class FIR_abalation(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
        res2net101 = Pre_Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
        res2net101.load_state_dict(torch.load('res2net101_v1b_26w_4s-0812c246.pth'))
        pretrained_dict = res2net101.state_dict()
        model_dict = self.encoder.state_dict()
        key_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(key_dict)
        self.encoder.load_state_dict(model_dict)
        self.ca1 = ChannelAttention(1024)
        self.ca2 = ChannelAttention(512)
        self.ca3 = ChannelAttention(256)
        self.ca4 = ChannelAttention(64)
        self.dense_5 = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )
        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.dense_4 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.conv_4 = RDB(64, 4, 64)
        self.fusion_4 = Decoder_MDCBlock1(64, 2, mode='iter2')

        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)

        self.dense_3 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.conv_3 = RDB(32, 4, 32)
        self.fusion_3 = Decoder_MDCBlock1(32, 3, mode='iter2')

        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)

        self.dense_2 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )
        self.conv_2 = RDB(16, 4, 16)
        self.fusion_2 = Decoder_MDCBlock1(16, 4, mode='iter2')

        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)

        self.dense_1 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )
        self.conv_1 = RDB(8, 4, 8)
        self.fusion_1 = Decoder_MDCBlock1(8, 5, mode='iter2')

        self.conv_outputx = ConvLayer(16, 3, kernel_size=3, stride=1)


        self.CRA1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.CRA2 = nn.Conv2d(512,128, kernel_size=1)
        self.CRA3 = nn.Conv2d(256, 64, kernel_size=1)
        self.CRA4 = nn.Conv2d(64, 32, kernel_size=1)

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        base_number = 32 #
        mod_pad_h = (base_number - h % base_number) % base_number
        mod_pad_w = (base_number - w % base_number) % base_number
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        ini = x
        res16x, res8x, res4x, res2x = self.encoder(x)
        res16x_w = self.ca1(res16x)
        res16x = self.CRA1(res16x*res16x_w)
        res8x_w = self.ca2(res8x)
        res8x = self.CRA2(res8x*res8x_w)
        res4x_w = self.ca3(res4x)
        res4x = self.CRA3(res4x*res4x_w)
        res2x_w = self.ca4(res2x)
        res2x = self.CRA4(res2x*res2x_w)

        res_dehaze = res16x
        in_ft = res16x * 2
        res16x = self.dense_5(in_ft) + in_ft - res_dehaze
        res16x_1, res16x_2 = res16x.split([(res16x.size()[1] // 2), (res16x.size()[1] // 2)], dim=1)
        feature_mem_up = [res16x_1]

        res16x = self.convd16x(res16x)

        res16x = F.interpolate(res16x, res8x.size()[2:], mode='bilinear')
        res8x = torch.add(res16x, res8x)
        res8x = self.dense_4(res8x) + res8x - res16x
        res8x_1, res8x_2 = res8x.split([(res8x.size()[1] // 2), (res8x.size()[1] // 2)], dim=1)
        res8x_1 = self.fusion_4(res8x_1, feature_mem_up)
        res8x_2 = self.conv_4(res8x_2)
        feature_mem_up.append(res8x_1)
        res8x = torch.cat((res8x_1, res8x_2), dim=1)

        res8x = self.convd8x(res8x)

        res8x = F.interpolate(res8x, res4x.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, res4x)
        res4x = self.dense_3(res4x) + res4x - res8x
        res4x_1, res4x_2 = res4x.split([(res4x.size()[1] // 2), (res4x.size()[1] // 2)], dim=1)
        res4x_1 = self.fusion_3(res4x_1, feature_mem_up)
        res4x_2 = self.conv_3(res4x_2)
        feature_mem_up.append(res4x_1)
        res4x = torch.cat((res4x_1, res4x_2), dim=1)

        res4x = self.convd4x(res4x)

        res4x = F.interpolate(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)

        res2x = self.dense_2(res2x) + res2x - res4x
        res2x_1, res2x_2 = res2x.split([(res2x.size()[1] // 2), (res2x.size()[1] // 2)], dim=1)
        res2x_1 = self.fusion_2(res2x_1, feature_mem_up)
        res2x_2 = self.conv_2(res2x_2)

        feature_mem_up.append(res2x_1)
        res2x = torch.cat((res2x_1, res2x_2), dim=1)
        res2x = self.convd2x(res2x)

        res2x = F.interpolate(res2x, ini.size()[2:], mode='bilinear')
        x = torch.add(res2x, res2x)
        x = self.dense_1(x) + x - res2x
        x_1, x_2 = x.split([(x.size()[1] // 2), (x.size()[1] // 2)], dim=1)
        x_1 = self.fusion_1(x_1, feature_mem_up)

        x_2 = self.conv_1(x_2)
        x = torch.cat((x_1, x_2), dim=1)

        outx = self.conv_outputx(x)

        return outx
    def forward(self,x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        x = self.forward_features(x)
        x = x[:, :, :H, :W]
        return x,x
def build_net():
    return FIR()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FIR().to(device)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    outx,_ = net(dummy_input)
    print(outx.shape)
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: ==> {}M".format(pytorch_total_params/1e6))
