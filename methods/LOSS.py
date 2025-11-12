import torch.nn as nn
import torch
from torchvision import models

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1) 
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4) 
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class GCP_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = Vgg19().cuda()
        self.sigma = 0.5
        self.distance = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
    def gaussian_distance(self, x, y):
        # 高斯距离
        return 1- torch.exp(-self.distance(x,y)**2 / (2 * self.sigma ** 2))
    def fft(self, x):
        # 计算 FFT，并将结果转换为复数的实部和虚部堆叠
        fft_result = torch.fft.fft2(x, dim=(-2, -1), norm='backward')
        fft_result = torch.fft.fftshift(fft_result)
        fft_result = torch.stack((fft_result.real, fft_result.imag), -1)
        return fft_result
    def forward(self, p, g, n):
        p_vgg, g_vgg, n_vgg= self.vgg(p), self.vgg(g.detach()),self.vgg(n),
        loss1 = 0
        loss2 = 0
        for i in range(len(p_vgg)):
            # p_freq = self.fft(p)
            # g_freq = self.fft(g)
            # d_pg_freq = self.distance(p_freq, g_freq) /20
            # loss1 += self.weights[i] * d_pg_freq
            # d_pn_freq = self.gaussian_distance(p_freq, n_freq) 
            # d_gn_freq = self.gaussian_distance(g_freq, n_freq) 
            # out_freq = d_pg_freq / (d_gn_freq + d_pn_freq + d_pg_freq + 1e-6)
            
            d_pn = self.gaussian_distance(p_vgg[i], n_vgg[i])
            d_gn = self.gaussian_distance(g_vgg[i], n_vgg[i])
            d_pg = self.gaussian_distance(p_vgg[i], g_vgg[i])
            out = d_pg / (d_gn + d_pn + d_pg+1e-6)
            loss2 += self.weights[i] * out
        return loss2

def build_gcp_loss():
    gcp_loss = GCP_Loss()
    return gcp_loss


class FFT_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.distance = nn.L1Loss()
    def fft(self, x):
        # 计算 FFT，并将结果转换为复数的实部和虚部堆叠
        fft_result = torch.fft.fft2(x, dim=(-2, -1), norm='backward')
        fft_result = torch.fft.fftshift(fft_result)
        fft_result = torch.stack((fft_result.real, fft_result.imag), -1)
        return fft_result
    def forward(self, label, pred):
        # 计算清晰图像的 FFT
        label_fft = self.fft(label)

        # 计算生成图像的 FFT
        pred_fft = self.fft(pred)

        # 计算 L1 损失
        loss = self.distance(pred_fft, label_fft)/20
        return loss

def build_fft_loss():
    fft_loss = FFT_Loss()
    return fft_loss

import pywt
from methods.HaarWavelet import HaarWavelet
class Coeffs_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.distance = nn.L1Loss()
    def min_max_normalize(self,tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-6)  # 防止除零错误
        return normalized_tensor
    def forward(self, coeffs, clear):
        _,H_clear,V_clear,D_clear,_ = HaarWavelet.wavelet_transform_2d(clear)
        # 从输入 coeffs 中获取 H、V、D
        H_haze, V_haze, D_haze = coeffs
        # 计算 L1 损失（空间域）
        l1_loss_spatial = (
            self.distance(H_haze, H_clear) +
            self.distance(V_haze, V_clear) +
            self.distance(D_haze, D_clear) 
        ) 

        return l1_loss_spatial

    
def build_coeffs_loss():
    coeffs_loss = Coeffs_Loss()
    return coeffs_loss


import torch
import torch.nn as nn
import torch.nn.functional as F
# import kornia.color as kcolor
from methods.rgb2hsv import RGB_HSV
class HSVContrastLoss(nn.Module):
    def __init__(self):
        """
        HSV 颜色对比损失，拉近去雾图像 (p) 和清晰标签 (g)，拉远去雾图像 (p) 和雾霾输入 (n)。
        """
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.change = RGB_HSV()
    def forward(self, p, g, n):
        """
        计算 HSV 颜色对比损失。
        :param p: 网络输出图像 (B, 3, H, W)
        :param g: 清晰标签图像 (B, 3, H, W)
        :param n: 输入雾霾图像 (B, 3, H, W)
        :return: 颜色对比损失值
        """
        # 转换到 HSV 颜色空间
        p = torch.clamp(p,min=0,max=1)
        g = torch.clamp(g,min=0,max=1)
        n = torch.clamp(n,min=0,max=1)
        p_hsv, g_hsv, n_hsv = self.change.rgb_to_hsv(p), self.change.rgb_to_hsv(g), self.change.rgb_to_hsv(n)
        # 提取亮度 (V) 和饱和度 (S) 通道
        p_v, p_s = p_hsv[:, 2, :, :], p_hsv[:, 1, :, :]
        g_v, g_s = g_hsv[:, 2, :, :], g_hsv[:, 1, :, :]
        n_v, n_s = n_hsv[:, 2, :, :], n_hsv[:, 1, :, :]

        # 计算 V 通道和 S 通道的 L1 损失
        d_pn_v = self.l1_loss(p_v, n_v)  # p 和 n 的亮度距离
        d_gn_v = self.l1_loss(g_v, n_v)  # g 和 n 的亮度距离
        d_pg_v = self.l1_loss(p_v, g_v)  # p 和 g 的亮度距离

        d_pn_s = self.l1_loss(p_s, n_s)  # p 和 n 的饱和度距离
        d_gn_s = self.l1_loss(g_s, n_s)  # g 和 n 的饱和度距离
        d_pg_s = self.l1_loss(p_s, g_s)  # p 和 g 的饱和度距离

        # 分子分母形式的对比损失
        loss_v = d_pg_v / (d_gn_v + d_pn_v + 1e-6)
        loss_s = d_pg_s / (d_gn_s + d_pn_s + 1e-6)

        return loss_v + loss_s

# 构建 HSV 对比损失函数
def build_hsv_contrast_loss():
    return HSVContrastLoss()

import torch
import torch.nn as nn
import kornia.color as kcolor


class YCbCrContrastLoss(nn.Module):
    def __init__(self):
        """
        YCbCr 色彩对比损失：
        仅对 Cb、Cr 通道计算输出(p)、清晰(g)、雾图(n) 的对比约束。
        """
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, p, g, n):
        """
        :param p: 网络输出图像 (B, 3, H, W)
        :param g: 清晰标签图像 (B, 3, H, W)
        :param n: 输入雾霾图像 (B, 3, H, W)
        :return: YCbCr 对比损失值
        """
        # 保证范围稳定
        p = torch.clamp(p, 0, 1)
        g = torch.clamp(g, 0, 1)
        n = torch.clamp(n, 0, 1)

        # RGB → YCbCr
        p_ycbcr = kcolor.rgb_to_ycbcr(p)
        g_ycbcr = kcolor.rgb_to_ycbcr(g)
        n_ycbcr = kcolor.rgb_to_ycbcr(n)

        # 提取 Cb、Cr 通道
        p_cb, p_cr = p_ycbcr[:, 1], p_ycbcr[:, 2]
        g_cb, g_cr = g_ycbcr[:, 1], g_ycbcr[:, 2]
        n_cb, n_cr = n_ycbcr[:, 1], n_ycbcr[:, 2]

        # 分通道计算对比损失
        d_pg_cb = self.l1_loss(p_cb, g_cb)
        d_pn_cb = self.l1_loss(p_cb, n_cb)
        d_gn_cb = self.l1_loss(g_cb, n_cb)

        d_pg_cr = self.l1_loss(p_cr, g_cr)
        d_pn_cr = self.l1_loss(p_cr, n_cr)
        d_gn_cr = self.l1_loss(g_cr, n_cr)

        # 通道独立比例损失
        loss_cb = d_pg_cb / (d_gn_cb + d_pn_cb + 1e-6)
        loss_cr = d_pg_cr / (d_gn_cr + d_pn_cr + 1e-6)

        return loss_cb + loss_cr


def build_ycbcr_contrast_loss():
    return YCbCrContrastLoss()
