import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class FRFSIMLoss(nn.Module):
    def __init__(self, device, radius=1.5, ps=3):
        super(FRFSIMLoss, self).__init__()
        self.radius = radius
        self.ps = ps
        # Initialize Gaussian filter for MSCN
        self.mscn_filter = self.get_gaussian_filter(kernel_size=7, sigma=7/6).to(device)
        # Initialize Sobel filters for gradient computation
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        sobel_y = sobel_x.transpose(2,3).to(device)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, D, R):
        """
        计算 FRFSIM 损失。
        D: 去雾图像，形状为 (B, 3, H, W)
        R: 参考图像，形状为 (B, 3, H, W)
        返回: 损失值
        """
        B = D.size(0)  # 获取批次大小
        loss = 0.0

        # 对每个样本进行处理
        for i in range(B):
            D_single = D[i:i+1, :, :, :]  # 获取第 i 个图像
            R_single = R[i:i+1, :, :, :]  # 获取第 i 个参考图像

            # Convert to grayscale
            Dg = self.rgb2gray(D_single)
            Rg = self.rgb2gray(R_single)
            C = self.calculate_C(Dg)

            # Extract RGB channels
            D_R, D_G, D_B = D_single[:, 0:1, :, :], D_single[:, 1:2, :, :], D_single[:, 2:3, :, :]
            R_R, R_G, R_B = R_single[:, 0:1, :, :], R_single[:, 1:2, :, :], R_single[:, 2:3, :, :]

            # Convert RGB to HSV
            D_s, D_v = self.rgb_to_hsv(D_single)
            R_s, R_v = self.rgb_to_hsv(R_single)

            # MSCN computation
            D_MSCN, _ = self.compute_mscn(Dg)
            R_MSCN, _ = self.compute_mscn(Rg)

            # MSCN similarity
            mc_mscn = torch.max(torch.var(D_MSCN, dim=[2,3], unbiased=False, keepdim=True),
                                torch.var(R_MSCN, dim=[2,3], unbiased=False, keepdim=True))
            w_mscn = mc_mscn / mc_mscn.view(mc_mscn.size(0), -1).sum(dim=1, keepdim=True).view(-1,1,1,1)
            CM = C[0].to(D.device)

            SM = (2 * D_MSCN * R_MSCN + CM) / (D_MSCN **2 + R_MSCN **2 + CM)
            mean_SM = SM.mean(dim=[1,2,3])  # Shape: (1,)

            # Dark channel similarity
            Ddc = torch.min(torch.min(D_R / 255.0, D_G / 255.0), D_B / 255.0)
            Rdc = torch.min(torch.min(R_R / 255.0, R_G / 255.0), R_B / 255.0)
            CD = C[1].to(D.device)
            SD = (2 * Ddc * Rdc + CD) / (Ddc **2 + Rdc **2 + CD + 1e-8)
            mean_SD = SD.mean(dim=[1,2,3])  # Shape: (1,)

            # Color similarity
            D_color = D_s * D_v
            R_color = R_s * R_v
            CC = C[2].to(D.device)
            SC = (2 * D_color * R_color + CC) / (D_color **2 + R_color **2 + CC+ 1e-8)
            mean_SC = SC.mean(dim=[1,2,3])  # Shape: (1,)

            # Gradient similarity
            D_gradient = self.compute_grad(Dg)
            R_gradient = self.compute_grad(Rg)
            CG = C[3].to(D.device)
            SG = (2 * D_gradient * R_gradient + CG) / (D_gradient **2 + R_gradient **2 + CG+ 1e-8)
            mean_SG = SG.mean(dim=[1,2,3])  # Shape: (1,)

            # Pooling
            # Define thresholds
            threshold1 = 0.85
            threshold2 = 0.0
            # Compute b1 and b2 using differentiable operations
            b1 = torch.where((mean_SD > threshold1) & (mean_SD <=1),
                             torch.full_like(mean_SD, 0.2),
                             torch.where((mean_SD >=0) & (mean_SD <= threshold1),
                                         torch.full_like(mean_SD, 0.8),
                                         torch.full_like(mean_SD, 0.5)))
            b2 = torch.where((mean_SD > threshold1) & (mean_SD <=1),
                             torch.full_like(mean_SD, 0.8),
                             torch.where((mean_SD >=0) & (mean_SD <= threshold1),
                                         torch.full_like(mean_SD, 0.2),
                                         torch.full_like(mean_SD, 0.5)))

            # Expand b1 and b2 to match frfsimmap shape
            b1 = b1.view(-1, 1, 1, 1)
            b2 = b2.view(-1, 1, 1, 1)
            # Compute frfsimmap
            frfsimmap = self.safe_pow(base=SM * SD, exponent=b1) * (SG * SC).pow(b2)
            frfsimval = frfsimmap.abs().mean(dim=[1,2,3])  # Shape: (1,)

            # Accumulate the loss for each sample
            loss += frfsimval

        return loss / B  # 返回平均损失

    def calculate_C(self, D):
        # 获取图像的最小值和最大值
        min_vals, _ = D.min(dim=2, keepdim=True)  # 计算 H 维度上的最小值
        min_vals, _ = min_vals.min(dim=3, keepdim=True)  # 计算 W 维度上的最小值
        
        max_vals, _ = D.max(dim=2, keepdim=True)  # 计算 H 维度上的最大值
        max_vals, _ = max_vals.max(dim=3, keepdim=True)  # 计算 W 维度上的最大值
    
        # 计算动态范围 d
        d = max_vals - min_vals  # 计算动态范围
        
        # 计算 C 的四个值
        C = torch.stack([
            (0.01 * d) ** 2,  # 第一个数值：0.01*d 的平方
            (0.01 * d) ** 2 / 2,  # 第二个数值：0.01*d 的平方除以 2
            (0.03 * d) ** 2 / 2,  # 第三个数值：0.03*d 的平方除以 2
            (0.03 * d) ** 2  # 第四个数值：0.03*d 的平方
        ], dim=1)  # 堆叠成一个形状为 (1, 4, 1, 1) 的张量
        
        # 将形状变为 (4,)
        C = C.squeeze()  # 去掉多余的维度，最终形状为 (4,)
        
        return C

    def rgb2gray(self, rgb):
        """Convert RGB to Grayscale using standard weights."""
        weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
        gray = torch.sum(rgb * weights, dim=1, keepdim=True)
        return gray

    def rgb_to_hsv(self, rgb):
        """
        Convert RGB to HSV.
        rgb: Tensor of shape (B, 3, H, W) with values in [0,1]
        Returns h, s, v each of shape (B, 1, H, W)
        """
        maxc, _ = rgb.max(dim=1, keepdim=True)
        minc, _ = rgb.min(dim=1, keepdim=True)
        v = maxc
        delta = maxc - minc + 1e-8  # Avoid division by zero
        s = delta / (maxc + 1e-8)
        
        return s, v

    def get_gaussian_filter(self, kernel_size=7, sigma=7/6):
        """Create a Gaussian filter."""
        # Create a 2D Gaussian kernel
        x = torch.arange(kernel_size) - kernel_size // 2
        x = x.float()
        gauss_1d = torch.exp(-0.5 * (x / sigma)**2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = torch.outer(gauss_1d, gauss_1d)
        gauss_2d = gauss_2d.view(1, 1, kernel_size, kernel_size)
        return gauss_2d

    def compute_mscn(self, img_gray):
        """Compute MSCN coefficients."""
        mu = F.conv2d(img_gray, self.mscn_filter, padding=3, groups=1)
        mu_sq = mu * mu
        sigma = torch.sqrt(abs(F.conv2d(img_gray * img_gray, self.mscn_filter, padding=3, groups=1) - mu_sq + 1e-8))
        mscn = (img_gray - mu) / (sigma + 1e-8)
        return mscn, mu

    def compute_grad(self, img_gray):
        """Compute gradient magnitude using Sobel filters."""
        grad_x = F.conv2d(img_gray, self.sobel_x, padding=1, groups=1)
        grad_y = F.conv2d(img_gray, self.sobel_y, padding=1, groups=1)
        grad = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return grad

    def safe_pow(self, base, exponent):
        """
        处理负底数和小数指数的情况，模拟 MATLAB 的处理方式。
        对于负底数和小数指数，转换为复数形式进行计算。

        base: 底数，Tensor
        exponent: 指数，Tensor
        返回: 计算结果
        """
        # 如果底数为负数，并且指数为小数，则转换为复数形式
        is_negative = base < 0
        base_abs = torch.abs(base)
        
        # 对于负数底数，转换为复数：负数底数可以表示为复数的模和相位
        complex_base = torch.complex(base_abs, torch.zeros_like(base))  # 负数底数转换为复数
        
        # 计算复数的指数
        complex_result = torch.pow(complex_base, exponent)

        # 对于负底数并且指数为小数，按照 MATLAB 的做法提取实部
        result = complex_result.real

        return result


from PIL import Image
import torch
from torchvision import transforms

def load_image(image_path: str) -> (torch.Tensor, tuple):
    """
    加载图像并转换为张量。
    :param image_path: 图像路径。
    :return: 形状为 [1, 3, H, W] 的归一化张量和原始尺寸 (H, W)。
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
    return image_tensor, original_size

def center_crop(image_tensor: torch.Tensor, target_size: tuple) -> torch.Tensor:
    """
    根据目标尺寸中心裁剪图像张量。
    :param image_tensor: 输入图像张量，形状为 [1, 3, H, W]。
    :param target_size: 裁剪后的目标尺寸 (H, W)。
    :return: 裁剪后的图像张量。
    """
    _, _, H, W = image_tensor.shape
    target_h, target_w = target_size
    top = (H - target_h) // 2
    left = (W - target_w) // 2
    return image_tensor[:, :, top:top + target_h, left:left + target_w]

if __name__ == "__main__":
    # 替换为您的输入图像路径
    img_defog_path = "img/AM_Google_754_haze.png"  # 去雾图像路径
    img_ref_path = "img/AM_Google_754_pred4.png"  # 参考图像路径

    # 加载图像并获取原始尺寸
    img_defog, defog_size = load_image(img_defog_path)
    img_ref, ref_size = load_image(img_ref_path)

    # 计算最小尺寸
    min_height = min(defog_size[1], ref_size[1])  # 最小高度
    min_width = min(defog_size[0], ref_size[0])  # 最小宽度
    target_size = (min_height, min_width)

    # 对两张图像进行中心裁剪
    img_defog = center_crop(img_defog, target_size)
    img_ref = center_crop(img_ref, target_size)

    # 确保设备一致
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_defog = img_defog.to(device)
    img_ref = img_ref.to(device)

    # 使用 FRFSIMLoss
    loss_fn = FRFSIMLoss(device=device)
    loss = loss_fn(img_defog, img_ref)
    print("FRFSIM-based Loss:", loss.item())
