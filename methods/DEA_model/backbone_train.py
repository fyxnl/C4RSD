import torch.nn as nn
import torch.nn.functional as F
import torch
from methods.DEA_model.modules import DEABlockTrain, DEBlockTrain, CGAFusion
# from modules import DEABlockTrain, DEBlockTrain, CGAFusion

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
class Encoder(nn.Module):
    def __init__(self, base_dim=32):
        super(Encoder, self).__init__()
        # down-sample
        self.down1 = nn.Sequential(nn.Conv2d(3, base_dim, kernel_size=3, stride = 1, padding=1))
        self.down2 = nn.Sequential(nn.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down4 = nn.Sequential(nn.Conv2d(base_dim*4, base_dim*8, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down5 = nn.Sequential(nn.Conv2d(base_dim*8, base_dim*16, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        # level1
        self.down_level1_block1 = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block2 = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block3 = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block4 = DEBlockTrain(default_conv, base_dim, 3)

        # level2
        self.fe_level_2 = nn.Conv2d(in_channels=base_dim * 2, out_channels=base_dim * 2, kernel_size=3, stride=1, padding=1)
        self.down_level2_block1 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.down_level2_block2 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.down_level2_block3 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.down_level2_block4 = DEBlockTrain(default_conv, base_dim * 2, 3)

        # level3
        self.fe_level_3 = nn.Conv2d(in_channels=base_dim * 4, out_channels=base_dim * 4, kernel_size=3, stride=1, padding=1)
        self.level3_block1 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block2 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block3 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block4 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block5 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block6 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block7 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block8 = DEABlockTrain(default_conv, base_dim * 4, 3)

    def forward(self, x):
        x_down1 = self.down1(x)
        x_down1 = self.down_level1_block1(x_down1)
        x_down1 = self.down_level1_block2(x_down1)
        x_down1 = self.down_level1_block3(x_down1)
        x_down1 = self.down_level1_block4(x_down1)

        x_down2 = self.down2(x_down1)
        x_down2_init = self.fe_level_2(x_down2)
        x_down2_init = self.down_level2_block1(x_down2_init)
        x_down2_init = self.down_level2_block2(x_down2_init)
        x_down2_init = self.down_level2_block3(x_down2_init)
        x_down2_init = self.down_level2_block4(x_down2_init)

        x_down3 = self.down3(x_down2_init)
        x_down3_init = self.fe_level_3(x_down3)
        x1 = self.level3_block1(x_down3_init)
        x2 = self.level3_block2(x1)
        x3 = self.level3_block3(x2)
        x4 = self.level3_block4(x3)
        x5 = self.level3_block5(x4)
        x6 = self.level3_block6(x5)
        x7 = self.level3_block7(x6)
        x8 = self.level3_block8(x7)
        x_down4 = self.down4(x8)
        x_down5 = self.down5(x_down4)
        return x_down5,x_down2,x_down3
class Decoder(nn.Module):
    def __init__(self, base_dim=32):
        super(Decoder, self).__init__()
        self.up_level1_block1 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block2 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block3 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block4 = DEBlockTrain(default_conv, base_dim, 3)
        # level2
        self.up_level2_block1 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block2 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block3 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block4 = DEBlockTrain(default_conv, base_dim * 2, 3)
        # level3
        # up-sample
        self.up4 = nn.Sequential(nn.ConvTranspose2d(base_dim*16, base_dim*8, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up5 = nn.Sequential(nn.ConvTranspose2d(base_dim*8, base_dim*4, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Conv2d(base_dim, 3, kernel_size=3, stride=1, padding=1))
        # feature fusion
        self. mix1 = CGAFusion(base_dim * 4, reduction=8)
        self.mix2 = CGAFusion(base_dim * 2, reduction=4)

    def forward(self, x,x_down2,x_down3):
        x = self.up4(x)
        x = self.up5(x)
        x_level3_mix = self.mix1(x_down3, x)
        
        x_up1 = self.up1(x_level3_mix)
        x_up1 = self.up_level2_block1(x_up1)
        x_up1 = self.up_level2_block2(x_up1)
        x_up1 = self.up_level2_block3(x_up1)
        x_up1 = self.up_level2_block4(x_up1)
        x_level2_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_level2_mix)
        x_up2 = self.up_level1_block1(x_up2)
        x_up2 = self.up_level1_block2(x_up2)
        x_up2 = self.up_level1_block3(x_up2)
        x_up2 = self.up_level1_block4(x_up2)
        out = self.up3(x_up2)

        return out
class DEANet(nn.Module):
    def __init__(self, base_dim=32):
        super(DEANet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def sda(self, zs, zt, n_components):
        """
        SDA操作：在计算图内进行源特征和目标特征的对齐
        """
        # 创建 zs 和 zt 的副本，这样就不会直接影响计算图中的梯度
        features_s = zs.data.clone()  # 复制数据
        B, C, H, W = features_s.size()
        features_s = features_s.view(B * C, H * W)
        features_t = zt.data.clone()  # 复制数据
        features_t = features_t.view(B * C, H * W)

        features_s_centered = features_s - torch.mean(features_s, dim=0)
        features_t_centered = features_t - torch.mean(features_t, dim=0)
        # print(features_s_centered.shape)
        _, S_s, Vh_s = torch.linalg.svd(features_s_centered,full_matrices=False)
        _, S_t, Vh_t = torch.linalg.svd(features_t_centered,full_matrices=False)
        # _, S_s, Vh_s = torch.svd(features_s_centered)
        # _, S_t, Vh_t = torch.svd(features_t_centered)
        V_s_reduced = Vh_s.T[:, :n_components]
        # print(V_s_reduced.shape)
        V_t_reduced = Vh_t.T[:, :n_components]
        # F_s = torch.mm(features_s,V_s_reduced)
        # F_t = torch.mm(features_t,V_t_reduced)
        T_ts = torch.mm(V_s_reduced.T, V_t_reduced)

        W_s = torch.sqrt(S_s[:n_components])
        W_t = torch.sqrt(S_t[:n_components])
        W_s_inv = 1.0 / (W_s + 1e-8)
        A_ts = torch.diag(W_s_inv) @ torch.diag(W_t)
        
        M_s = torch.mm(torch.mm(torch.mm(V_s_reduced, T_ts), A_ts), V_t_reduced.T)

        features_s_transformed = torch.mm(features_s,M_s)
        features_s_transformed = features_s_transformed.view(B, C, H, W)

        # 将计算结果赋回原张量的 data，这样它就可以影响计算图了
        zs.data.copy_(features_s_transformed)

        return zs  # 返回更新后的原张量 zs
    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        base_number = 32
        mod_pad_h = (base_number - h % base_number) % base_number
        mod_pad_w = (base_number - w % base_number) % base_number
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    def forward(self, syn):
        B1,C1,H1, W1 = syn.shape
        # B2,C2,H2,W2 = real.shape
        syn = self.check_image_size(syn)
        # real = self.check_image_size(real)
        x_syn,down2_syn,down3_syn=self.encoder(syn)
        syn_out = self.decoder(x_syn,down2_syn,down3_syn)
        return syn_out[:, :, :H1, :W1]
        # else:
        #     x_syn,down2_syn,down3_syn=self.encoder(syn)
        #     # print(x_syn.shape)
        #     x_real,down2_real,down3_real=self.encoder(real)
        #     x_syn= self.sda(x_syn,x_real, n_components=8*8)
        #     # print(x_syn.shape)
        #     syn_out = self.decoder(x_syn,down2_syn,down3_syn)
        #     real_out = self.decoder(x_real,down2_real,down3_real)
        #     return syn_out[:, :, :H1, :W1],real_out[:, :, :H2, :W2]
def build_net():
    return DEANet()
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DEANet().to(device)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    output_tensor,_ = net(dummy_input,dummy_input,True)
    print(output_tensor[0].shape)
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: ==> {}M".format(pytorch_total_params/1e6))