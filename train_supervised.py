# -*- coding: utf-8 -*-
import os
import json
import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from metric.cal_metric import cal_batch_psnr,cal_batch_ssim
from methods import LOSS, MyModel, msssim,Teacher, DEA_model
from methods.vgg.CR import ContrastLoss
from options.train_options import TrainOptions
from utils.datautils import Dataset_train
from utils.seed import setup_seed
# import wandb  # 导入 wandb
# import pyiqa
# 设置随机种子
setup_seed(42)

def init_distributed_mode():
    """
    初始化分布式环境。
    """
    dist.init_process_group(backend="nccl")  # 使用 NCCL 后端（适合 GPU）
    rank = dist.get_rank()  # 全局 rank
    local_rank = int(os.environ["LOCAL_RANK"])  # 本地 rank
    torch.cuda.set_device(local_rank)  # 设置当前 GPU
    return rank, local_rank

def cleanup_distributed():
    """
    清理分布式环境。
    """
    dist.destroy_process_group()


def train(rank, local_rank, world_size):
    """
    训练函数。
    """
    # 数据加载器
    # syn_path='/media/ti/4140f3a2-df5e-4822-9832-48f99154c59d/lijfdata/datasets/mixhaze_2to1'
    # syn_path = '/media/ti/4140f3a2-df5e-4822-9832-48f99154c59d/lijfdata/datasets/HID/DHID/TrainingSet'
    # syn_path = '/media/ti/4140f3a2-df5e-4822-9832-48f99154c59d/lijfdata/datasets/RSID/train'
    # syn_path = '/media/ti/4140f3a2-df5e-4822-9832-48f99154c59d/lijfdata/datasets/rice2/train'
    syn_path = '/media/ti/4140f3a2-df5e-4822-9832-48f99154c59d/lijfdata/datasets/Feng/train'
    # syn_path = '/home/good/lijf/rice2/train'
    # syn_path = '/home/good/lijf/Haze4K/train'
    train_dataset = Dataset_train(syn_path, opt.img_size)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset,
                            batch_size=opt.batch_size,
                            sampler=train_sampler,
                            num_workers=0, pin_memory=False,
                            drop_last=True)
    # 初始化模型和优化器
    device = torch.device(f"cuda:{local_rank}")
    net = MyModel.backbone_train.build_net().to(device)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    steps = len(train_loader) * opt.epochs
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    # pth_path = os.path.join(opt.train_res_dir, "train_supervised", "models", "supervised_2000.pth")
    # pth_path = '/home/ti/lijf/code/trambawave/results_rice1_vgg/train_supervised/models/supervised_1800.pth'
    # checkpoint = torch.load(pth_path, map_location=device,weights_only=True)
    # net.load_state_dict(checkpoint['model'])
    # Loss functions
    from methods import LOSS
    L_coeffs = LOSS.build_coeffs_loss()
    L_abs = nn.L1Loss()
    L_msssim = msssim.build_msssim_loss()
    # ssim_loss = pyiqa.create_metric('ssimc', device=device, as_loss=True)
    # psnr_loss = pyiqa.create_metric('psnr', device=device, as_loss=True)
    L_cc = LOSS.build_hsv_contrast_loss()
    # L_ycbcr = LOSS.build_ycbcr_contrast_loss()
    L_vgg = ContrastLoss(ablation=False)
    # 开始训练
    step=0
    for epoch in range(1, opt.epochs+1):
        net.train()
        train_sampler.set_epoch(epoch)  # 确保每个 epoch 的数据划分不同
        for data in train_loader:
            hazy= data["hazy"].to(device)
            clear = data["gt"].to(device)

            pred, coeffs = net(hazy)
            # Compute losses
            l_abs = L_abs(pred,clear)
            l_ms = L_msssim(pred, clear)
            # l_ss = 1-ssim_loss(pred, clear) 
            l_hf = L_coeffs(coeffs,clear)
            l_hsv = L_cc(clear,pred,hazy)
            # l_ycbcr = L_ycbcr(p=clear, g=pred, n=hazy)
            # l_vgg = L_vgg(pred,clear,hazy)
            # l_p = -psnr_loss(pred,clear)
            # loss = 2*l_ms + 5*l_abs + l_hf + 0.001*l_hsv + 0.004*l_vgg
            
            loss = 2*l_ms + 5*l_abs + l_hf + 0.001*l_hsv
            # # 反向传播和优化
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if rank == 0 and step % 10 ==0:
                log_msg = (f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                           f"step: {step}/{steps} | epoch: {epoch}/{opt.epochs} | "
                           f"lr: {scheduler.get_last_lr()[0]:.9f} | "
                           f"loss: {loss.item():.5f} |l_abs: {l_abs:.5f}| l_ms: {l_ms:.5f}| l_hsv: {l_hsv:.5f}|l_hf: {l_hf:.5f}  ")
                            # f"loss: {loss.item():.5f} | l_ms: {l_ms:.5f}| l_abs: {l_abs:.5f}")
                print(log_msg)
                with open(os.path.join(res_dir, "log", "loss.txt"), "a") as f:
                    f.write(log_msg + "\n")
        if epoch % opt.save_epoch == 0 and rank == 0:
            print("save model")
            path = os.path.join(res_dir, "models", f"supervised_{epoch}.pth")
            torch.save({'model': net.state_dict()}, path)

if __name__ == "__main__":
    # 初始化分布式模式
    rank, local_rank = init_distributed_mode()
    world_size = dist.get_world_size()

    # 解析参数
    opt = TrainOptions().parse()
    res_dir = os.path.join(opt.train_res_dir, "train_supervised")
    os.makedirs(os.path.join(res_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(res_dir, "log"), exist_ok=True)


    # 保存配置文件（仅主进程）
    if rank == 0:
        with open(os.path.join(res_dir, "log", "config.txt"), "w") as f:
            json.dump(opt.__dict__, f, indent=2)

    try:
        train(rank, local_rank, world_size)
    finally:
        cleanup_distributed()

#torchrun --nproc_per_node=1 train_supervised.py --epochs 100 --batch_size 18 --save_epoch 10

