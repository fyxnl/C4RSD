import os
import sys
sys.path.append("..")

import torch
import torchvision
from tqdm import tqdm
import pyiqa
from utils.datautils import get_val_loader
from methods import MyModel, DEA_model
from options.test_options import TestOptions

from utils.seed import setup_seed
setup_seed(42)




def inference_no_gt():
    mymodel.eval()  # 设置模型为评估模式
    # 定义无参考指标
    metrics = {
        # 'nima': pyiqa.create_metric('nima', device=device),
        # 'entropy': pyiqa.create_metric('entropy', device=device),
        # 'brisque': pyiqa.create_metric('brisque', device=device),
        # 'niqe': pyiqa.create_metric('niqe', device=device),
        # 'pi': pyiqa.create_metric('pi', device=device),
        # 'piqe': pyiqa.create_metric('piqe', device=device),
    }
    # 初始化总分字典
    total_metrics = {metric: 0 for metric in []}
    
    with torch.no_grad():
        for data in tqdm(val_loader):
            hazy_image = data["hazy"].to(device)
            image_name = data["name"][0]
            
            # 预测图像
            pred_image,_= mymodel(hazy_image)
            pred_image = torch.clamp(pred_image, 0, 1)

            # 保存生成的图像
            torchvision.utils.save_image(pred_image, os.path.join(save_dir, image_name))

            # 计算无参考指标
            results = {}
            for metric_name, metric_func in metrics.items():
                # print(metric_name)
                results[metric_name] = metric_func(pred_image).item()
                # print(results[metric_name])
            for metric_name, score in results.items():
                total_metrics[metric_name] += score

        # 计算并打印平均值
        num_samples = len(val_loader)
        for metric_name, total_score in total_metrics.items():
            avg_score = total_score / num_samples
            metric = metrics[metric_name]
            
            if metric.lower_better:  # 如果越低越好
                print(f"Average {metric_name} score: {avg_score:.4f} (lower is better)")
            else:  # 如果越高越好
                print(f"Average {metric_name} score: {avg_score:.4f} (higher is better)")


def inference_with_gt():
    mymodel.eval()  # 设置模型为评估模式
    # 定义有参考指标
    metrics = {
        # 'lpips': pyiqa.create_metric('lpips', device=device),
        'psnr': pyiqa.create_metric('psnr', device=device),
        'ssim': pyiqa.create_metric('ssim', device=device),
        # 'vif': pyiqa.create_metric('vif', device=device),
    }
    # 初始化总分字典
    total_metrics = {metric: 0 for metric in ['psnr', 'ssim']}
    
    with torch.no_grad():
        for data in tqdm(val_loader):
            hazy_image = data["hazy"].to(device)
            clear_image = data["gt"].to(device)
            image_name = data["name"][0]

            # 预测图像
            pred_image= mymodel(hazy_image)
            pred_image = torch.clamp(pred_image, 0, 1)
            # print(pred_image.shape)
            # print(clear_image.shape)
            # 保存生成的图像
            torchvision.utils.save_image(pred_image, os.path.join(save_dir, image_name))

            # 计算有参考指标
            results = {}
            for metric_name, metric_func in metrics.items():
                results[metric_name] = metric_func(pred_image, clear_image).item()
            for metric_name, score in results.items():
                total_metrics[metric_name] += score

        # 计算并打印平均值
        num_samples = len(val_loader)
        for metric_name, total_score in total_metrics.items():
            avg_score = total_score / num_samples
            metric = metrics[metric_name]
            
            if metric.lower_better:  # 如果越低越好
                print(f"Average {metric_name} score: {avg_score:.4f} (lower is better)")
            else:  # 如果越高越好
                print(f"Average {metric_name} score: {avg_score:.4f} (higher is better)")



if __name__ == "__main__":
    opt = TestOptions().parse()
    device = opt.device
    from collections import OrderedDict
    #加载模型
    mymodel = DEA_model.backbone_train.build_net().to(device) #注意消融的时候要改HFIP.py，因为load的是参数
    pth_path1=os.path.join(opt.pth_dir, "train_supervised","models", opt.model_name)
    pth1 = torch.load(pth_path1, map_location=device, weights_only=True)
    new_state_dict = OrderedDict()
    for k, v in pth1['model'].items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    mymodel.load_state_dict(new_state_dict)
    print("Successfully loaded model using alternate method.")

    #参数量和计算量
    from thop import profile
    input = torch.randn(1, 3, 256, 256).to(device) 
    Flops, params = profile(mymodel, inputs=(input,)) # macs
    print('Flops: % .4fG'%(Flops / 1000000000))# 计算量
    print('params参数量: % .4fM'% (params / 1000000)) #参数量：等价与上面的summary输出的Total params值

    if opt.synthesis:
        save_dir = os.path.join(opt.test_res_dir, "inference_supervised",opt.model_name, "synthesis", opt.test_dataset)
        os.makedirs(save_dir, exist_ok=True)
        val_loader = get_val_loader(opt.test_dataset)
        inference_with_gt()

    if opt.realworld:
        save_dir = os.path.join(opt.test_res_dir,  "inference_supervised",opt.model_name, "realworld", opt.test_dataset)
        os.makedirs(save_dir, exist_ok=True)
        val_loader = get_val_loader(opt.test_dataset)
        inference_no_gt()




#python inference_supervised.py --test_dataset haze4k --model_name supervised_100.pth --synthesis 
#python inference_supervised.py --test_dataset Fattal --model_name supervised_30.pth --realworld