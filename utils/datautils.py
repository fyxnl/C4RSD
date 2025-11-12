import torch
import os
from PIL import Image
import random
from torchvision.transforms import functional as FF
from torchvision.transforms import Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, Resize
import torch.utils.data as data
from torch.utils.data import DataLoader

def preprocess_feature(img):
    img = ToTensor()(img)
    clip_normalizer = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    img = clip_normalizer(img)
    return img
class Dataset_train(data.Dataset):
    def __init__(self, path, img_size):
        super(Dataset_train, self).__init__()

        self.haze_names = os.listdir(os.path.join(path, "hazy"))
        self.haze_pathes = [os.path.join(path, "hazy", name) for name in self.haze_names]
        self.clear_dir = os.path.join(path, "clear")
        self.size = img_size
    def augData(self, data, target):
        rand_hor = random.randint(0, 1)
        rand_rot = random.randint(0, 3)
        data = RandomHorizontalFlip(rand_hor)(data)
        target = RandomHorizontalFlip(rand_hor)(target)
        if rand_rot:
            data = FF.rotate(data, 90 * rand_rot)
            target = FF.rotate(target, 90 * rand_rot)
        target = ToTensor()(target)
        return preprocess_feature(data), target
    def __getitem__(self, index):
        haze = Image.open(self.haze_pathes[index]).convert('RGB')
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 10000)
                haze = Image.open(self.haze_pathes[index]).convert('RGB')
        haze_name = self.haze_names[index]
        
        clear_name = haze_name
        clear_path = os.path.join(self.clear_dir, clear_name)
        if not os.path.exists(clear_path):
            clear_name = haze_name.split("_")[0] + '.png'
            clear_path = os.path.join(self.clear_dir, clear_name)
        clear = Image.open(clear_path).convert('RGB')
        # print(haze_name)
        # print(clear_name)
        i, j, h, w = RandomCrop.get_params(haze, output_size=(self.size, self.size))
        haze = FF.crop(haze, i, j, h, w)
        clear = FF.crop(clear, i, j, h, w)
        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))

        tar_data = {"hazy": haze,"gt": clear,"name": clear_name}

        return tar_data
    
    def __len__(self):
        return len(self.haze_pathes)
    
class Dataset_synthesis(data.Dataset):
    def __init__(self, path):
        super(Dataset_synthesis, self).__init__()

        self.haze_names= os.listdir(os.path.join(path, "hazy"))
        self.haze_pathes = [os.path.join(path, "hazy", name) for name in self.haze_names]
        self.clear_dir = os.path.join(path, "clear")

    def __getitem__(self, index):
        haze = Image.open(self.haze_pathes[index]).convert('RGB')
        haze_name = self.haze_names[index]
        name_variants = [
        haze_name.split("_")[0] + '.png',  # 第一种命名方式
        haze_name.split("_")[0] + '.jpg',  # 第二种命名方式
        haze_name  # 第三种命名方式
        ]

    # 遍历命名方式，找到第一个存在的路径
        for clear_name in name_variants:
            clear_path = os.path.join(self.clear_dir, clear_name)
            if os.path.exists(clear_path):
                break
        clear = Image.open(clear_path).convert('RGB')

        # tensor
        haze = preprocess_feature(haze)
        clear = ToTensor()(clear)

        tar_data = {"hazy": haze,
                    "gt": clear,
                    "name": haze_name}

        return tar_data
    
    def __len__(self):
        return len(self.haze_pathes)

class Dataset_real_world(data.Dataset):
    def __init__(self, path):
        super(Dataset_real_world, self).__init__()

        self.haze_names= os.listdir(path)
        self.haze_pathes = [os.path.join(path, name) for name in self.haze_names]

    def __getitem__(self, index):
        haze = Image.open(self.haze_pathes[index]).convert('RGB')

        haze = preprocess_feature(haze)

        tar_data = {"hazy": haze,
                    "name": self.haze_names[index]}

        return tar_data
    
    def __len__(self):
        return len(self.haze_pathes)

def get_train_dataset(dataset, img_size):
    supported_dataset = {
            "Feng": Dataset_train
        }
    path_dict = {
            "Feng":'/media/ti/4140f3a2-df5e-4822-9832-48f99154c59d/lijfdata/datasets/Feng/train'
        }

    try:
        data_root= path_dict[dataset]
    except:
        raise ValueError("dataset not supported")

    train_dataset = supported_dataset[dataset](data_root, img_size)
    return train_dataset



def get_val_loader(dataset):
    supported_dataset = {
        "RTTS": Dataset_real_world,
        "URHI": Dataset_real_world,
        "Fattal": Dataset_real_world,
        "SOTSoutdoor": Dataset_synthesis,
        "SOTSindoor": Dataset_synthesis,
        "dense_haze": Dataset_real_world,
        "sandstorm": Dataset_real_world,
        "yewan": Dataset_real_world,
        "miniRTTS": Dataset_real_world,
        "URHI_500": Dataset_real_world,
        "RSID":Dataset_synthesis,
        "DHID":Dataset_synthesis,
        "rice1":Dataset_synthesis,
        "rice2":Dataset_synthesis,
        "T-cloud":Dataset_synthesis,
        "haze4k":Dataset_synthesis,
    }
    path_dict = {
        "RTTS": "/media/ti/4140f3a2-df5e-4822-9832-48f99154c59d/lijfdata/datasets/RTTS",
        "miniRTTS": "miniRTTS",
        "URHI": "/home/lijf/mine/code/resize_for_test/URHI",
        "URHI_500": "/media/ti/4140f3a2-df5e-4822-9832-48f99154c59d/lijfdata/datasets/URHI_500",
        "Fattal": "/media/ti/4140f3a2-df5e-4822-9832-48f99154c59d/lijfdata/datasets/Fattal",
        "SOTSoutdoor": "/media/ti/4140f3a2-df5e-4822-9832-48f99154c59d/lijfdata/datasets/RESIDE/SOTS/outdoor",
        "SOTSindoor": "/home/lijf/mine/datasets/RESIDEOTS/SOTS/indoor",
        "dense_haze": "/home/lijf/mine/datasets/NID/hazy/dense_haze",
        "sandstorm": "/home/lijf/mine/datasets/real_world_datasets/Sandstorm",
        "yewan": "/home/lijf/mine/datasets/real_world_datasets/Yewan",
        "RSID":"/media/ti/4140f3a2-df5e-4822-9832-48f99154c59d/lijfdata/datasets/RSID/test",
        "DHID":"/media/ti/4140f3a2-df5e-4822-9832-48f99154c59d/lijfdata/datasets/HID/DHID/TestingSet/Test",
        "rice1":"/media/ti/4140f3a2-df5e-4822-9832-48f99154c59d/lijfdata/datasets/rice1/test",
        "rice2":"/home/good/lijf/rice2/test",
        "T-cloud":"/media/ti/4140f3a2-df5e-4822-9832-48f99154c59d/lijfdata/datasets/T-Cloud/test",
        "haze4k":"/home/good/lijf/Haze4K/test",
    }
    try:
        data_root = path_dict[dataset]
    except:
        raise ValueError("dataset not supported")
    
    val_dataset = supported_dataset[dataset](data_root)

    val_loader = DataLoader(val_dataset,
                            batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=False,
                            drop_last=False)
    return val_loader