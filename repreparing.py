import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class SRDataset(Dataset):
    def __init__(self, hr_dir, scale=2, patch_size=48):
        """
        hr_dir: 高分辨率图像路径
        scale: 放大倍数
        patch_size: 从 LR 图像提取的 patch 大小 (HR patch 大小 = patch_size * scale)
        """
        self.hr_images = sorted(glob.glob(os.path.join(hr_dir, '*')))
        self.scale = scale
        self.patch_size = patch_size

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        # HR 图像
        hr = Image.open(self.hr_images[idx]).convert("RGB")

        # 生成对应的 LR (简单用 Bicubic 缩小)
        w, h = hr.size
        lr = hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)

        # 转 Tensor 之前做 patch 裁剪
        lr, hr = self._get_patch(lr, hr)

        # 数据增强
        lr, hr = self._augment(lr, hr)

        # 转 tensor 并归一化到 [0,1]
        lr = TF.to_tensor(lr)
        hr = TF.to_tensor(hr)

        return lr, hr

    def _get_patch(self, lr, hr):
        """随机裁剪 patch，LR patch 大小为 patch_size，HR patch 大小为 patch_size*scale"""
        lr_w, lr_h = lr.size
        ps = self.patch_size
        x = random.randint(0, lr_w - ps)
        y = random.randint(0, lr_h - ps)

        lr_patch = lr.crop((x, y, x + ps, y + ps))
        hr_patch = hr.crop((x * self.scale, y * self.scale,
                            (x + ps) * self.scale, (y + ps) * self.scale))
        return lr_patch, hr_patch

    def _augment(self, lr, hr):
        """随机翻转和旋转 (保证配对同步增强)"""
        # 随机水平翻转
        if random.random() < 0.5:
            lr = TF.hflip(lr)
            hr = TF.hflip(hr)
        # 随机垂直翻转
        if random.random() < 0.5:
            lr = TF.vflip(lr)
            hr = TF.vflip(hr)
        # 随机旋转 0, 90, 180, 270 度
        if random.random() < 0.5:
            angle = random.choice([0, 90, 180, 270])
            lr = lr.rotate(angle, expand=True)
            hr = hr.rotate(angle, expand=True)

        return lr, hr
