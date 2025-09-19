import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class SRDataset(Dataset):
    def __init__(self, hr_dir, scale=2, patch_size=128):
        super(SRDataset, self).__init__()
        self.hr_dir = hr_dir
        self.hr_images = sorted(os.listdir(hr_dir))
        self.scale = scale
        self.patch_size = patch_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr = Image.open(os.path.join(self.hr_dir, self.hr_images[idx])).convert("RGB")
        w, h = hr.size

        # 随机裁剪 HR patch
        if w > self.patch_size and h > self.patch_size:
            x = random.randint(0, w - self.patch_size)
            y = random.randint(0, h - self.patch_size)
            hr = hr.crop((x, y, x + self.patch_size, y + self.patch_size))

        # 生成 LR
        lr = hr.resize((self.patch_size // self.scale, self.patch_size // self.scale), Image.BICUBIC)

        return self.to_tensor(lr), self.to_tensor(hr)

