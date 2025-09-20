import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from repreparing import SRDataset
from rcan import RCAN
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
评估指标：
PSNR: 衡量重建图像与真值的误差，值越高越好（单位 dB）     范围： (0, ∞)
SSIM: 衡量结构、亮度、对比度的相似性，值越接近 1 越好     范围： [0, 1]
'''

# 参数
scale = 2
batch_size = 4
epochs = 150
lr = 1e-4

# 数据集路径
hr_dir = "dataset/HR"
dataset = SRDataset(hr_dir, scale=scale)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 验证集（此处为了简单直接用 dataset，也可以使用单独 val_dataset）
val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 模型
model = RCAN(n_resgroups=3, n_resblocks=5, n_feats=64, scale=scale).to(device)

# 损失函数 & 优化器
criterion = nn.L1Loss()
optimizer = Adam(model.parameters(), lr=lr)

best_psnr = 0

# --- 新增验证函数 ---
def evaluate(model, dataloader):
    model.eval()
    total_psnr, total_ssim, count = 0, 0, 0
    with torch.no_grad():
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            preds = model(lr_imgs)

            # 转到 numpy
            sr = preds.squeeze(0).permute(1, 2, 0).cpu().numpy()
            hr = hr_imgs.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # 裁剪到 [0,1]
            sr = np.clip(sr, 0, 1)
            hr = np.clip(hr, 0, 1)

            psnr_val = psnr_metric(hr, sr, data_range=1.0)
            ssim_val = ssim_metric(hr, sr, channel_axis=2, data_range=1.0)

            total_psnr += psnr_val
            total_ssim += ssim_val
            count += 1

    return total_psnr / count, total_ssim / count


# --- 训练循环 ---
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(dataloader, total=len(dataloader))

    for lr_imgs, hr_imgs in pbar:
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

        preds = model(lr_imgs)
        loss = criterion(preds, hr_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_description(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(dataloader)

    # --- 计算验证指标 ---
    avg_psnr, avg_ssim = evaluate(model, val_loader)
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")

    # 保存最佳模型（以 PSNR 为标准）
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), "rcan_best.pth")
        print(f"💾 保存最优模型: PSNR={best_psnr:.2f} -> rcan_best.pth")

print("训练完成")
print(f"💾 保存最优模型: PSNR={best_psnr:.2f} -> rcan_best.pth")