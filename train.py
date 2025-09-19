import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from repreparing import SRDataset
from rcan import RCAN  # 如果是 DenseRCAN 就写 from rcan import DenseRCAN
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import numpy as np

# ===================== 设置参数 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scale = 2
batch_size = 4
epochs = 150
lr = 1e-4

# 数据集路径（只需要 HR 图）
hr_dir = "dataset/HR"

# ===================== 数据加载 =====================
# 训练集
train_dataset = SRDataset(hr_dir, scale=scale)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 验证集（这里简单用训练集拷贝一份当验证集，你也可以替换成真实的验证集）
val_dataset = SRDataset(hr_dir, scale=scale)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# ===================== 模型 =====================
model = RCAN(n_resgroups=3, n_resblocks=5, n_feats=64, scale=scale).to(device)
# 如果想用 DenseRCAN：
# model = DenseRCAN(n_resgroups=3, n_resblocks=5, n_feats=64, scale=scale).to(device)

# ===================== 损失函数 & 优化器 =====================
criterion = nn.L1Loss()
optimizer = Adam(model.parameters(), lr=lr)

# ===================== 计算评价指标 =====================
def evaluate(model, dataloader):
    """在验证集上计算 PSNR 和 SSIM"""
    model.eval()
    total_psnr, total_ssim, count = 0, 0, 0
    with torch.no_grad():
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = model(lr_imgs)  # 超分推理

            # Tensor(C,H,W) -> Numpy(H,W,C)
            sr = sr_imgs.squeeze(0).permute(1, 2, 0).cpu().numpy()
            hr = hr_imgs.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # 保证范围在 [0,1]
            sr = np.clip(sr, 0, 1)
            hr = np.clip(hr, 0, 1)

            # 计算 PSNR 和 SSIM
            psnr_val = psnr_metric(hr, sr, data_range=1.0)
            ssim_val = ssim_metric(hr, sr, data_range=1.0, channel_axis=2)

            total_psnr += psnr_val
            total_ssim += ssim_val
            count += 1

    return total_psnr / count, total_ssim / count

# ===================== 训练 =====================
best_psnr = 0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

    for batch in pbar:
        lr_imgs, hr_imgs = batch
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

        preds = model(lr_imgs)
        loss = criterion(preds, hr_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_loss = epoch_loss / len(train_loader)

    # ===== 每个 epoch 结束后计算验证集 PSNR/SSIM =====
    avg_psnr, avg_ssim = evaluate(model, val_loader)
    print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.4f} dB | SSIM: {avg_ssim:.4f}")

    # 如果 PSNR 更优，则保存模型
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), "rcan_best.pth")
        print(f"💾 保存最优模型 (PSNR={best_psnr:.4f} dB) -> rcan_best.pth")

print("✅ 训练完成")
print(f"🏆 最佳 PSNR: {best_psnr:.4f} dB 已保存至 rcan_best.pth")
