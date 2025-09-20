import torch
from PIL import Image
import torchvision.transforms as transforms
from rcan import RCAN
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import numpy as np
import os

def load_model(weight_path, scale):
    model = RCAN(n_resgroups=3, n_resblocks=5, n_feats=64, scale=scale)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model

def super_resolve(model, img_path, hr_path, scale, save_path):
    img = Image.open(img_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    lr = to_tensor(img).unsqueeze(0)

    with torch.no_grad():
        sr = model(lr)
        sr = torch.clamp(sr, 0, 1)

    sr_img = to_pil(sr.squeeze(0))
    sr_img.save(save_path)
    print(f"放大结果已保存到: {save_path}")

    # 如果有 HR 真值，计算指标
    if hr_path is not None and os.path.exists(hr_path):
        hr_img = Image.open(hr_path).convert("RGB")
        hr = to_tensor(hr_img).numpy().transpose(1, 2, 0)
        sr_np = sr.squeeze(0).permute(1, 2, 0).numpy()

        hr = np.clip(hr, 0, 1)
        sr_np = np.clip(sr_np, 0, 1)

        psnr_val = psnr_metric(hr, sr_np, data_range=1.0)
        ssim_val = ssim_metric(hr, sr_np, channel_axis=2, data_range=1.0)

        print(f"PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

if __name__ == "__main__":
    scale = 2
    weight_path = "rcan_best.pth"
    img_name = 'test1-compress'
    lr_image_path = f"./test_img/{img_name}.jpg"
    hr_image_path = f"./test_img/{img_name}_HR.png"  # 真值路径（如果有的话）
    save_path = f"./predicted/result_{img_name}.png"

    model = load_model(weight_path, scale)
    super_resolve(model, lr_image_path, hr_image_path, scale, save_path)
