import torch
from PIL import Image
import torchvision.transforms as transforms
from rcan import RCAN  # 你的 RCAN 模型定义文件


def load_model(weight_path, scale, n_resgroups=3, n_resblocks=5, n_feats=64):
    """加载模型权重"""
    model = RCAN(n_resgroups=n_resgroups, n_resblocks=n_resblocks, n_feats=n_feats, scale=scale)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model


def prepare_lr_image(img_path, scale, is_hr=True):
    """读取图片并生成LR版本（如果是高清HR，则缩小获得LR）"""
    img = Image.open(img_path).convert("RGB")

    if is_hr:  # 如果输入是高清原图
        w, h = img.size
        lr_img = img.resize((w // scale, h // scale), Image.BICUBIC)
        print(f"[INFO] 输入为高清图，已缩小为 LR: {lr_img.size}")
        return lr_img
    else:
        print(f"[INFO] 输入已是低分辨率图: {img.size}")
        return img


def super_resolve(model, lr_img, save_path):
    """模型推理"""
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # 转为Tensor，并加batch维度
    lr_tensor = to_tensor(lr_img).unsqueeze(0)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)
        sr_tensor = torch.clamp(sr_tensor, 0, 1)

    sr_img = to_pil(sr_tensor.squeeze(0))
    sr_img.save(save_path)
    print(f"✅ 超分结果已保存到: {save_path}")


if __name__ == "__main__":
    # ======= 配置 ========
    scale = 2
    weight_path = "rcan_best.pth"  # 训练好的权重文件
    input_image = f"./test_img/0804_lr(1).jpg"  # 测试图片路径
    save_path = "result_SR.png"  # 输出保存路径
    is_hr_input = True  # ✅ True 表示上传的是高清原图；False 表示已经是低清图
    # ====================

    # 加载模型
    model = load_model(weight_path, scale, n_resgroups=3, n_resblocks=5, n_feats=64)

    # 生成 LR 图
    lr_img = prepare_lr_image(input_image, scale, is_hr=is_hr_input)

    # 推理 & 保存结果
    super_resolve(model, lr_img, save_path)
