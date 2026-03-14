import torch
import cv2
import numpy as np
from PIL import Image
from fusion_model import CLIPFusionModel

SEG_CKPT_PATH = "D:\code\segmentation\evaluation_results\best.pth"
# ===================== 新手可选改 =====================
TEST_IMG_PATH = "./test.jpg"  # 测试图片路径（放一张图片到仓库根目录，命名为test.jpg）
TEXT_PROMPT = "medical image"  # 文本提示（适配医疗图像分割，可改）
NUM_CLASSES = 1  # 分割类别数（仓库模型默认1类，无需改）

# ===================== 初始化模型 =====================
# 自动选择GPU/CPU（有GPU用GPU，无则用CPU）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 使用设备：{DEVICE}")

# 创建融合模型实例
model = CLIPFusionModel(
    seg_ckpt_path=SEG_CKPT_PATH,
    num_classes=NUM_CLASSES,
    clip_model_name="ViT-B/32"  # 轻量版CLIP，避免显存不足
).to(DEVICE)
model.eval()  # 推理模式

# ===================== 加载测试图片 =====================
try:
    # 读取图片并转为RGB格式
    img = Image.open(TEST_IMG_PATH).convert("RGB")
except FileNotFoundError:
    print(f"❌ 未找到测试图片：{TEST_IMG_PATH}")
    print("💡 请放一张图片到仓库根目录，并重命名为test.jpg")
    exit()  # 图片不存在则终止运行

# 图片预处理（转为模型可识别的张量）
img_np = np.array(img) / 255.0  # 归一化到0-1
img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

# ===================== 执行CLIP+分割融合推理 =====================
print("🚀 开始融合推理...")
with torch.no_grad():  # 不计算梯度，加快速度
    fusion_out = model(img_tensor, text_prompt=TEXT_PROMPT)

# ===================== 可视化并保存结果 =====================
# 处理分割掩码（转为可视化格式）
seg_mask = fusion_out.squeeze().cpu().numpy()
# 归一化掩码到0-255（避免数值溢出）
seg_mask = (seg_mask - seg_mask.min()) / (seg_mask.max() - seg_mask.min() + 1e-8)  # 加1e-8防止除0
seg_mask = (seg_mask * 255).astype(np.uint8)
# 调整掩码尺寸匹配原图
seg_mask = cv2.resize(seg_mask, (img_np.shape[1], img_np.shape[0]))

# 叠加掩码到原图（可视化融合效果）
img_cv = cv2.cvtColor(img_np.astype(np.float32), cv2.COLOR_RGB2BGR)
seg_mask_color = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR) / 255.0
result_img = cv2.addWeighted(img_cv, 0.7, seg_mask_color, 0.3, 0)

# 保存结果到仓库根目录
cv2.imwrite("./fusion_result.jpg", result_img * 255)
print("🎉 验证完成！融合结果已保存为：fusion_result.jpg")