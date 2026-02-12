import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

FOCUS_ROOT = r"D:\FOCUS-dataset"
SUBSETS = ["testing", "training", "validation"]
RESULT_ROOT = os.path.join(FOCUS_ROOT, "FOCUS_visualization_results")


COLOR_DICT = {
    1: (0, 255, 0),      # cardiac - 绿
    2: (0, 180, 255),    # thorax - 蓝
    3: (255, 165, 0),    # chamber1 - 橙
    4: (255, 0, 0),      # chamber2 - 红
    5: (255, 255, 0),
}


# ================= ellipse =================
def read_ellipse(txt_path):
    res = []
    if not os.path.exists(txt_path):
        return res
    with open(txt_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 6:
                continue
            cx, cy, a, b, angle = map(float, p[:5])
            label = p[5]
            res.append((cx, cy, a, b, angle, label))
    return res


# ================= rectangle =================
def read_rect(txt_path):
    res = []
    if not os.path.exists(txt_path):
        return res
    with open(txt_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 9:
                continue
            pts = [(float(p[i]), float(p[i+1])) for i in range(0, 8, 2)]
            label = p[8]
            res.append((pts, label))
    return res


# ===== mask转彩色 =====
def mask_to_overlay(img_np, mask_np):

    color_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)

    for cls, color in COLOR_DICT.items():
        color_mask[mask_np == cls] = color

    # 半透明融合
    overlay = cv2.addWeighted(img_np, 0.65, color_mask, 0.35, 0)

    return overlay


# ================= visualization =================
def visualize(subset, img_name):

    base = os.path.splitext(img_name)[0]
    subset_path = os.path.join(FOCUS_ROOT, subset)

    img_path = os.path.join(subset_path, "images", img_name)
    mask_path = os.path.join(subset_path, "annfiles_mask", f"{base}_mask.png")
    ell_path = os.path.join(subset_path, "annfiles_ellipse", f"{base}.txt")
    rec_path = os.path.join(subset_path, "annfiles_rectangle", f"{base}.txt")

    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # ===== mask overlay =====
    if os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(img.size)
        mask_np = np.array(mask)

        img_np = mask_to_overlay(img_np, mask_np)

    plt.figure(figsize=(8,8))
    ax = plt.gca()
    ax.imshow(img_np)

    # ===== ellipse=====
    for cx, cy, a, b, angle, label in read_ellipse(ell_path):

        ell = patches.Ellipse(
            (cx, cy),
            2*a, 2*b,
            angle=angle,
            linewidth=0,              # 去边框
            edgecolor=None,
            facecolor=(0,1,0,0.25)    # 半透明绿色区域
        )

        ax.add_patch(ell)

    # ===== rectangle=====
    for pts, label in read_rect(rec_path):

        poly = patches.Polygon(
            pts,
            linewidth=0,              # 去边框
            edgecolor=None,
            facecolor=(1,0,0,0.25)    # 半透明红色区域
        )

        ax.add_patch(poly)

    ax.axis('off')

    save_dir = os.path.join(RESULT_ROOT, subset)
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(
        os.path.join(save_dir, f"vis_{img_name}"),
        dpi=200,
        bbox_inches='tight'
    )
    plt.close()


# ================= run =================
if __name__ == "__main__":

    for subset in SUBSETS:
        img_dir = os.path.join(FOCUS_ROOT, subset, "images")

        for img in os.listdir(img_dir):
            if img.endswith(('.png','.jpg','.jpeg')):
                visualize(subset, img)

    print("全部可视化完成")
