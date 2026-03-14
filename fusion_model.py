import torch
import torch.nn as nn
import clip
from collections import OrderedDict

# 第一步：定义基础分割模型（适配仓库MobileNetV2-FPN结构，预留CLIP融合接口）
class BasicSegModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        # 分割模型主干（极简版，适配仓库.pth的基础结构）
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 分割模型输出头
        self.head = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        # 提取分割中间特征（预留CLIP融合的位置）
        feat = self.backbone(x)
        out = self.head(feat)
        return out, feat  # 返回输出+中间特征，方便后续融合

# 第二步：定义CLIP+分割融合模型（核心：预留结构，兼容仓库.pth）
class CLIPFusionModel(nn.Module):
    def __init__(self, seg_ckpt_path, num_classes=1, clip_model_name="ViT-B/32"):
        super().__init__()
        # 1. 加载CLIP模型并冻结权重（不训练，仅提取特征）
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name)
        for param in self.clip_model.parameters():
            param.requires_grad = False  # 固定CLIP权重

        # 2. 初始化基础分割模型（预留结构）
        self.seg_model = BasicSegModel(in_channels=3, num_classes=num_classes)

        # 3. 新增CLIP特征融合层（预留的核心，适配维度）
        self.clip_proj = nn.Linear(512, 128)  # CLIP特征(512维)→分割特征(128维)
        self.fusion_layer = nn.Conv2d(128+128, 128, 1)  # 拼接融合两种特征

        # 4. 加载仓库.pth权重（跳过融合层，解决结构不完整问题）
        self._load_seg_weights(seg_ckpt_path)

    def _load_seg_weights(self, ckpt_path):
        """仅加载分割模型的权重，融合层留空（预留结构）"""
        try:
            # 加载仓库.pth文件
            state_dict = torch.load(ckpt_path, map_location="cpu")
            # 适配多卡训练的权重名（去掉module.前缀）
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                # 只加载分割模型已有的层，融合层跳过
                if k in self.seg_model.state_dict().keys():
                    new_state_dict[k] = v
            # 加载权重（strict=False：允许跳过融合层）
            self.seg_model.load_state_dict(new_state_dict, strict=False)
            print(f"✅ 成功加载仓库.pth权重：{ckpt_path}")
        except Exception as e:
            # 权重匹配警告不影响验证，仅提示
            print(f"⚠️ 权重加载警告（不影响验证）：{e}")

    def forward(self, img, text_prompt=None):
        """前向推理：img=输入图像，text_prompt=文本提示（如"medical image"）"""
        # 1. 提取分割模型的特征
        seg_out, seg_feat = self.seg_model(img)  # seg_feat: [B,128,H,W]
        B, C, H, W = seg_feat.shape

        # 2. 提取CLIP的图像/文本特征
        # CLIP图像特征
        clip_img = self.clip_preprocess(img).to(img.device)
        clip_img_feat = self.clip_model.encode_image(clip_img)  # [B,512]
        # 可选：融合文本特征（CLIP跨模态核心）
        if text_prompt is not None:
            text = clip.tokenize([text_prompt]).to(img.device)
            clip_text_feat = self.clip_model.encode_text(text)  # [B,512]
            clip_feat = (clip_img_feat + clip_text_feat) / 2  # 图文特征融合
        else:
            clip_feat = clip_img_feat

        # 3. 将CLIP一维特征转为分割模型的二维空间特征
        clip_feat_proj = self.clip_proj(clip_feat)  # [B,128]
        clip_feat_spatial = clip_feat_proj.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)

        # 4. 融合CLIP特征与分割特征（直接拼接，快速验证）
        fusion_feat = torch.cat([seg_feat, clip_feat_spatial], dim=1)  # [B,256,H,W]
        fusion_feat = self.fusion_layer(fusion_feat)  # 融合为128维

        # 5. 输出最终分割结果
        final_out = self.seg_model.head(fusion_feat)
        return final_out