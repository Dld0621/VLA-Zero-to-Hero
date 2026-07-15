# Stage 1: VLM 基础

> 理解视觉-语言模型如何将图像和文本映射到同一嵌入空间。

---

## 学习目标

完成本阶段后，你应该能够：

1. 解释 CLIP 的对比学习原理
2. 使用 HuggingFace Transformers 加载预训练 VLM
3. 计算图像-文本相似度
4. 理解 ViT 如何将图像 token 化

---

## 1.1 CLIP 原理

CLIP（Contrastive Language-Image Pre-training）是 VLA 的视觉-语言对齐基石。

### 核心思想

在大量（图像，文本）对上训练两个编码器，使得：
- 匹配的图像-文本对在嵌入空间距离近
- 不匹配的距离远

### 对比损失

```python
import torch
import torch.nn.functional as F

def contrastive_loss(image_features, text_features, temperature=0.07):
    """
    image_features: [N, D]
    text_features: [N, D]
    """
    # 归一化
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # 计算相似度矩阵
    logits = image_features @ text_features.T / temperature  # [N, N]

    # 对角线是正样本
    labels = torch.arange(N)

    # 双向交叉熵
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2
```

### 关键洞察

- **对比学习** 不需要人工标注的类别标签，只需要图文配对
- **Zero-shot**：训练后可以直接做图像分类（用类别名称作为文本查询）
- 后续 VLA 模型（OpenVLA 用 SigLIP，RT-2 用 CLIP）都基于这种对齐思想

---

## 1.2 运行 CLIP 推理

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# 加载模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 准备输入
image = Image.open("robot_scene.jpg")
texts = [
    "a robot arm reaching for a cup",
    "a person sitting at a desk",
    "a kitchen with appliances",
]

# 预处理
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# 推理
outputs = model(**inputs)
logits = outputs.logits_per_image  # [1, 3] 图像与每个文本的相似度
probs = logits.softmax(dim=1)

# 结果
for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob:.3f}")
```

**练习**：换不同的图像和文本，观察相似度变化。思考为什么 CLIP 对空间关系（"left of", "above"）理解较弱？

---

## 1.3 ViT 视觉 Token 化

Vision Transformer（ViT）将图像切分为 patch，每个 patch 变成一个 token：

```
输入图像: 224x224x3
    ↓
切分为 16x16 的 patch → 14x14 = 196 个 patch
    ↓
每个 patch 线性映射到 D 维 → 196 个 visual token
    ↓
加上 [CLS] token → 197 个 token
    ↓
输入 Transformer Encoder
```

### 可视化 Attention

```python
import matplotlib.pyplot as plt

# 获取最后一层的 attention
outputs = model.vision_model(inputs.pixel_values, output_attentions=True)
attentions = outputs.attentions  # tuple of [B, heads, N, N]

# 可视化 [CLS] token 对所有 patch 的注意力
attn = attentions[-1][0, 0, 0, 1:].reshape(14, 14).detach().numpy()
plt.imshow(attn, cmap='viridis')
plt.title("CLS attention over image patches")
plt.show()
```

**练习**：对比 CLIP ViT 和 DINOv2 的 attention map，观察两者关注区域的不同。

---

## 1.4 VLM 推理示例

使用 LLaVA（或类似模型）进行视觉问答：

```bash
pip install transformers accelerate
```

```python
from transformers import AutoProcessor, AutoModelForPreTraining
from PIL import Image

# 加载 LLaVA（示例，具体模型名可能变化）
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = AutoModelForPreTraining.from_pretrained("llava-hf/llava-1.5-7b-hf")

image = Image.open("scene.jpg")
prompt = "USER: <image>\nWhat objects can the robot interact with?\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```

---

## 验证检查点

完成以下任务即通过 Stage 1：

- [ ] 能解释 CLIP contrastive loss 的公式
- [ ] 能运行 CLIP 推理并解释相似度结果
- [ ] 能计算给定图像-文本对的相似度分数
- [ ] 理解 ViT 如何将 224x224 图像转换为 197 个 token

---

## 延伸阅读

- [CLIP 论文](https://arxiv.org/abs/2103.00020)
- [LLaVA 论文](https://arxiv.org/abs/2304.08485)
- [SigLIP 论文](https://arxiv.org/abs/2303.15343)
