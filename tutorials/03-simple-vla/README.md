# Stage 3: 简单 VLA

> 搭建一个最小的 VLA 推理 pipeline，从预训练组件组装到真实模型推理。

---

## 学习目标

完成本阶段后，你应该能够：

1. 用预训练组件搭建最小 VLA 架构
2. 成功运行 OpenVLA 推理
3. 理解模型输出的含义和反归一化
4. 理解 VLA 推理的完整 pipeline

---

## 3.1 最小 VLA 架构

组合预训练组件，构建端到端 VLA：

```python
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoTokenizer

class SimpleVLA(nn.Module):
    """
    使用预训练 CLIP + 小型语言模型搭建的最小 VLA。
    仅用于教学，不用于实际控制。
    """
    def __init__(self, action_dim=7):
        super().__init__()
        # 加载预训练 CLIP
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # 冻结 CLIP（可选）
        for param in self.clip.parameters():
            param.requires_grad = False

        clip_dim = 512  # CLIP base 的输出维度

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(clip_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, images, texts):
        """
        Args:
            images: list of PIL Images
            texts: list of strings
        Returns:
            actions: [B, action_dim]
        """
        # CLIP 编码
        inputs = self.clip_processor(text=texts, images=images, return_tensors="pt", padding=True)
        outputs = self.clip(**inputs)

        image_features = outputs.image_embeds  # [B, 512]
        text_features = outputs.text_embeds     # [B, 512]

        # 融合
        fused = torch.cat([image_features, text_features], dim=-1)
        fused = self.fusion(fused)

        # 动作
        actions = self.policy_head(fused)
        return actions
```

**关键理解**：
- 视觉编码器和文本编码器都是**预训练**的，提供强大的表征
- 只有融合层和策略头是**随机初始化**的，需要训练
- 这就是 VLA 的**迁移学习**本质：利用 VLM 的知识，只学习动作映射

---

## 3.2 使用 OpenVLA 推理

### 环境准备

```bash
pip install transformers>=4.40.0 accelerate torch
```

### 完整推理代码

```python
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# 加载模型
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True,
)

model = model.to("cuda")
model.eval()

# 加载图像
image = Image.open("scene.jpg").convert("RGB")

# 构建 prompt
# OpenVLA 使用特定格式："In: ...\nOut:"
prompt = "In: What action should the robot take to pick up the red cup?\nOut:"

# 预处理
inputs = processor(prompt, image).to("cuda")

# 推理
with torch.no_grad():
    action = model.predict_action(inputs, unnorm_key="bridge")

print(f"Predicted action: {action}")
```

### OpenVLA 输出解析

OpenVLA 默认输出 7 维 delta 位姿：

```
action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

| 维度 | 含义 | 单位 |
|------|------|------|
| 0-2 | 末端位置增量 (x, y, z) | 米（经反归一化后） |
| 3-5 | 末端旋转增量 (roll, pitch, yaw) | 弧度 |
| 6 | 夹爪开合 | 连续值，需阈值化 |

### 反归一化（Unnormalization）

模型输出是归一化的，需要映射回实际物理量：

```python
# 不同数据集有不同的统计量
data_stats = {
    "bridge": {
        "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "std": [0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 1.0],
    },
    "libero": {
        # LIBERO 的统计量
    },
}

# 反归一化
action_physical = action * std + mean
```

`unnorm_key` 参数告诉模型使用哪个数据集的统计量。

---

## 3.3 VLA 推理 Pipeline 完整流程

```
1. 图像采集
   └── 从相机获取 RGB 图像 (H, W, 3)

2. 语言指令
   └── 用户输入: "pick up the red cup"

3. 预处理
   └── 图像: resize → normalize → tensor
   └── 文本: tokenize → input_ids

4. 模型推理
   └── VLA model.forward(image, text) → action

5. 后处理
   └── 反归一化: action * std + mean
   └── 裁剪到安全范围
   └── 可选: 平滑滤波

6. 执行
   └── 发送动作到机器人控制器
   └── 或: 发送到仿真环境

7. 循环
   └── 获取新图像 → 重复步骤 1-6
```

---

## 3.4 使用 Octo（轻量替代）

如果 GPU 显存不足（< 16GB），可以使用 Octo：

```python
import tensorflow as tf  # Octo 基于 JAX/TensorFlow
from octo.model.octo_model import OctoModel

# 加载模型
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")

# 准备输入
from PIL import Image
import numpy as np

image = np.array(Image.open("scene.jpg").resize((256, 256)))
task = model.create_tasks(texts=["pick up the red cup"])

# 推理
actions = model.sample_actions(
    image,
    task,
    unnormalize=True,
)
print(actions["actions"])
```

Octo 的优势：
- 27M-93M 参数，可在单卡甚至 CPU 运行
- 支持 Goal Image Conditioning
- 更灵活的输入格式

---

## 3.5 推理优化技巧

### 1. 量化（Quantization）

减少显存占用：

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    quantization_config=bnb_config,
    trust_remote_code=True,
)
```

### 2. 批量推理

一次处理多个样本：

```python
images = [Image.open(f"scene_{i}.jpg") for i in range(4)]
texts = ["pick up the cup"] * 4
inputs = processor(texts, images, return_tensors="pt", padding=True).to("cuda")
actions = model.predict_action(inputs, unnorm_key="bridge")
```

### 3. 推理频率控制

VLA 模型推理慢，常用 Action Chunking 降低频率：

```python
# 每 5 步重新推理一次
action_chunk = model.predict_action(inputs, unnorm_key="bridge")
for t in range(5):
    robot.step(action_chunk[t])
```

---

## 验证检查点

- [ ] 能成功加载并运行 OpenVLA 推理
- [ ] 理解 OpenVLA 输出的 7 个维度的含义
- [ ] 理解 `unnorm_key` 的作用
- [ ] 能解释从图像到动作的完整 pipeline
- [ ] （可选）能在 Octo 上运行推理

---

## 常见问题

**Q: 运行 OpenVLA 时出现 OOM？**
A: 尝试：1) 使用 bfloat16；2) 8-bit 量化；3) 换用 Octo；4) 使用更小 batch_size。

**Q: 模型输出的动作看起来随机？**
A: 检查 `unnorm_key` 是否匹配你的数据分布。如果使用随机图像，输出也会随机。

**Q: 推理速度太慢？**
A: 使用 Action Chunking，或换用更小的模型（Octo-Base）。

---

## 延伸阅读

- [OpenVLA 官方文档](https://openvla.github.io/)
- [Octo 官方文档](https://octo-models.github.io/)
- [Transformers 多模态教程](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder)
