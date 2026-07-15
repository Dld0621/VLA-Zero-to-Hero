# Stage 4: 微调实践

> 在自定义数据或仿真环境上微调 VLA 模型。

---

## 学习目标

完成本阶段后，你应该能够：

1. 准备符合 VLA 格式的训练数据
2. 使用 LoRA 高效微调大模型
3. 在 LIBERO 仿真环境上训练和评估
4. 理解微调的关键超参数

---

## 4.1 数据准备

### VLA 数据格式

标准的数据样本包含：

```json
{
  "observation": {
    "image": "rgb_image.jpg",
    "wrist_image": "wrist_camera.jpg",
    "instruction": "pick up the red cup and place it on the left"
  },
  "action": [0.01, -0.02, 0.005, 0.0, 0.0, 0.01, 1.0],
  "dataset_name": "my_dataset",
  "robot_type": "franka"
}
```

### 数据预处理

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import json

class VLADataset(Dataset):
    """VLA 训练数据集。"""

    def __init__(self, data_root, processor, action_mean, action_std):
        self.data_root = data_root
        self.processor = processor
        self.action_mean = torch.tensor(action_mean)
        self.action_std = torch.tensor(action_std)

        # 加载所有轨迹
        self.samples = []
        with open(f"{data_root}/trajectories.jsonl") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像
        image = Image.open(f"{self.data_root}/{sample['image']}").convert("RGB")

        # 文本指令
        instruction = sample["instruction"]

        # 动作（归一化）
        action = torch.tensor(sample["action"])
        action = (action - self.action_mean) / self.action_std

        # 使用 processor 预处理
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        # 去掉 batch 维度
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["actions"] = action

        return inputs
```

---

## 4.2 LoRA 高效微调

直接微调 7B 参数模型需要大量显存。LoRA（Low-Rank Adaptation）只训练少量低秩适配器：

```python
from peft import LoraConfig, get_peft_model

# 配置 LoRA
lora_config = LoraConfig(
    r=32,                          # 低秩维度
    lora_alpha=32,                 # 缩放因子
    target_modules=[
        "q_proj", "v_proj",         # 注意力层的投影矩阵
        "k_proj", "o_proj",
    ],
    lora_dropout=0.1,
    bias="none",
)

# 应用到模型
model = get_peft_model(base_model, lora_config)

# 只训练 LoRA 参数（约 0.1% 的总参数量）
model.print_trainable_parameters()
# 输出: trainable params: 7M || all params: 7B || trainable%: 0.1
```

### 训练循环

```python
from torch.optim import AdamW
from torch.utils.data import DataLoader

# 数据
 dataset = VLADataset("data/my_robot", processor, action_mean, action_std)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 优化器（只优化 LoRA 参数）
optimizer = AdamW(model.parameters(), lr=1e-4)

# 训练
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        # 前向
        actions_pred = model(
            input_ids=batch["input_ids"],
            pixel_values=batch["pixel_values"],
        )

        # MSE 损失
        loss = torch.nn.functional.mse_loss(actions_pred, batch["actions"])

        # 反向
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## 4.3 在 LIBERO 上微调

LIBERO 是轻量的仿真基准，适合快速实验。

### 环境安装

```bash
pip install libero
```

### 数据加载

```python
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
import numpy as np

# 加载 LIBERO 任务
benchmark = get_benchmark("libero_spatial")

# 获取演示数据
for task_id in range(len_benchmark(benchmark)):
    task = benchmark.get_task(task_id)
    demos = task.get_demos()  # 获取人类演示

    for demo in demos:
        for t in range(len(demo)):
            obs = demo[t]["obs"]
            image = obs["agentview_image"]
            action = demo[t]["actions"]
            instruction = task.language_instruction

            # 保存为训练样本
            save_sample(image, action, instruction)
```

### 微调脚本

```bash
python tutorials/04-fine-tuning/finetune_openvla.py \
    --pretrained_model openvla/openvla-7b \
    --dataset libero \
    --batch_size 4 \
    --num_steps 5000 \
    --learning_rate 1e-4 \
    --lora_r 32 \
    --output_dir checkpoints/my_vla
```

完整脚本见 `tutorials/04-fine-tuning/finetune_openvla.py`（待补充）。

---

## 4.4 关键超参数

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| Learning Rate | 1e-4 ~ 5e-4 | LoRA 可用较大学习率 |
| Batch Size | 4-8 | 受显存限制 |
| LoRA Rank (r) | 16-64 | 任务越复杂 rank 越大 |
| LoRA Alpha | 2*r | 通常设为 rank 的 2 倍 |
| Training Steps | 5k-20k | 预训练模型不需要太多步 |
| Action Chunk Size | 4-8 | 与预训练模型一致 |

### 学习率调度

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps,
)
```

---

## 4.5 评估

### 指标

- **成功率**：任务完成的百分比
- **平均回报**：累积奖励
- **动作 MSE**：预测动作与专家动作的均方误差

### 仿真评估

```python
import libero

env = libero.get_env(task)
successes = 0

for episode in range(num_eval_episodes):
    obs = env.reset()
    done = False

    while not done:
        # VLA 推理
        image = obs["agentview_image"]
        action = vla_model.predict(image, task.language_instruction)

        # 执行
        obs, reward, done, info = env.step(action)

    if info["success"]:
        successes += 1

print(f"Success rate: {successes / num_eval_episodes:.2%}")
```

---

## 4.6 调试技巧

### 损失不下降？

1. 检查数据归一化是否正确
2. 降低学习率
3. 检查 action 维度是否匹配
4. 确认 LoRA 配置正确（`print_trainable_parameters`）

### 过拟合？

1. 增加数据量
2. 增大 LoRA dropout
3. 降低 LoRA rank
4. 提前停止

### 仿真中表现差？

1. 检查 `unnorm_key` 是否匹配训练数据
2. 增加 Action Chunking 长度
3. 在训练数据中添加更多该任务的样本

---

## 验证检查点

- [ ] 能准备符合格式的 VLA 训练数据
- [ ] 能使用 LoRA 配置微调模型
- [ ] 能在 LIBERO 上完成训练循环
- [ ] 能在测试任务上评估并达到 50%+ 成功率

---

## 延伸阅读

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [OpenVLA 微调文档](https://github.com/openvla/openvla?tab=readme-ov-file#fine-tuning)
- [LIBERO 基准](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [HuggingFace PEFT 库](https://huggingface.co/docs/peft)
