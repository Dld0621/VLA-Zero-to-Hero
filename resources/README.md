# 资源汇总

> 数据集、预训练模型、仿真环境、工具库的一站式索引。

---

## 数据集

### 机器人操作数据集

| 名称 | 规模 | 机器人 | 语言标注 | 链接 |
|------|------|--------|---------|------|
| **Open X-Embodiment** | 1M+ 轨迹 | 22+ 平台 | 有 | [GitHub](https://github.com/google-deepmind/open_x_embodiment) |
| **Bridge Data V2** | 60k 轨迹 | WidowX | 有 | [Site](https://rail.eecs.berkeley.edu/datasets/bridge_release/) |
| **LIBERO** | 10k 轨迹 | Franka | 有 | [GitHub](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| **RH20T** | 110k 轨迹 | 多种 | 有 | [GitHub](https://github.com/rh20t/rh20t) |
| **RT-1 Dataset** | 130k 轨迹 | RT-1 机器人 | 有 | 部分通过 OXE 提供 |
| **Aloha** | 双手操作 | ALOHA | 无（纯演示） | [GitHub](https://github.com/tonyzhaozh/aloha) |

### 仿真基准

| 名称 | 引擎 | 特点 | 链接 |
|------|------|------|------|
| **LIBERO** | Bullet | 130 语言条件任务，轻量 | [GitHub](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| **MetaWorld** | MuJoCo | 50 操作任务，VLA 常用 | [GitHub](https://github.com/Farama-Foundation/Metaworld) |
| **Robosuite** | MuJoCo | 丰富传感器，可扩展 | [GitHub](https://github.com/ARISE-Initiative/robosuite) |
| **SimplerEnv** | 多种 | 真实感渲染，评测 VLA | [GitHub](https://github.com/simpler-conversation/SimplerEnv) |
| **ManiSkill** | SAPIEN | GPU 并行仿真，大规模训练 | [GitHub](https://github.com/haosulab/ManiSkill) |

---

## 预训练模型

### VLA 模型

| 模型 | 参数量 | 架构 | 代码 | 特点 |
|------|--------|------|------|------|
| **OpenVLA-7B** | 7B | DINOv2+SigLIP / Llama 2 | [GitHub](https://github.com/openvla/openvla) | 最活跃开源 VLA |
| **Octo-Base** | 27M | Transformer | [GitHub](https://github.com/octo-models/octo) | 轻量通用策略 |
| **Octo-Small** | 93M | Transformer | [GitHub](https://github.com/octo-models/octo) | 中等规模 |
| **π0** | 3B | PaliGemma + Flow Matching | [GitHub](https://github.com/physical-intelligence/pi0) | 精细操作 |
| **RT-1** | 35M | EfficientNet + Transformer | [GitHub](https://github.com/google-research/robotics_transformer) | 开山之作 |
| **RDT-1B** | 1B | DiT | [GitHub](https://github.com/thu-ml/RDT) | 清华开源，中文友好 |

### 视觉编码器

| 模型 | 特点 | 链接 |
|------|------|------|
| **CLIP ViT** | 语言对齐，通用 | [HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32) |
| **DINOv2** | 自监督，空间理解强 | [HuggingFace](https://huggingface.co/facebook/dinov2-base) |
| **SigLIP** | 零样本能力强 | [HuggingFace](https://huggingface.co/google/siglip-base-patch16-224) |
| **SAM** | 分割级理解 | [GitHub](https://github.com/facebookresearch/segment-anything) |

### 语言模型

| 模型 | 参数量 | 特点 | 链接 |
|------|--------|------|------|
| **Llama 2** | 7B-70B | 开源，推理强 | [HuggingFace](https://huggingface.co/meta-llama) |
| **Phi-3** | 3.8B | 小参数，端侧 | [HuggingFace](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |
| **Qwen2** | 0.5B-72B | 中文优化 | [HuggingFace](https://huggingface.co/Qwen) |

---

## 工具库

| 名称 | 用途 | 链接 |
|------|------|------|
| **Transformers** | 加载预训练模型 | [HuggingFace](https://huggingface.co/docs/transformers) |
| **Accelerate** | 分布式训练/推理 | [HuggingFace](https://huggingface.co/docs/accelerate) |
| **Datasets** | 数据集处理 | [HuggingFace](https://huggingface.co/docs/datasets) |
| **einops** | 张量操作 | [GitHub](https://github.com/arogozhnikov/einops) |
| **timm** | 视觉模型库 | [GitHub](https://github.com/huggingface/pytorch-image-models) |
| **Diffusers** | 扩散模型 | [HuggingFace](https://huggingface.co/docs/diffusers) |
| **WandB** | 实验追踪 | [Site](https://wandb.ai) |
| **Hydra** | 配置管理 | [GitHub](https://github.com/facebookresearch/hydra) |

---

## 在线课程与教程

| 资源 | 类型 | 链接 |
|------|------|------|
| Stanford CS25 | VLM 研讨课 | [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rM6GirPQv-_4) |
| Berkeley Deep RL | 强化学习基础 | [Site](http://rail.eecs.berkeley.edu/deeprlcourse/) |
| Embodied AI Workshop | 年度研讨会 | [Site](https://embodied-ai.org/) |
| OpenVLA Docs | 官方文档 | [Site](https://openvla.github.io/) |
| Diffusion Policy Tutorial | 扩散策略 | [Site](https://diffusion-policy.cs.columbia.edu/) |

---

## 社区与讨论

| 平台 | 说明 |
|------|------|
| [r/MachineLearning](https://reddit.com/r/MachineLearning) | Reddit ML 社区，常有 VLA 论文讨论 |
| [Twitter/X #VLA](https://twitter.com/search?q=%23VLA) | 最新论文发布和讨论 |
| [HuggingFace Forums](https://discuss.huggingface.co/) | 模型使用问题 |
| [Open X-Embodiment Discord](https://discord.gg/openx) | 数据集和模型讨论 |

---

## 持续更新

欢迎提交 PR 补充新资源！请按以下格式：

```markdown
| **名称** | 一句话描述 | [链接](URL) |
```
