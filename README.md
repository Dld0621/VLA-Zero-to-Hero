# VLA Zero to Hero · 从零入门视觉-语言-动作模型

> 帮助初学者系统学习 VLA（Vision-Language-Action）的入门开源项目。从 VLM 基础到端到端机器人策略，循序渐进，代码可运行。

[![Stars](https://img.shields.io/github/stars/Dld0621/VLA-Zero-to-Hero?style=flat-square)](https://github.com/Dld0621/VLA-Zero-to-Hero)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)

---

## 什么是 VLA？

**VLA（Vision-Language-Action）** 是具身智能领域的核心范式：模型同时接收**视觉图像**和**自然语言指令**，直接输出**机器人动作**，实现"看图 + 听话 → 执行"的端到端闭环。

```
视觉输入 (RGB图像) ──┐
                    ├──→ VLA 模型 ──→ 机器人动作 (关节角度/末端位姿/增量)
语言指令 ("拿起红杯子") ──┘
```

VLA 的本质是将大语言模型的语义理解能力与视觉感知融合，再映射到低维动作空间，是通往通用机器人智能的关键路径。

---

## 学习路径

本项目按 **4 个阶段** 组织，建议按顺序学习：

| 阶段 | 主题 | 目标 | 预计时间 |
|------|------|------|---------|
| **Stage 1** | VLM 基础 | 理解视觉-语言模型的编码器-解码器架构 | 1-2 天 |
| **Stage 2** | 动作表示 | 掌握机器人动作空间的各种表示方式 | 1 天 |
| **Stage 3** | 简单 VLA | 搭建一个最小的 VLA 推理 pipeline | 2-3 天 |
| **Stage 4** | 微调实践 | 在仿真环境或真实数据上微调 VLA 模型 | 3-5 天 |

---

## 仓库结构

```
VLA-Zero-to-Hero/
├── README.md                          # 本文件
├── docs/                              # 概念文档
│   ├── 01-what-is-vla.md             # VLA 核心概念详解
│   ├── 02-key-papers.md              # 关键论文导读（10篇）
│   ├── 03-learning-path.md           # 完整学习路线
│   └── 04-glossary.md                # 术语表
├── tutorials/                         # 分阶段教程
│   ├── 01-vlm-basics/                # Stage 1: VLM 基础
│   ├── 02-action-representation/     # Stage 2: 动作表示
│   ├── 03-simple-vla/                # Stage 3: 简单 VLA
│   └── 04-fine-tuning/               # Stage 4: 微调实践
├── examples/                          # 可运行示例代码
│   ├── minimal_vla.py                # 最小 VLA 推理示例
│   └── inference_demo.py             # 完整推理 demo
├── setup/                             # 环境配置
│   └── environment.yml               # Conda 环境文件
└── resources/                         # 资源汇总
    └── README.md                     # 数据集、模型、工具链接
```

---

## 快速开始

### 环境准备

```bash
# 克隆仓库
git clone https://github.com/Dld0621/VLA-Zero-to-Hero.git
cd VLA-Zero-to-Hero

# 创建 conda 环境
conda env create -f setup/environment.yml
conda activate vla-zero

# 验证安装
python -c "import torch; print(torch.__version__)"
```

### 运行最小示例

```bash
# 一个最简化的 VLA 推理示例（使用占位模型）
python examples/minimal_vla.py \
    --image_path data/example_scene.jpg \
    --instruction "pick up the red cup"
```

---

## 前置知识

学习 VLA 之前，建议具备以下基础：

- **Python** 编程（NumPy、PyTorch 基础）
- **深度学习** 基础（CNN、Transformer、注意力机制）
- **多模态学习** 初步了解（CLIP 等视觉-语言对齐方法）
- **机器人学** 基础（正运动学、关节空间 vs 任务空间）

如不具备，本项目 `tutorials/01-vlm-basics/` 提供补充材料。

---

## 关键论文速览

| 论文 | 年份 | 贡献 |
|------|------|------|
| RT-1 / RT-2 | 2022/2023 | Google DeepMind 的机器人 Transformer 系列，VLA 的先驱 |
| OpenVLA | 2024 | 7B 参数开源 VLA 模型，DINOv2 + SigLIP + Llama 2 |
| π0 (pi-zero) | 2024 | Physical Intelligence 的流匹配 VLA，强调精细操作 |
| Octo | 2024 | 开源通用机器人策略，支持多机器人、多模态输入 |
| SPOC | 2024 | 结构化策略与开放词汇命令，提升泛化能力 |

完整论文导读见 [`docs/02-key-papers.md`](docs/02-key-papers.md)。

---

## 核心概念一句话总结

- **VLM（Vision-Language Model）**：看图 + 读文字 → 输出文字描述
- **VLA（Vision-Language-Action）**：看图 + 读文字 → 输出机器人动作
- **Action Chunking**：一次预测未来多步动作，减少推理延迟
- **Diffusion Policy**：用扩散模型生成连续动作分布
- **End-to-End**：从像素到动作的单一模型，无需显式中间表示

完整术语表见 [`docs/04-glossary.md`](docs/04-glossary.md)。

---

## 推荐数据集与模型

| 名称 | 类型 | 说明 |
|------|------|------|
| Open X-Embodiment | 数据集 |  largest 开源机器人数据集，22+ 机器人平台 |
| LIBERO | 数据集 | 单臂操作基准，适合快速实验 |
| OpenVLA-7B | 模型 | 开源 VLA，可直接推理或微调 |
| Octo-Base | 模型 | 开源通用策略，支持灵活输入 |

更多资源见 [`resources/README.md`](resources/README.md)。

---

## 贡献指南

欢迎提交 Issue 和 PR！你可以：

- 补充更多论文解读
- 添加新的代码示例
- 修正文档错误
- 分享学习笔记

---

## 许可证

[MIT License](LICENSE)

---

## Acknowledgments

本项目参考了以下优秀开源项目的设计思路：

- [OpenVLA](https://github.com/openvla/openvla) — 开源 VLA 模型
- [Octo](https://github.com/octo-models/octo) — 开源通用机器人策略
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) — 扩散策略
