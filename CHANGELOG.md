# Changelog

> 所有值得注意的变更都将记录在此文件中。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased]

### Fixed
- 修正 DIAMOND 官方仓库链接：`ethz-rl/diamond` → `eloialonso/diamond`（作者 Vincent Micheli 的个人仓库）
- 修正 IRIS 官方仓库链接：`janner/iris` → `eloialonso/iris`
- 修复 `sim_closed_loop_demo.py` 中 `ScriptedPolicy._get_gripper_pos` 使用未定义 `self.gripper_id` 的隐患
- 修复 `freshman_zero_to_one.py` 中 `HumanHandVisualizer.__init__` 的 `self.ay` 笔误
- 修复 `world_model_vla_pipeline.py` 中 `fusion_1_data_generator` 注释与实现不符的问题，并增加 WM reward 预测作为增强信号
- 修复 `world_model_vla_pipeline.py` 中 `fusion_4_wam` 对 `_cached_data` 的隐式依赖，增加前置检查与友好错误提示

### Added
- 新增 `CONTRIBUTING.md`：完整的贡献指南，包含 Issue/PR 规范、内容质量标准、审查清单
- 新增 `CHANGELOG.md`：版本变更记录
- 新增 `requirements.txt`：标准 Pip 依赖文件（与 `setup/environment.yml` 同步）
- 新增 `tests/test_imports.py`：基础导入测试，确保核心模块无语法错误

### Changed
- README.md 全面重构：新增视觉框架图、四大支柱详解、30 秒快速开始、适合人群推荐表
- 合并 Dexterous-Retargeting-Guide 仓库内容，统一为 Embodied AI Zero to Zero

---

## [2026.07.24]

### Added
- 完成 Dexterous-Retargeting-Guide 与 VLA-Zero-to-Hero 仓库融合
- 新增 27 篇核心文档，覆盖重定向、VLA、RL、世界模型四大支柱
- 新增 18+ 个可运行示例代码
- 新增 10 阶段教程（双轨道：重定向轨道 + VLA 轨道）
- 新增 `docs/18-frontier-papers-online.md`：20+ 篇前沿论文在线链接
- 新增 `docs/17-research-trends-and-positioning.md`：2026 研究趋势分析（六大研究转向）
- 新增 `docs/16-arxiv-retargeting-scan.md`：80+ 篇 Arxiv 重定向论文扫描

### Changed
- 统一环境依赖为 `embodied-ai` Conda 环境
- 重写 README.md 以反映四大支柱内容结构
- 更新 LICENSE 版权持有者为 Embodied AI Zero to Zero Contributors

---

## [2026.07.20]

### Added
- 新增 `examples/freshman_zero_to_one.py`：大一新生零外部依赖的完整重定向 pipeline
- 新增 `examples/dexmv_style_retargeting/`：DexMV SLSQP + Huber Loss 高精度实现
- 新增 `docs/11-dexmv-research-guide.md`：DexMV 论文深度解读
- 新增 `docs/12-freshman-zero-to-one.md`：从零开始的重定向实战指南

---

## [2026.07.15]

### Added
- 新增 VLA 内容模块：`docs/01-what-is-vla.md`、`docs/02-key-papers.md`、`docs/03-learning-path.md`
- 新增 `examples/minimal_vla.py`：最小可运行 VLA 架构演示
- 新增 `examples/vla_demo.py`：OpenVLA / SmolVLA 推理演示
- 新增 `tutorials/03-simple-vla/`：从零搭建 VLA 教程
- 新增 `tutorials/04-fine-tuning/`：LIBERO 微调完整代码

---

## [2026.07.10]

### Added
- 新增 RL 与世界模型模块
- 新增 `examples/rl_demo.py`：Q-Learning / SAC + HER 演示
- 新增 `examples/world_model_demo.py`：线性世界模型 + MPC 规划
- 新增 `examples/dreamer_rssm.py`：DreamerV3 RSSM 完整实现
- 新增 `docs/06-rl-fundamentals-for-vla.md`：面向 VLA 学习者的 RL 基础
- 新增 `docs/07-world-models-for-vla.md`：面向 VLA 学习者的世界模型指南

---

## [2026.07.01]

### Added
- 项目初始化：Embodied AI Zero to Zero
- 基础文档：关节概念、重定向概念、方法分类、评估指标
- 核心示例：`fk_ik_demo.py`、`landmark_to_joint.py`、`minimal_retargeting.py`
- 资源索引：`resources/README.md`、`setup/environment.yml`
