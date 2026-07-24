# Pretrained Models & URDF Files

> 本目录存放 AnyTeleop 和 HaMeR 复现所需的全部预训练模型文件和机器人 URDF 模型。

## 目录结构

```
pretrained/
├── README.md                         # 本文件
├── hamer/                            # HaMeR (CVPR 2024) 模型文件
│   ├── hamer_demo_data.tar.gz        # HaMeR 主模型权重包（~700MB）
│   └── model_final_f05665.pkl        # ViTDet-H 人体检测器权重（~2.6GB）
├── anyteleop/                        # AnyTeleop (RSS 2023) 依赖
│   └── frankmocap/                   # FrankMocap 手部姿态估计
│       └── extra_data/hand_module/
│           ├── SMPLX_HAND_INFO.pkl   # SMPL-X 手部元数据（~0.1MB）
│           ├── mean_mano_params.pkl  # MANO 均值参数（~0.5KB）
│           └── pretrained_weights/
│               └── pose_shape_best.pth  # 手部姿态回归权重（~102MB）
├── urdf/                             # 机器人 URDF/MJCF 模型
│   └── mujoco_menagerie/
│       ├── shadow_hand/              # Shadow Hand 20-DOF
│       ├── allegro_hand/             # Allegro Hand 16-DOF
│       └── franka_fr3/               # Franka FR3 7-DOF 机械臂
└── mano/                             # ⚠️ 需手动注册下载
    └── (MANO_RIGHT.pkl 待放置)       # https://mano.is.tue.mpg.de/
```

## HaMeR 使用方式

```bash
# 1. 解压主模型权重包到 HaMeR 项目根目录
mkdir -p /path/to/hamer/_DATA
tar -xzf pretrained/hamer/hamer_demo_data.tar.gz -C /path/to/hamer/_DATA/

# 2. 放置 ViTDet 权重（Detectron2 通常自动下载，也可手动指定）
cp pretrained/hamer/model_final_f05665.pkl /path/to/hamer/detectron2/

# 3. 放置 MANO 模型（需先注册下载）
cp pretrained/mano/MANO_RIGHT.pkl /path/to/hamer/_DATA/data/mano/MANO_RIGHT.pkl
```

## AnyTeleop / FrankMocap 使用方式

```bash
# 将 extra_data 目录复制到 FrankMocap 项目根目录
cp -r pretrained/anyteleop/frankmocap/extra_data /path/to/frankmocap/extra_data

# 还需手动下载（需注册）：
# SMPLX_NEUTRAL.pkl → https://smpl-x.is.tue.mpg.de/
# 放置到 extra_data/smpl/SMPLX_NEUTRAL.pkl
```

## URDF 使用方式

```bash
# 复制到你的项目
cp -r pretrained/urdf/mujoco_menagerie/shadow_hand /your/project/assets/
cp -r pretrained/urdf/mujoco_menagerie/allegro_hand /your/project/assets/
cp -r pretrained/urdf/mujoco_menagerie/franka_fr3 /your/project/assets/
```

## ⚠️ 需手动注册下载的文件

| 文件 | 下载地址 | 许可证 | 放置路径 |
|------|---------|--------|---------|
| `MANO_RIGHT.pkl` | [https://mano.is.tue.mpg.de/](https://mano.is.tue.mpg.de/) | 非商业科研 | `pretrained/mano/MANO_RIGHT.pkl` |
| `SMPLX_NEUTRAL.pkl` | [https://smpl-x.is.tue.mpg.de/](https://smpl-x.is.tue.mpg.de/) | 非商业科研 | `pretrained/anyteleop/frankmocap/extra_data/smpl/SMPLX_NEUTRAL.pkl` |

> 上述文件需要注册账号并同意许可证后才能下载，无法通过脚本自动获取。

## 文件来源

| 文件 | 来源 URL | 许可证 |
|------|---------|--------|
| `hamer_demo_data.tar.gz` | [UT Austin](https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz) | HaMeR 项目 |
| `model_final_f05665.pkl` | [Detectron2 Model Zoo](https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl) | Apache 2.0 |
| `SMPLX_HAND_INFO.pkl` | [Facebook Research](https://dl.fbaipublicfiles.com/eft/fairmocap_data/hand_module/SMPLX_HAND_INFO.pkl) | MIT |
| `mean_mano_params.pkl` | [Facebook Research](https://dl.fbaipublicfiles.com/eft/fairmocap_data/hand_module/mean_mano_params.pkl) | MIT |
| `pose_shape_best.pth` | [Facebook Research](https://dl.fbaipublicfiles.com/eft/fairmocap_data/hand_module/checkpoints_best/pose_shape_best.pth) | MIT |
| URDF 模型 | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Apache 2.0 |
| FrankMocap 源码 | [Facebook Research](https://github.com/facebookresearch/frankmocap) | MIT |