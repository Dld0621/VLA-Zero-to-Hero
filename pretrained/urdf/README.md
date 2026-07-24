# URDF / MJCF 模型文件说明

本目录包含各开源灵巧手及机械臂的 URDF/MJCF 模型文件，用于 Retargeting 仿真和复现。

## 模型清单

| 模型 | 路径 | 文件类型 | 大小 | 来源 |
|------|------|---------|------|------|
| **LEAP Hand** | `leap_hand_sim/assets/leap_hand/` | URDF + STL | ~12 MB | [LEAP_Hand_Sim](https://github.com/leap-hand/LEAP_Hand_Sim) |
| **ORCA Hand v1/v2** | `orcahand_description/v1/`, `v2/` | URDF + MJCF | URDF 已包含，STL ~560MB | [orcahand_description](https://github.com/orcahand/orcahand_description) |
| **Shadow Hand** | `mujoco_menagerie/shadow_hand/` | MJCF + OBJ | ~4 MB | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) |
| **Allegro Hand** | `allegro_hand_right/`, `allegro_hand_left/` | URDF + OBJ | ~2 MB | GeoRT 项目 / [Allegro ROS](https://github.com/simlabrobotics/allegro_hand_ros_v4) |
| **Franka FR3** | `mujoco_menagerie/franka_fr3/` | MJCF + OBJ/STL | ~56 MB | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) |

## 使用说明

### LEAP Hand
```python
import mujoco
model = mujoco.MjModel.from_xml_path('leap_hand_sim/assets/leap_hand/robot.urdf')
```

### ORCA Hand
ORCA Hand 的 URDF 和 MJCF 文件已包含。由于 STL mesh 文件总计约 560MB（970 个文件），未纳入 Git。如需完整模型，请从原始仓库下载：
```bash
git clone https://github.com/orcahand/orcahand_description.git
```

### Shadow Hand
```python
import mujoco
model = mujoco.MjModel.from_xml_path('mujoco_menagerie/shadow_hand/scene_right.xml')
```

### Allegro Hand
```python
# 使用 Pinocchio 或 PyBullet 加载 URDF
import pinocchio as pin
model = pin.buildModelFromUrdf('allegro_hand_right/allegro_hand_right.urdf')
```

### Franka FR3
```python
import mujoco
model = mujoco.MjModel.from_xml_path('mujoco_menagerie/franka_fr3/scene.xml')
```

## 许可证

各模型遵循其原始仓库的许可证：
- LEAP Hand: MIT (代码) / CC BY-NC-SA (CAD)
- ORCA Hand: MIT
- Shadow Hand (MuJoCo Menagerie): Apache 2.0
- Allegro Hand: Apache 2.0 (ROS 驱动)
- Franka FR3 (MuJoCo Menagerie): Apache 2.0
