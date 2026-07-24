# Stage 2: Rule-based Retargeting

> 从人手 landmarks 到机器人关节角的直接映射方法，最简单实用的 retargeting 基线。

---

## 核心思想

将人手的关键点坐标直接转换为机器人关节角，通过：

1. **坐标系转换**：相机坐标 → 手腕局部坐标
2. **角度计算**：从 landmarks 计算弯曲角和外展角
3. **比例映射**：将人手角度按比例映射到机器人角度
4. **关节限位**：裁剪到机器人关节范围

---

## 完整 Pipeline

```python
# 1. 获取 landmarks（来自 MediaPipe）
landmarks = mediapipe_hands.detect(image)  # [21, 3]

# 2. 预处理（局部坐标 + 归一化 + 镜像）
landmarks_local = preprocess(landmarks, is_left=True)

# 3. 计算弯曲角
angles = compute_flexion_angles(landmarks_local)

# 4. 映射到机器人关节
robot_joints = angles * scale_factor

# 5. 限位裁剪
robot_joints = np.clip(robot_joints, joint_min, joint_max)

# 6. 发送到机器人
robot.set_joints(robot_joints)
```

---

## 运行示例

```bash
cd examples
python landmark_to_joint.py --hand right --gesture open
python landmark_to_joint.py --hand left --gesture fist
```

---

## 关键参数调优

| 参数 | 作用 | 经验值 |
|------|------|--------|
| `scale_factor` | 补偿 landmark→curl 衰减 | 1.60 |
| `normalization_denominator` | 归一化分母 | 0.95 |
| `ema_alpha` | 时域滤波系数 | 0.3 |

---

## 优缺点

**优点**：
- 实现简单，无需训练数据
- 实时性好（<1ms）
- 工程上最稳定

**缺点**：
- 泛化性差（换操作者/机器人需重新调参）
- 无法补偿尺寸差异
- 拇指校准一般
