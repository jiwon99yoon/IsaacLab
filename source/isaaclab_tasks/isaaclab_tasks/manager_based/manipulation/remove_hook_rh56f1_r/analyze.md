# Remove Hook Task 환경 분석 (Isaac-RemoveHook-Hdr35-RH56F1-v0)
## 강화학습 설정 및 실제 데이터 매핑 분석

---

## 1. 환경 개요

### 1.1 Task Description
**목표**: 자동차 샤시 조립 라인에서 와이어 후크를 스프링에서 제거하고 목표 위치로 이동

- **Robot**: HDR35_20 (35kg payload, 6-DoF arm) + RH56F1_R (Inspire Hand Right, 6-DoF under-actuated)
- **Object**: 자동차 프론트 샤시 조립품 (Chassis + Springs + Wire hooks)
- **Goal**: **왼쪽 후크 (left_ring)**를 잡아 스프링에서 분리 → 왼쪽 0.2m, 위쪽 0.3m로 이동
  - NOTE: RH56F1_R (오른손)은 left_ring을, DG5F_L (왼손)은 right_ring을 타겟
- **Environment**: 16개 병렬 시뮬레이션 (향후 256개로 확장 가능)

### 1.2 로봇 구성

| 구성 요소 | 사양 | DoF | 비고 |
|-----------|------|-----|------|
| **Arm** | HDR35_20 (Hyundai Robotics) | 6 | 35kg payload, industrial arm |
| **Hand** | RH56F1_R (Inspire Hand Right) | 6 | Under-actuated, 5 fingers |
| **Total DoF** | - | **12** | 6 (arm) + 6 (hand) |
| **Object** | Wire with hooks (Articulation) | 2 joints | left_rjoint, right_rjoint (PhysicsRevoluteJoint) |
| **Chassis** | Static assembly (AssetBase) | 0 | Springs + Frames |

### 1.3 Task의 특수성

**DexSuite와의 차이점**:
- **Object**: 단순한 큐브/구 → **복잡한 와이어/후크** (Articulation with left_rjoint, right_rjoint)
- **Task**: Lift & Reorient → **Remove hook from spring** (더 정밀한 조작 필요)
- **Contact**: 물체 전체 → **얇은 와이어 후크** (접촉 면적 매우 작음)
- **Environment**: Table → **Fixed chassis assembly** (공중에 고정)

### 1.4 실제 실험 환경 (Real-World Setup)

**로봇 시스템** (2대 구성):
| Robot | Arm | F/T Sensor | Gripper | Tactile Sensor | 타겟 후크 |
|-------|-----|------------|---------|----------------|-----------|
| **오른손 (현재 사용)** | HDR35-20 | ATI NET Axia80-M50 | RH56F1 (Inspire, 6-DoF) | **Tashan 정전용량식** (팁+손바닥) | **left_ring** |
| 왼손 (향후) | HDR35-20 | ATI NET Axia80-M50 | DG5F (Tesollo, 20-DoF) | Aidin 초소형 토크 센서 (팁) | right_ring |

**Vision 시스템**:
- **Zivid2 M70**: 고성능 물체 인식용 (Isaac Sim 연동 예정)
- Intel RealSense 455: 저성능 동작학습용 (2대)

**실험 세팅 특징**:
- 샤시모듈은 **고정**되어 있음 (strut, spring, 샤시 전체 고정)
- Wire는 걸려있는 상태로 시작 (로봇이 접촉하지 않으면 고정 상태 유지)
- ⚠️ **Wire**: 실제는 deformable object이나, **시뮬레이션에선 rigid body로 설정**

**현재 상태**:
- ✅ Hardware 준비 완료 (로봇, 센서, 카메라)
- ⏳ ATI F/T sensor USD file 미완성 (시뮬레이션 미반영)
- ⏳ Zivid camera 연동 대기 (일단 GT point cloud 사용)
- 🎯 **현재 목표**: 시뮬레이션에서 먼저 학습 성공 검증

---

## 2. Observation Space 상세

### 2.1 전체 구조

우리 환경은 **3개 그룹**으로 구성 (DexPoint 구조 참고):

```
Total Observation Dimension: 1640
- Policy Group (95):      목표 정보 (hook pose + actions history)
- Proprio Group (585):    로봇 상태 (joints, contacts, fingertips)
- Perception Group (960): 물체 인식 (hook/wire point cloud)

Policy Group 계산: hook_pos(15) + hook_quat(20) + actions(60) = 95
```

**History Length**: 모든 observation은 **5 step history** (시간적 맥락 제공)

### 2.2 Group 1: Policy (95 dim) - 목표 정보

| Observation | Dimension | Description | 시뮬레이션 데이터 | 실제 데이터 획득 방법 |
|-------------|-----------|-------------|-------------------|----------------------|
| `hook_pos_b` | 15 (3 × 5 history) | Hook position in robot frame | `object.data.body_pos_w` (left_ring) | **Zivid camera** RGB-D → 3D position estimation |
| `hook_quat_b` | 20 (4 × 5 history) | Hook orientation (quaternion) | `object.data.body_quat_w` | **Zivid camera** point cloud → PCA or marker-based orientation |
| `actions` | 60 (12 × 5 history) | Previous actions (temporal consistency) | `env.action_manager.action` | Robot controller에서 직접 기록 (sent commands) |

**Note**:
- `actions`는 `history_length=5`로 최근 5개 step의 action 포함 (t, t-1, t-2, t-3, t-4)
- ~~`last_action` 별도 term 없음~~ (analyze.md 이전 버전 오류 수정)
- Hook position/orientation은 **vision system 의존** (향후)
  - 현재 시뮬레이션: Isaac Sim에서 **Ground Truth** position 직접 획득 (`body_pos_w`)
  - 향후 실제 환경: Zivid camera RGB-D → 3D pose estimation
- 실제 환경에서는 **calibration 필수** (camera ↔ robot base frame)

### 2.3 Group 2: Proprio (585 dim) - 로봇 상태

| Observation | Dimension | Description | 시뮬레이션 데이터 | 실제 데이터 획득 방법 |
|-------------|-----------|-------------|-------------------|----------------------|
| `joint_pos` | 60 (12 × 5 history) | Joint positions (6 arm + 6 hand) | `robot.data.joint_pos` | **Robot encoders** (absolute or incremental) |
| `joint_vel` | 60 (12 × 5 history) | Joint velocities | `robot.data.joint_vel` | Encoder differentiation or velocity sensors |
| `hand_tips_state_b` | 390 (78 × 5 history) | Fingertip states (6 bodies × 13) | `robot.data.body_pos_w` + `body_vel_w` | **Forward kinematics** from joint encoders |
| └ 구성 | 13 per body | 3 pos + 4 quat + 3 vel_lin + 3 vel_ang | PhysX rigid body state | FK로 계산 (URDF/USD model 기반) |
| └ bodies | 6 bodies | 1 palm (gripper_base_link) + 5 fingertips (.*_tip) | Body tracking | - |
| `contact` | 75 (15 × 5 history) | Contact forces (5 fingers × 3D) | `ContactSensor.data.force_matrix_w` | ✅ **Tashan tactile sensor** (아래 참고) |

**Contact Sensor 필터 설정**:
```python
ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/" + link_name,
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Wire/left_ring*"],  # left_ring만 감지!
)
# 원래: ["{ENV_REGEX_NS}/Wire"] → Wire 전체와의 접촉 감지
# 변경: ["{ENV_REGEX_NS}/Wire/left_ring*"] → left_ring 고리만 감지
```

| 접촉 대상 | Contact 감지 |
|-----------|-------------|
| **left_ring (타겟 고리)** | ✅ Yes |
| right_ring | ❌ No (필터링) |
| wire body | ❌ No (필터링) |
| 샤시/스프링 | ❌ No (필터링) |

**Contact Force 획득 방법 (실제 로봇)**:

✅ **우리 환경 (HDR35_20 + RH56F1_R)**:
- **Tashan 정전용량식 촉각 센서**: RH56F1 팁 및 손바닥에 부착
- 개별 손가락 접촉력 측정 가능 → **시뮬레이션과 직접 대응 가능!**
- 5개 fingertip sensors → 각각 3D force vector (15 dim)
- → **Sim-to-Real Gap 없음**: 실제 센서 데이터를 직접 사용 가능
- **NOTE**: RH56F1_R (오른손)은 **left_ring**을 타겟으로 함

⏳ **ATI F/T Sensor (향후 추가 예정)**:
- ATI NET Axia80-M50 (손목 장착)
- 손목 전체 힘/토크 측정 (6D wrench)
- 현재 USD file 미완성 → 시뮬레이션에 아직 미반영
- → 향후 observation에 추가 가능 (전역 grasp force monitoring)

**Clipping**:
- `hand_tips_state_b`: `clip=(-2.0, 2.0)` (robot base frame 기준 ±2m)
- `contact`: `clip=(-20.0, 20.0)` N (Inspire Hand effort 고려)

### 2.4 Group 3: Perception (960 dim) - 물체 인식

| Observation | Dimension | Description | 시뮬레이션 데이터 | 실제 데이터 획득 방법 |
|-------------|-----------|-------------|-------------------|----------------------|
| `object_point_cloud` | 960 (192 × 5 history) | **left_ring** point cloud (64 points × 3D) | Isaac Sim mesh sampling | ⏳ **Zivid camera** (연동 예정) |

**버그 수정 (2026-01-07)**:
```python
# 기존 (버그): Wire 전체에서 샘플링 + root_pos_w로 transform → 점이 mesh 위로 뜸
params={"num_points": 64, "flatten": True}

# 수정: left_ring만 샘플링 + body_pos_w로 transform → 정확한 위치
params={
    "num_points": 64,
    "flatten": True,
    "object_cfg": SceneEntityCfg("object", body_names=["left_ring"]),
}
```

**문제 원인**:
1. Wire는 Articulation이라 root와 body(left_ring) 위치가 다름
2. USD 내부 구조에서 mesh offset이 존재
3. 전체 Wire에서 64개 점 → 타겟(left_ring)에 집중 안 됨

**수정 효과**:
- ✅ Point cloud가 left_ring mesh에 정확히 붙음
- ✅ 타겟(잡아야 할 고리)에 집중된 observation
- ✅ 불필요한 정보(wire body, right_ring) 제거

**Point Cloud 획득 방법 (실제 로봇)**:

✅ **우리 환경 (Zivid2 M70)**:
- **Zivid2 M70**: Isaac Sim 연동 가능한 사양으로 선정
- RGB-D 이미지 캡처 → point cloud 생성
- 해상도: ~1M points (full scene)
- → **64 points로 downsampling 필요**

⏳ **현재 상태 (시뮬레이션 단계)**:
- Zivid camera는 구매했지만 **아직 연동 안됨**
- 일단 시뮬레이션에서 **perfect point cloud 사용** (ground-truth)
- → Sim-to-real transfer 시 vision system 완성 필요

**Processing Pipeline (향후 실제 로봇)**:
```
Zivid capture → Full point cloud → Segmentation (left_ring 영역만)
→ Downsampling (64 points) → Transform to robot frame
```

**Segmentation 방법 (향후 구현)**:
- **Color-based**: Hook는 특정 색상 (금속 hook)
- **Learning-based**: Mask R-CNN 등으로 hook detection
- **Geometric**: RANSAC으로 평면(chassis) 제거 → hook 영역 분리

**Downsampling (향후 구현)**:
- Farthest Point Sampling (FPS) - DexPoint 방식
- Random sampling
- Voxel grid downsampling

**Note**:
- Point cloud는 **robot base frame 기준**으로 변환 필요 (camera calibration 완료 필요)
- `clip=(-5.0, 5.0)` (wire explosion 방지용 강한 clipping)
- **현재는 시뮬레이션에서 잘 작동하는지 먼저 검증 중**

### 2.5 Observation Processing Pipeline

```
Raw Data → Clipping → Noise Addition → History Stacking (×5) → Concatenation (1660 dim)
→ Running Mean/Std Normalization → Policy/Critic Network
```

**Noise Addition** (Domain Randomization):
- Training 시에만 적용: `Unoise(n_min=-0.0, n_max=0.0)` (현재 비활성화)
- 향후 sim-to-real transfer 시 활성화 가능

---

## 3. Action Space

### 3.1 Action Configuration

| 항목 | 설정 | 비고 |
|------|------|------|
| **Type** | `RelativeJointPositionAction` | 상대적 위치 제어 (현재 위치 + action) |
| **Dimension** | 12 | 6 (arm) + 6 (hand) |
| **Scale** | 0.1 | action 범위 = [-0.1, 0.1] rad per step |
| **Control Frequency** | 60 Hz | dt=1/120s (sim), decimation=2 |
| **Clip** | 100.0 | RL-Games config (과도한 action 방지) |

### 3.2 Joint Mapping

**Arm (HDR35_20, 6-DoF)**:
```python
j1: Base rotation        # [-180°, 180°]
j2: Shoulder pitch       # [-135°, 90°]
j3: Shoulder roll        # [-80°, 180°]
j4: Elbow               # [-360°, 360°]
j5: Wrist pitch         # [-125°, 125°]
j6: Wrist roll          # [-360°, 360°]
```

**Hand (RH56F1_R Inspire, 6-DoF under-actuated)**:
```python
# Thumb: 2 joints (spread + bend)
right_thumb_1_joint     # Spread (abduction/adduction)
right_thumb_2_joint     # Bend (flexion)

# Other 4 fingers: 1 joint each (bend only)
right_index_1_joint     # Index bend
right_middle_1_joint    # Middle bend
right_ring_1_joint      # Ring bend
right_little_1_joint    # Little bend
```

**Under-actuated 특성**:
- 각 손가락은 **1개 actuator**만 제어
- 나머지 joint는 **mimic joint** (기계적 coupling)
- → Adaptive grasping (물체 형상에 자동으로 맞춤)

### 3.3 실제 로봇 제어

**시뮬레이션 → 실제**:
```
Policy output (action) → Scale (×0.1) → Add to current joint_pos
→ Send to robot controller (position command)
```

**실제 로봇 controller**:
- HDR35_20: Industrial robot controller (position or velocity mode)
- Inspire Hand: Custom controller (underactuated mechanism)
- 60 Hz control loop (시뮬레이션과 동일)

---

## 4. Reward Structure

### 4.1 Reward Components

| Reward Term | Weight | Type | Description | Formula |
|-------------|--------|------|-------------|---------|
| **Phase 1: Reaching Hook** |
| `fingers_to_hook` | 2.0 | Dense | 손가락이 후크에 접근 | `1 - tanh(d_fingertips / 0.1)` |
| **Phase 2: Grasping Hook** |
| `good_finger_contact` | 3.0 | Sparse | 손가락 접촉 (엄지 + 1개 이상) | `I(thumb > 0.5N) & I(other > 0.5N)` |
| **Phase 3: Transport to Target** |
| `hook_to_target` | 10.0 | Dense | 후크를 목표 위치로 이동 | `1 - tanh(d_hook_target / 0.02)` |
| **Penalties** |
| `action_l2` | -0.005 | Penalty | Action magnitude | `-‖a‖²` (clamped) |
| `action_rate_l2` | -0.01 | Penalty | Action smoothness | `-‖a_t - a_{t-1}‖²` (clamped) |
| `early_termination` | -1.0 | Penalty | 비정상 종료 (velocity explosion) | `-I(abnormal_state)` |

**Total Reward**:
```
R_total = 2.0×fingers_to_hook + 3.0×good_finger_contact + 10.0×hook_to_target
          - 0.005×action_l2 - 0.01×action_rate_l2 - 1.0×early_termination
```

### 4.2 Reward 상세 설명

#### 4.2.1 Phase 1: Reaching Hook (`fingers_to_hook`)

**목적**: 손가락을 후크에 접근시키기

**사용되는 데이터**:
| 항목 | 코드 | 설명 |
|------|------|------|
| **Robot 측** | `body_pos_w[:, asset_cfg.body_ids]` | `gripper_base_link` + `*_tip` (palm + 5 fingertips) |
| **Hook 측** | `body_pos_w[:, object_cfg.body_ids]` | **left_ring body position** (Ground Truth from sim) |

**구현**:
```python
# 손가락(palm + 5 fingertips)과 left_ring hook 간 최대 거리
asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]  # (num_envs, 6, 3)
object_pos = object.data.body_pos_w[:, object_cfg.body_ids]  # (num_envs, 1, 3) → left_ring
d_fingertips = torch.norm(asset_pos - object_pos, dim=-1).max(dim=-1).values
reward = 1 - tanh(d_fingertips / 0.1)  # std=0.1m
```

**버그 수정 (2026-01-07)**:
- 기존: `object.data.root_pos_w` → Wire articulation root (left_ring이 아님!)
- 수정: `object.data.body_pos_w[:, body_ids]` → **left_ring body position**

**특성**:
- `d=0m` → `reward=1.0` (완전 접촉)
- `d=0.1m` → `reward≈0.24` (std 거리)
- `d=0.5m` → `reward≈0.003` (멀리 떨어짐)

**실제 데이터**:
- `fingertips_pos`: Forward kinematics (joint encoders)
- `hook_pos`: 현재 시뮬레이션에선 **Ground Truth** (향후 Zivid camera vision)

#### 4.2.2 Phase 2: Grasping Hook (`good_finger_contact`)

**목적**: 안정적인 파지 (엄지 opposition + 다른 손가락 1개)

**Contact Sensor 필터**:
- `filter_prim_paths_expr=["{ENV_REGEX_NS}/Wire/left_ring*"]`
- **left_ring 고리에 닿았을 때만** contact force 측정됨
- 샤시, 스프링, wire body 등에 닿아도 contact = 0

**구현**:
```python
# Threshold: 0.5N (얇은 wire 고려, DexSuite의 1.0N보다 낮음)
# NOTE: Contact는 left_ring에 닿았을 때만 발생!
good_contact = (thumb_force > 0.5N) & (
    (index_force > 0.5N) | (middle_force > 0.5N) |
    (ring_force > 0.5N) | (little_force > 0.5N)
)
reward = good_contact.float()  # Binary: 0 or 1
```

**Under-actuated Hand 고려**:
- DG5F (fully-actuated): 엄지 + 중지 필수 (엄격)
- **Inspire (under-actuated)**: 엄지 + 아무거나 (완화)
- → Adaptive grasping 특성 활용

**실제 데이터**:
- ✅ **Tashan tactile sensor**: 5개 fingertip에 장착 → 개별 접촉력 측정 가능
- **Sim-to-Real Gap 없음**: 시뮬레이션과 동일한 데이터 구조
- → 실제 로봇에서도 동일한 reward 계산 가능
- **NOTE**: RH56F1_R (오른손)은 **left_ring**을 타겟으로 함

#### 4.2.3 Phase 3: Transport to Target (`hook_to_target`)

**목적**: 후크를 목표 위치로 정밀하게 이동

**구현**:
```python
# Target: 초기 hook position + offset (-0.2, 0.0, 0.3)
target_pos = initial_hook_pos + torch.tensor([-0.2, 0.0, 0.3])
d_hook_target = torch.norm(hook_pos - target_pos)
reward = 1 - tanh(d_hook_target / 0.02)  # std=0.02m (tight tolerance, 2cm)
```

**특성**:
- `d=0cm` → `reward=1.0` (정확히 도달)
- `d=2cm` → `reward≈0.24` (std 거리)
- `d=10cm` → `reward≈0.0004` (실패)

**실제 데이터**:
- `hook_pos`: Zivid camera vision
- `target_pos`: Fixed world coordinates (미리 계산)

### 4.3 제거된 Reward (Sim-to-Real 고려)

| Removed Term | 이유 | 대안 |
|--------------|------|------|
| `hook_spring_separation` | Spring position tracking 불가 (vision 어려움) | `hook_to_target`으로 간접 달성 |
| `spring_collision_penalty` | Chassis body tracking 필요 (복잡함) | Implicit learning (collision → bad reward) |

**Sim-to-Real Transfer 전략**:
- **Vision-based rewards only** (hook position, target)
- Spring/Chassis는 직접 tracking 안 함
- → 간접적으로 학습 (hook 제거 성공 = 자동으로 spring 회피)

---

## 5. Termination Conditions

| Termination | Condition | 목적 |
|-------------|-----------|------|
| `time_out` | 8.0s (480 steps @ 60Hz) | Episode 길이 제한 |
| `object_out_of_bound` | **left_ring body**가 chassis 기준 ±1m 벗어남 | Hook 폭발 방지 & Reset |
| `abnormal_robot` | Joint velocity > limit × **2** | 물리 폭발 방지 |

**Out of Bound Range** (chassis-relative, 2026-01-07 축소):
```python
chassis_pos = (-0.75, 1.3, 0.1)  # World coordinates
bound_range = {
    "x": (-1.75, 0.25),  # chassis_x ± 1.0 (기존 ±2.0에서 축소)
    "y": (0.3, 2.3),     # chassis_y ± 1.0 (기존 ±2.0에서 축소)
    "z": (-0.9, 1.1),    # chassis_z ± 1.0 (기존 ±2.0에서 축소)
}
# asset_cfg에 body_names=["left_ring"] 지정하여 hook body position 체크
```

**Abnormal Robot Termination (2026-01-07 수정)**:
```python
# 기존: ×1000 (너무 느슨 - hdr35의 mass/urdf 없을 때 임시 세팅)
return (robot.data.joint_vel.abs() > (robot.data.joint_vel_limits * 1000)).any(dim=1)

# 수정: ×2 (iiwa 환경에서 검증된 값 차용)
return (robot.data.joint_vel.abs() > (robot.data.joint_vel_limits * 2)).any(dim=1)
```

**버그 수정 (2026-01-07)**:
```python
# 기존 (버그): wire root만 체크 → hook 폭발해도 감지 못함
"asset_cfg": SceneEntityCfg("object")  # root_pos_w 사용

# 수정: left_ring body position 체크 → hook 폭발 시 termination & reset
"asset_cfg": SceneEntityCfg("object", body_names=["left_ring"])  # body_pos_w 사용
```

| 상황 | 기존 | 수정 후 |
|------|------|---------|
| Hook(left_ring) 폭발 | ❌ 감지 못함 | ✅ Termination → Reset |
| Wire root 폭발 | ✅ Termination | ✅ Termination |

**실제 로봇**:
- Safety limits (workspace boundary)
- Force limits (ATI F/T sensor monitoring)
- Emergency stop (operator intervention)

---

## 6. RL Training Configuration

### 6.1 PPO Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Algorithm** | PPO (Proximal Policy Optimization) | On-policy RL |
| **Network** | Actor-Critic MLP [512, 256, 128] | Separate networks |
| **Activation** | ELU | Smooth, non-saturating |
| **Learning Rate** | 1e-3 | Adaptive schedule |
| **Discount Factor (γ)** | 0.99 | Long-term rewards |
| **GAE λ** | 0.95 | Advantage estimation |
| **Horizon Length** | 36 steps | Rollout length |
| **Minibatch Size** | 144 | **SMALL** (16 envs × 36 / 4) |
| **Mini Epochs** | 5 | PPO updates per rollout |
| **Clip Ratio** | 0.2 | PPO policy clipping |
| **Value Coef** | 4.0 | Critic loss weight |
| **Entropy Coef** | 0.001 | Exploration bonus |
| **Max Epochs** | 7500 | Training iterations |

### 6.2 Environment Settings

| Parameter | Current | Recommended | 비고 |
|-----------|---------|-------------|------|
| **Num Envs** | 16 | 256 | **CRITICAL**: 너무 작음 |
| **Env Spacing** | 8.0m | 4.0m | 줄일 수 있음 |
| **Episode Length** | 8.0s (480 steps) | - | Hook removal은 lift보다 오래 걸림 |
| **Control Freq** | 60 Hz | - | Decimation=2 |
| **Sim dt** | 1/120s | - | PhysX timestep |

**⚠️ CRITICAL ISSUE: Minibatch Size Too Small**
- Current: `batch_size = 16 envs × 36 horizon = 576` → `minibatch = 144` (4 minibatches)
- DexSuite: `batch_size = 4096 × 36 = 147,456` → `minibatch = 36,864` (4 minibatches)
- **문제**: 16개 환경은 학습 안정성 부족
- **해결**: 256개 환경으로 증가 → `batch_size = 9,216` → `minibatch = 2,304`

### 6.3 Domain Randomization

**Startup Randomization** (episode 시작 시 1회):
```python
robot_friction:       [0.9, 1.1]     # Robot body friction
object_friction:      [0.5, 1.0]     # Wire/hook friction
joint_stiffness:      [0.9, 1.1]×    # Actuator stiffness
joint_damping:        [0.9, 1.1]×    # Joint damping
joint_friction:       [0.0, 5.0]     # Joint friction
```

**Reset Randomization** (매 episode):
```python
reset_robot_joints:   [0.0, 0.0]     # NO randomization (exact init pose)
reset_root:           [0.0, 0.0]     # Robot base position (no movement)
```

**⚠️ Wire Position**: NO randomization
- Wire는 정확히 초기 위치에 있어야 spring에 걸림
- 위치 randomization → chassis 관통 → 물리 폭발

**실제 로봇**:
- 조명 변화 (vision robustness)
- Wire 위치 변동 (±5mm, 현실적 variation)
- Robot wear (joint backlash, friction 변화)

---

## 7. Network Architecture

### 7.1 Actor-Critic Structure

```
┌──────────────────────────────────────────────────────────┐
│ Input: Concatenated Observations (1660 dim)              │
│  - Policy (115) + Proprio (585) + Perception (960)       │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│ Running Mean/Std Normalization (learned during training) │
└────────────────────┬─────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Actor MLP      │     │  Critic MLP     │
│  [512, 256, 128]│     │  [512, 256, 128]│
│  (ELU)          │     │  (ELU)          │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Action (12 dim) │     │  Value (1 dim)  │
│ Gaussian μ      │     │  State Value    │
│ Fixed σ=0       │     │                 │
└─────────────────┘     └─────────────────┘
```

### 7.2 Feature Extraction

**No PointNet** (DexPoint와 차이):
- DexPoint: PointNet for point cloud → feature embedding
- **우리**: Simple concatenation (point cloud flattened)
- → 더 단순, 하지만 point cloud structure 무시

**향후 개선**:
- PointNet 추가 → better perception
- Transformer for temporal modeling (history 대신)

---

## 8. 시뮬레이션 vs 실제 데이터 상세 비교

### 8.1 Vision System (Perception Group)

| 데이터 | 시뮬레이션 | 실제 로봇 | Gap Analysis |
|--------|-----------|----------|--------------|
| **Hook Point Cloud** | Isaac Sim mesh sampling (perfect) | **Zivid camera** RGB-D + segmentation | ✓ 획득 가능 |
| └ Segmentation | Ground-truth (USD prim) | Learning-based or color-based | ⚠️ Detection failure 가능 |
| └ Noise | 없음 (perfect) | Sensor noise, occlusion | ⚠️ Domain gap |
| └ Resolution | 64 points (downsampled) | ~1M points → 64 (FPS) | ✓ 동일하게 처리 가능 |
| └ Frame | Robot base frame | Camera frame → Transform | ✓ Calibration 필요 |

**Sim-to-Real Gap 완화**:
1. **Domain Randomization**:
   - Point cloud noise injection (Gaussian, outlier)
   - Random dropout (occlusion simulation)
   - Jittering (registration error)

2. **Data Augmentation**:
   - Training 시 vision noise 추가
   - Camera pose perturbation

### 8.2 Proprioception (Proprio Group)

| 데이터 | 시뮬레이션 | 실제 로봇 | Gap Analysis |
|--------|-----------|----------|--------------|
| **Joint Position** | PhysX simulation (perfect) | **Robot encoders** (absolute/incremental) | ✓ 거의 동일 |
| └ Noise | 없음 | Quantization noise (~0.001 rad) | ✓ 무시 가능 |
| └ Delay | 없음 (instant) | ~1-2ms (communication) | ✓ 무시 가능 (60Hz) |
| **Joint Velocity** | Numerical differentiation | Encoder differentiation or tachometer | ✓ 동일 |
| └ Noise | 없음 | Higher noise (differentiation) | ⚠️ Filtering 필요 |
| **Hand Tips State** | PhysX body tracking (perfect) | **Forward kinematics** (computed) | ✓ 동일 계산 |
| └ Error | 없음 | FK error (~mm level) | ✓ 무시 가능 |

**Sim-to-Real Gap**:
- Joint pos/vel는 gap 거의 없음 (encoder 정확도 높음)
- Hand tips state는 FK로 정확히 계산 가능

### 8.3 Contact Forces (Proprio Group)

| 데이터 | 시뮬레이션 | 실제 로봇 | Gap Analysis |
|--------|-----------|----------|--------------|
| **Fingertip Contact** | PhysX ContactSensor (각 finger) | ✅ **Tashan tactile sensor** | ✓ **Gap 없음** |
| └ Sensor 사양 | Perfect force measurement | Tashan 정전용량식 촉각 센서 | ✓ 5개 fingertip 측정 가능 |
| └ Data format | 3D force vector (Fx, Fy, Fz) | 3D force vector (동일) | ✓ 구조 동일 |
| └ Noise level | 없음 (perfect) | Sensor noise (~0.1N) | ⚠️ 약간의 noise (무시 가능) |

**✅ GOOD NEWS: Contact Observation 가능!**

**우리 환경 설정**:
- `contact`: 75 dim (5 fingers × 3D × 5 history)
- Reward `good_finger_contact`: 접촉력 > 0.5N
- **Tashan tactile sensor**: RH56F1 팁 및 손바닥에 부착

**실제 로봇 데이터 획득**:
1. ✅ **Tashan tactile sensor가 각 fingertip에 장착**
2. 개별 손가락별 3D force vector 측정 가능
3. 시뮬레이션과 **동일한 데이터 구조**
4. → **Sim-to-Real Gap 최소화!**

**추가 센서 (향후)**:
- ⏳ **ATI F/T sensor (ATI NET Axia80-M50)**: 손목 장착, 전역 힘/토크 측정
- 현재 USD file 미완성 → 시뮬레이션에 아직 미반영
- → 향후 observation에 추가 가능 (safety monitoring, advanced control)

**Sim-to-Real Transfer Strategy**:
- Tactile sensor data를 그대로 사용 가능
- Sensor noise simulation 추가 (domain randomization)
- Calibration: sensor frame ↔ robot frame

### 8.4 Policy Group (Goal Information)

| 데이터 | 시뮬레이션 | 실제 로봇 | Gap Analysis |
|--------|-----------|----------|--------------|
| **Hook Position** | PhysX body tracking (perfect) | **Zivid camera** vision | ⚠️ Vision error |
| **Hook Orientation** | PhysX quaternion | Zivid + PCA or marker | ⚠️ Orientation harder |
| **Actions History** | Buffer (perfect memory) | Controller log | ✓ 동일 |

---

## 9. Sim-to-Real Transfer 전략

### 9.1 현재 설정의 Sim-to-Real Gap

| 구성 요소 | Gap Level | 해결 방법 |
|-----------|----------|----------|
| **Vision (Point Cloud)** | ⚠️ Medium (연동 전) | ⏳ Zivid camera 연동 필요 + Domain randomization |
| **Proprioception (Joints)** | ✓ Low | Gap 거의 없음 (encoder 정확) |
| **Contact Forces** | ✓ **Low** | ✅ **Tashan tactile sensor로 측정 가능!** |
| **Physics (Wire dynamics)** | ⚠️ Medium | Real-world data collection + Domain randomization |

### 9.2 추천 수정 사항

#### 9.2.1 현재 설정 유지 (Contact Observation 그대로)
```python
# ✅ GOOD: Contact observation 유지
# Tashan tactile sensor가 실제로 측정 가능하므로 유지!
class RemoveHookObservationsCfg:
    @configclass
    class ProprioObsCfg(ObsGroup):
        contact: ObsTerm = ObsTerm(...)  # 유지!
```

**이유**:
- Tashan tactile sensor가 각 fingertip에 장착되어 있음
- 시뮬레이션과 동일한 데이터 구조
- → Sim-to-real gap 없음!

#### 9.2.2 Reward 수정 불필요 (현재 설정 적절)
```python
# ✅ GOOD: Contact-based reward 유지
class RemoveHookHdr35RewardCfg(remove_hook.RemoveHookRewardsCfg):
    good_finger_contact = RewTerm(func=mdp.contacts, weight=3.0)  # 유지!
```

#### 9.2.3 Vision Domain Randomization 추가 (향후)
```python
# Point cloud noise
object_point_cloud = ObsTerm(
    func=mdp.object_point_cloud_b,
    noise=Unoise(n_min=-0.01, n_max=0.01),  # ±1cm noise
    clip=(-5.0, 5.0),
    params={"num_points": 64, "flatten": True},
)
```

#### 9.2.4 환경 개수 증가
```python
# remove_hook_env_cfg.py
scene: RemoveHookSceneCfg = RemoveHookSceneCfg(
    num_envs=256,  # 16 → 256
    env_spacing=4.0,  # 8.0 → 4.0
)
```

### 9.3 실제 로봇 배포 체크리스트

**Hardware** (우리 환경):
- [x] ✅ Robot: HDR35_20 (2대)
- [x] ✅ Gripper: RH56F1_R (Inspire Hand, 6-DoF)
- [x] ✅ Tactile sensor: Tashan 정전용량식 (fingertips)
- [x] ✅ F/T sensor: ATI NET Axia80-M50 (손목)
- [x] ✅ Vision: Zivid2 M70 (구매 완료)
- [ ] ⏳ Zivid camera calibration (robot base frame)
- [ ] ⏳ Workspace safety boundary 설정

**Software** (향후 구현):
- [ ] ⏳ Zivid ROS driver 설정 (point cloud publish)
- [ ] ⏳ Point cloud segmentation node (wire/hook detection)
- [ ] ⏳ Tashan tactile sensor ROS driver (contact force publish)
- [ ] ⏳ Robot interface node (action → HDR35_20 controller)
- [ ] ⏳ Policy inference node (observation → action, 60 Hz)
- [ ] ⏳ ATI F/T sensor ROS driver (향후, USD file 완성 후)

**Calibration** (필수):
- [ ] ⏳ Camera ↔ Robot base frame extrinsic calibration
- [ ] ⏳ Tactile sensor calibration (force offset, scaling)
- [ ] ⏳ FK model verification (USD vs real robot)
- [ ] ⏳ Vision system accuracy test (hook position error < 5mm)

**Testing** (단계별):
- [ ] ⏳ Teleoperation test (manual control)
- [ ] ⏳ Tactile sensor validation (grasp test)
- [ ] ⏳ Open-loop policy test (no feedback)
- [ ] ⏳ Closed-loop policy test (with vision + tactile feedback)
- [ ] ⏳ Safety test (emergency stop, force limits)

**현재 상태**:
- ✅ Hardware 준비 완료 (로봇, 센서, 카메라)
- ⏳ Software integration 대기 (ROS driver, calibration)
- 🎯 **현재 목표**: 시뮬레이션에서 먼저 학습 성공 검증

---

## 10. 현재 학습 상태 및 문제점

### 10.1 Training Setup
```bash
# 현재 실행 명령어
python scripts/rl_games/train.py --task Isaac-RemoveHook-Hdr35-RH56F1-v0
```

**Config Path**:
- Env: `source/isaaclab_tasks/.../remove_hook/config/hdr35_20_rh56f1_r/`
- RL: `agents/rl_games_ppo_cfg.yaml`

### 10.2 알려진 문제점 및 버그 수정 이력

| 문제 | 상태 | 해결 방법 |
|------|------|----------|
| **Wire 물리 폭발** | ✓ 해결 | Joint damping=1.0, position 수정 |
| **Chassis replication** | ✓ 해결 | Instanceable USD 사용 (RigidBody, FixedJoint 제거) |
| **Robot init joint 미적용** | ✓ 해결 | `reset_robot_joints` EventTerm 활성화 |
| **환경 개수 부족 (16개)** | ❌ 미해결 | 256개로 증가 필요 |
| **Contact observation** | ✓ 해결 | ✅ Tashan tactile sensor로 측정 가능 (Gap 없음!) |
| **Reach reward 버그** | ✓ 해결 (2026-01-07) | `root_pos_w` → `body_pos_w[:, body_ids]` (left_ring) |
| **Termination 버그** | ✓ 해결 (2026-01-07) | `out_of_bound`에서 left_ring body position 체크 |
| **Point Cloud 위치 오프셋** | ✓ 해결 (2026-01-07) | Wire 전체 → left_ring만 샘플링 + body_pos_w 사용 |
| **Termination 배수 너무 느슨** | ✓ 해결 (2026-01-07) | ×1000 → ×2 (iiwa 값 차용) |
| **Out of bound 범위 너무 넓음** | ✓ 해결 (2026-01-07) | ±2.0m → ±1.0m |
| **Wire reset 시 위치 오류** | ⏸️ 보류 (01071616) | 주석처리 - spring 충돌 문제 미해결 |

**버그 수정 상세 (2026-01-07)**:

1. **`object_ee_distance` (Reach Reward)**:
   ```python
   # 기존 (버그): wire root position 사용
   object_pos = object.data.root_pos_w

   # 수정: left_ring body position 사용
   object_pos = object.data.body_pos_w[:, object_cfg.body_ids]
   ```

2. **`out_of_bound` (Termination)**:
   ```python
   # 기존 (버그): wire root만 체크 → hook 폭발해도 감지 못함
   "asset_cfg": SceneEntityCfg("object")

   # 수정: left_ring body 체크 → hook 폭발 시 termination & reset
   "asset_cfg": SceneEntityCfg("object", body_names=["left_ring"])
   ```

3. **Contact Sensor Filter**:
   ```python
   # 기존: Wire 전체와 접촉 감지
   filter_prim_paths_expr=["{ENV_REGEX_NS}/Wire"]

   # 수정: left_ring만 접촉 감지
   filter_prim_paths_expr=["{ENV_REGEX_NS}/Wire/left_ring*"]
   ```

4. **`object_point_cloud_b` (Point Cloud)**:
   ```python
   # 기존 (버그): Wire 전체에서 샘플링 + root_pos_w로 transform → 점이 mesh 위로 뜸
   params={"num_points": 64, "flatten": True}

   # 수정: left_ring만 샘플링 + body_pos_w로 transform
   params={
       "num_points": 64,
       "flatten": True,
       "object_cfg": SceneEntityCfg("object", body_names=["left_ring"]),
   }
   ```
   - **원인**: Wire는 Articulation이라 root와 body(left_ring) 위치가 다름
   - **효과**: Point cloud가 left_ring mesh에 정확히 붙음 + 타겟에 집중

5. **Wire Reset 이슈 (01071616 - 보류)**:

   **Wire USD 구조** (wire_revolute_collision_flattened.usd):
   ```
   wire_model (defaultPrim, Articulation Root, NO RigidBodyAPI)
   ├── wire (Xform, RigidBodyAPI) - 본체, z=+0.05075 offset
   ├── right_ring (Xform, RigidBodyAPI)
   │   └── right_rjoint (PhysicsRevoluteJoint) - wire와 연결
   │       - Joint Limits: -60° ~ +10°, default=0°
   └── left_ring (Xform, RigidBodyAPI) ← Target!
       └── left_rjoint (PhysicsRevoluteJoint) - wire와 연결
           - Joint Limits: -10° ~ +60°, default=0°
   ```

   **문제 상황**:
   - Spawn 시: `wire_model` (Articulation Root) 기준으로 배치 → 정상
   - Reset 시: `wire` (PhysX base link = 첫 번째 RigidBody) 기준으로 배치 → z offset 차이 발생
   - wire는 wire_model 기준 z=+0.05075 offset이 있어서 reset 시 위치가 달라짐

   **시도한 해결책**:
   ```python
   # reset_object: z offset 보정
   "pose_range": {"z": [0.05075, 0.05075]}  # wire vs wire_model offset

   # reset_object_joints: joint angle을 0으로 reset
   "position_range": [0.0, 0.0]
   ```

   **현재 상태**:
   - 위치는 보정되었으나, reset 시 고리가 spring과 충돌하여 계속 움직임
   - 학습이 불가능한 상태 → **주석처리 (STOP HERE 01071616)**
   - 향후 spring collision 문제 해결 필요

### 10.3 예상되는 학습 어려움

**Task 난이도**:
- DexSuite (Lift): 쉬움 (큐브 들기)
- **Remove Hook**: 어려움 (얇은 wire 파지 + 정밀 조작)
- → 학습 시간 더 오래 걸릴 것

**Under-actuated Hand**:
- Inspire (6-DoF): Adaptive grasping은 좋지만 정밀도 낮음
- DG5F (20-DoF): 정밀 제어 가능
- → Inspire로 hook removal은 challenging

**Vision Dependency**:
- Hook은 매우 작음 (wire diameter ~2mm)
- Point cloud 64 points로 충분한가?
- → Vision noise에 취약할 수 있음

---

## 11. 향후 개선 방향

### 11.1 즉시 적용 가능

1. **환경 개수 증가**: 16 → 256
2. **Contact observation 유지**: ✅ Tashan tactile sensor로 실제 측정 가능
3. **Vision noise 추가**: Domain randomization (향후 Zivid 연동 시)
4. **Reward tuning**: Weight 조정 (contact reward 유지)

### 11.2 중기 개선

1. **PointNet 추가**: Point cloud feature extraction
2. **Curriculum learning**: 쉬운 task부터 (wire 위치 고정 → 변동)
3. **Real-world data collection**: Wire dynamics validation

### 11.3 장기 목표

1. **Sim-to-real transfer**: 실제 로봇 테스트
2. **Multi-task learning**: Hook removal + assembly
3. **Human demonstration**: Imitation learning 추가

---

## 12. 요약 및 결론

### 12.1 Observation 구조
```
Total: 1640 dim
├─ Policy (95):      Hook pose (35) + Actions history (60)
├─ Proprio (585):    Joints (120) + Fingertips state (390) + Contact (75) (✅ Tashan sensor로 측정 가능)
└─ Perception (960): Hook/Wire point cloud (⏳ Zivid camera 연동 예정, 현재 GT 사용)
```

### 12.2 핵심 Reward
```
R_total = 2.0×reaching + 3.0×contact + 10.0×transport - penalties
```

### 12.3 Sim-to-Real 전략

**현재 가능 (Gap 없음/낮음)**:
- ✅ Proprioception (Robot encoders) - Gap 거의 없음
- ✅ Contact forces (Tashan tactile sensor) - **Gap 없음!** 실제 측정 가능
- ✅ Actions (Controller commands) - Gap 없음

**현재 준비 중 (연동 예정)**:
- ⏳ Vision (Zivid camera) - 구매 완료, 연동 대기 중
- ⏳ ATI F/T sensor - USD file 미완성, 향후 추가 예정

**어려움 (Domain Gap)**:
- ⚠️ Physics gap (Wire dynamics) - 시뮬에선 rigid, 실제론 deformable → Domain randomization 필요

**핵심 메시지**:
> **현재는 시뮬레이션에서 학습 성공을 먼저 검증**하는 단계입니다.
> - Contact observation: ✅ **유지** (Tashan tactile sensor로 실제 측정 가능)
> - Vision (Point cloud): ⏳ 현재 GT 사용, 향후 Zivid 연동 시 domain randomization 추가
> - 우리 환경은 tactile sensor가 있어서 sim-to-real gap이 **최소화**될 수 있습니다!

---

**작성일**: 2025-01-07 (실제 환경 정보 반영: 2026-01-07)
**분석 대상**: Isaac-RemoveHook-Hdr35-RH56F1-v0
**RL Algorithm**: PPO (RL-Games)
**참고**: DexPoint, DexPBT, Isaac Lab Documentation
