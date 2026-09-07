# UR10e 기반 Dexterous Manipulation 환경 분석
## DexPBT & DexPoint 형식 분석 문서

---

## 1. 환경 개요

### 1.1 실험 목적
**Underactuated (6-DoF Inspire Hand) vs Fully-actuated (20-DoF DG5F Hand)**의 강화학습 성능 비교
- 동일한 RL algorithm (PPO)
- 동일한 arm (UR10e 6-DoF)
- 동일한 task (Object Lifting)
- **유일한 차이**: Hand actuation structure

### 1.2 환경 구성

#### **UR10e + DG5F Right Hand**
- **Arm**: UR10e (6-DoF collaborative robot)
- **Hand**: DG5F Right (20-DoF fully-actuated, 5 fingers)
- **Objects**: 16가지 primitive shapes (1.0x, Kuka-Allegro baseline과 동일)
- **Total DoF**: 26 (6 arm + 20 hand)

#### **UR10e + Inspire Right Hand**
- **Arm**: UR10e (6-DoF collaborative robot, DG5F와 동일)
- **Hand**: Inspire Right (6-DoF underactuated, 5 fingers)
- **Objects**: 16가지 primitive shapes (0.5x, 손 크기에 맞춰 스케일)
- **Total DoF**: 12 (6 arm + 6 hand)

---

## 2. 공통점과 차이점

### 2.1 공통점 (Controlled Variables) ✅

| 항목 | 설정 |
|------|------|
| **RL Algorithm** | PPO (Proximal Policy Optimization) |
| **Network** | MLP [512, 256, 128] (3-layer, ELU activation) |
| **Learning Rate** | 1e-3 (adaptive schedule) |
| **Discount Factor (γ)** | 0.99 |
| **GAE λ** | 0.95 |
| **Horizon Length** | 36 steps |
| **Minibatch Size** | 36,864 |
| **Episode Length** | 240 steps (4.0s @ 60Hz) |
| **Num Envs** | 4,096 (parallel simulation) |
| **Observation Groups** | Policy, Proprio, Perception |
| **Reward Structure** | 동일 weights (position_tracking=2.0, success=10.0, etc.) |
| **Domain Randomization** | 동일 protocol (friction, stiffness, mass, etc.) |
| **Object Geometries** | 16가지 (Cuboids, Spheres, Capsules, Cones) |

### 2.2 차이점 (Experimental Variables) ❌

| 항목 | UR10e + DG5F | UR10e + Inspire |
|------|--------------|-----------------|
| **Hand DoF** | 20 (fully-actuated) | 6 (underactuated) |
| **Action Dimension** | 26 (6 arm + 20 hand) | 12 (6 arm + 6 hand) |
| **Object Size** | 1.0x (baseline) | 0.5x (손 크기 비례) |
| **Observation Size** | 1870 or 1890 (FT 유무) | ~1870 |
| **Contact Force Clip** | (-50, 50) N | (-20, 20) N |
| **Actuation Type** | Independent finger control | Coupled finger motion |

---

## 3. Observation 상세 (DexPoint 형식)

### 3.1 Observation Groups

우리 환경은 **3개 그룹**으로 구성 (DexPoint의 point cloud + proprio 구조 참고):

```
Total Observation Dimension:
- DG5F (WITHOUT FT): 1870
- DG5F (WITH FT): 1890 (ati_force +3, ati_force_threshold +1)
- Inspire: ~1870
```

#### **Group 1: Policy (185 dim)** - 목표 정보
| Observation | Dimension | Description |
|-------------|-----------|-------------|
| `object_quat_b` | 20 (4 × 5 history) | 물체 자세 (quaternion, body frame) |
| `target_object_pose_b` | 35 (7 × 5 history) | 목표 pose (pos + quat, body frame) |
| `actions` | 130 (26 × 5 history) | 이전 action history |

#### **Group 2: Proprio (725 dim)** - 로봇 상태
| Observation | Dimension | Description |
|-------------|-----------|-------------|
| `joint_pos` | 130 (26 × 5 history) | Joint positions |
| `joint_vel` | 130 (26 × 5 history) | Joint velocities |
| `hand_tips_state_b` | 390 (78 × 5 history) | Fingertip states (pos, vel, quat) × 6 tips |
| `contact` | 75 (15 × 5 history) | Contact forces (5 fingers × 3D) |
| `ati_force` (FT only) | 15 (3 × 5 history) | ATI sensor force (3D vector) |
| `ati_force_threshold` (FT) | 5 (1 × 5 history) | Force threshold (scalar) |

#### **Group 3: Perception (960 dim)** - 물체 인식
| Observation | Dimension | Description |
|-------------|-----------|-------------|
| `object_point_cloud` | 960 (192 × 5 history) | Object point cloud (64 points × 3D) |

### 3.2 History Length
- **모든 observation은 5 step history** (temporal information)
- DexPoint처럼 시간적 맥락 제공 (과거 정보로부터 학습)

### 3.3 Observation Processing
```
Raw Observation → Clipping → Noise Addition → Normalization (Running Mean/Std) → History Stacking → Policy Network
```

---

## 4. Reward 상세 (DexPBT 형식)

### 4.1 Reward Components

| Reward Term | Weight | Description | Formula |
|-------------|--------|-------------|---------|
| **Sparse Rewards** |
| `success` | 10.0 | 물체가 목표 도달 | `exp(-||p_obj - p_goal||^2 / σ_pos^2) × exp(-||q_obj - q_goal||^2 / σ_rot^2)` |
| **Dense Rewards** |
| `position_tracking` | 2.0 | 물체-목표 거리 | `exp(-||p_obj - p_goal||^2 / 0.2^2)` |
| `fingers_to_object` | 1.0 | 손가락-물체 거리 | `exp(-d_fingertips / 0.4^2)` |
| `good_finger_contact` | 0.5 | 손가락 접촉 보상 | `∑ I(F_i > 1.0N)` (threshold-based) |
| **Penalty Terms** |
| `action_l2` | -0.005 | Action magnitude | `-||a_t||^2` |
| `action_rate_l2` | -0.005 | Action smoothness | `-||a_t - a_{t-1}||^2` |
| `ground_contact_penalty` | -0.3 | 테이블 접촉 방지 | `-I(contact_with_table)` |
| `early_termination` | -1.0 | 비정상 종료 | `-I(abnormal_state)` |
| **FT Sensor Only** |
| `ati_excessive_force_penalty` | -0.005 | 과도한 힘 방지 | `-ReLU(||F|| - threshold)` |

### 4.2 Reward Shaping Strategy
- **Sparse reward (success)**: 최종 목표 달성
- **Dense rewards**: 학습 가이드 (hand-crafted shaping)
- **Penalty terms**: 불안정한 행동 억제

---

## 5. Network Architecture

### 5.1 Actor-Critic Structure (DexPoint 참고)

```
[Observation Groups] → [Feature Extraction] → [Actor & Critic Networks]

┌──────────────────────────────────────────────────────────────┐
│ Input: Concatenated Observations (1870 or 1890 dim)          │
│  - Policy (185) + Proprio (725) + Perception (960)           │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ Running Mean/Std Normalization                                │
│  - Input normalization (learned during training)              │
└────────────────────┬─────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Actor MLP      │     │  Critic MLP     │
│  [512, 256, 128]│     │  [512, 256, 128]│
│  (ELU activation)│    │  (ELU activation)│
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Action (26 dim) │     │  Value (1 dim)  │
│ Gaussian Policy │     │  State Value    │
└─────────────────┘     └─────────────────┘
```

### 5.2 Network Details

#### **Feature Extraction**
- **No PointNet** (DexPoint와 다름)
- Simple **concatenation** of observation groups
- Running mean/std normalization만 사용

#### **Actor Network**
- **Input**: 1870 (or 1890) dim
- **Hidden**: [512, 256, 128]
- **Activation**: ELU
- **Output**: 26 (or 12) dim Gaussian mean (μ)
- **Std**: Fixed (σ = const, learned during training)

#### **Critic Network**
- **Input**: 1870 (or 1890) dim (동일)
- **Hidden**: [512, 256, 128] (동일)
- **Activation**: ELU
- **Output**: 1 dim (state value V(s))

### 5.3 Action Distribution
```
a_t ~ N(μ_θ(s_t), σ_θ)
```
- **μ_θ**: Actor network output
- **σ_θ**: Fixed standard deviation (initialized to 0)

---

## 6. Training Pipeline (DexPBT 참고)

### 6.1 Training Loop

```
┌────────────────────────────────────────────────────────────┐
│ Phase 1: Rollout Collection (Parallel Simulation)          │
│  - 4096 envs × 36 horizon = 147,456 samples per iteration  │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────┐
│ Phase 2: Advantage Estimation (GAE λ=0.95)                 │
│  - Compute advantages: A_t = ∑(γλ)^k δ_{t+k}              │
│  - Normalize advantages                                     │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────┐
│ Phase 3: PPO Update (5 mini-epochs)                        │
│  - Minibatch size: 36,864                                   │
│  - Clip ratio: 0.2                                          │
│  - Value loss coef: 4.0                                     │
│  - Entropy coef: 0.001                                      │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────┐
│ Phase 4: Curriculum Update (ADR)                           │
│  - Adjust domain randomization based on success rate       │
│  - Update observation noise ranges                          │
│  - Modify gravity curriculum                                │
└────────────────────────────────────────────────────────────┘
```

### 6.2 Curriculum Learning (ADR)

우리 환경은 **Automatic Domain Randomization (ADR)** 사용:

| Curriculum Term | Range | Adjust Based On |
|-----------------|-------|-----------------|
| Observation Noise | [0.0, 0.05] | Success rate |
| Gravity | [0.0, -9.81] | Success rate |
| Joint Friction | [0.0, 5.0] | Success rate |
| Object Mass | [0.2, 2.0]× | Success rate |
| Joint Stiffness/Damping | [0.8, 1.2]× | Success rate |

### 6.3 Domain Randomization

**Startup Randomization** (episode 시작 시):
- Robot physics material: friction [0.9, 1.1]
- Object physics material: friction [0.5, 1.0]
- Joint stiffness/damping: [0.8, 1.2]×
- Object mass: [0.2, 2.0]×

**Reset Randomization** (매 episode):
- Object pose: x, y, z, roll, pitch, yaw
- Robot joint pos: ±0.5 rad (arm), ±0.05 rad (hand)
- Table pose: ±5cm

---

## 7. 학습 성능 분석 (현재 결과)

### 7.1 DG5F Hand (Best: 2025-12-03)
```
✅ WITH FT Sensor:
  - Success: 5.31 (최고 성능)
  - Contact: 0.38
  - Position Tracking: 1.35
  - Epochs: 7499

⚠️ WITHOUT FT Sensor:
  - Success: 3.40 (두 번째)
  - Contact: 0.26
  - Position Tracking: 0.89
  - Epochs: 7499
```

### 7.2 Inspire Hand (현재 학습 중)
```
🔴 학습 초기 단계:
  - Success: 0.00
  - Contact: 0.0005 (거의 없음)
  - Position Tracking: 0.0002
  - Epochs: 1383 (목표: 10000)
```

### 7.3 주요 발견
1. **FT sensor가 contact를 줄이지만 success는 높임**
   - WITH FT: contact 낮음 (0.09), success 높음 (1.24)
   - WITHOUT FT: contact 높음 (0.17), success 낮음 (0.99)

2. **학습 불안정성 높음**
   - Success std = 1.4~2.0 (mean의 1.4~1.6배)
   - 실험 간 변동성 매우 큼

3. **Inspire는 아직 학습 필요**
   - Contact가 거의 없음 → threshold 조정 필요?
   - 더 긴 학습 시간 필요

---

## 8. 그림 초안 (AI 생성용 프롬프트)

### 그림 1: 환경 구조 다이어그램

**프롬프트**:
```
Create a technical diagram showing:

LEFT SIDE: UR10e + DG5F (20-DoF)
- UR10e arm (6 joints)
- DG5F hand (20 joints, 5 fingers)
- 1.0x object (cube/sphere)

RIGHT SIDE: UR10e + Inspire (6-DoF)
- UR10e arm (6 joints, 동일)
- Inspire hand (6 joints, 5 fingers)
- 0.5x object (smaller cube/sphere)

MIDDLE: Comparison table
- DoF: 26 vs 12
- Objects: 1.0x vs 0.5x
- Actuation: Fully vs Underactuated

Style: Clean technical diagram, side-by-side comparison
Colors: DG5F in blue, Inspire in orange
```

### 그림 2: Network Architecture

**프롬프트**:
```
Create a neural network architecture diagram similar to the DexPoint figure:

INPUT LAYER (1870 dim):
- Policy obs (185): object pose, target, actions
- Proprio obs (725): joints, contacts, fingertips
- Perception (960): point cloud

MIDDLE LAYERS:
- Running Mean/Std Normalization
- Split into Actor MLP and Critic MLP
- Each: [512, 256, 128] with ELU

OUTPUT LAYER:
- Actor → Action (26 dim Gaussian)
- Critic → Value (1 dim)

Style: Flow diagram with boxes and arrows
Colors: Inputs (green), Networks (orange), Outputs (red)
```

### 그림 3: Training Pipeline

**프롬프트**:
```
Create a training pipeline diagram similar to DexPBT Figure 2:

STEP 1: Parallel Simulation (4096 envs)
- Actor-Critic network
- Isaac Sim environments
- Collect 36-step rollouts

STEP 2: PPO Update
- Minibatch size: 36,864
- 5 mini-epochs
- Clip ratio: 0.2

STEP 3: Curriculum Update (ADR)
- Adjust domain randomization
- Based on success rate
- Update noise ranges

STEP 4: Save Checkpoint
- Every 50 iterations
- Save best model

Style: Flow diagram with feedback loops
Colors: Simulation (blue), Training (green), Curriculum (orange)
```

### 그림 4: Observation Structure

**프롬프트**:
```
Create an observation structure diagram:

TOP: 3 Observation Groups
- Policy (185): Goal information
- Proprio (725): Robot state
- Perception (960): Object point cloud

MIDDLE: Processing Pipeline
- History stacking (×5 timesteps)
- Clipping
- Normalization

BOTTOM: Concatenated vector (1870 dim)
→ Actor/Critic networks

Style: Layered diagram with dimensions
Colors: Policy (blue), Proprio (green), Perception (red)
```

---

## 9. 참고사항

### 9.1 DexPoint와의 차이점
- **Point Cloud 처리**: DexPoint는 PointNet 사용, 우리는 단순 concatenation
- **Imagined Point Cloud**: 없음 (우리는 직접 관찰만)
- **Network 구조**: 더 단순 (MLP만, no PointNet)

### 9.2 DexPBT와의 차이점
- **Population-Based Training**: 사용 안 함 (단일 agent PPO)
- **LSTM**: 사용 안 함 (MLP만, history stacking으로 대체)
- **Mutation**: 없음 (고정 hyperparameters)

### 9.3 우리만의 특징
- **Controlled Comparison**: Underactuated vs Fully-actuated
- **Object Size Scaling**: 손 크기에 비례
- **FT Sensor**: ATI force/torque sensor 실험
- **ADR Curriculum**: 자동 난이도 조절

---

## 10. 결론 및 향후 계획

### 10.1 현재 상태
✅ **완료된 것**:
- DG5F hand 학습 완료 (best: success=5.31)
- FT sensor 유무 비교 완료
- 환경 설정 검증 완료

🔄 **진행 중**:
- Inspire hand 학습 (현재 1383 epochs)

❌ **미완료**:
- Inspire vs DG5F 성능 비교 (Inspire 학습 필요)
- 물체별 성능 분석 (code modification 필요)
- Contact force clip 영향 분석

### 10.2 다음 스텝
1. **Inspire 학습 완료** (목표: 10000 epochs)
2. **성능 비교 분석** (Inspire vs DG5F)
3. **논문 작성** (controlled comparison 강조)
4. **Visualization** (학습 곡선, 그래프)

---

**작성일**: 2025-12-15
**분석 기준**: Isaac Lab + RL-Games (PPO)
**참고 논문**: DexPBT, DexPoint



-=----------------------------------------
파일을 분석한 결과, observation structure는 완전히 동일하지 않습니다. 자유도 차이로 인해 observation 차원 자체가 다릅니다.

  MDP 비교 분석 (FT 센서 없는 버전)

  1. Observation Space 상세 비교

  Policy Group (history=5)

  | Component            | Formula        | Inspire (12-DoF) | DG5F (26-DoF) |
  |----------------------|----------------|------------------|---------------|
  | object_quat_b        | 4              | 4                | 4             |
  | target_object_pose_b | 3 pos + 4 quat | 7                | 7             |
  | actions              | DoF            | 12               | 26            |
  | Single step          |                | 23               | 37            |
  | With history ×5      |                | 115 dim          | 185 dim       |

  Proprio Group (history=5)

  | Component         | Formula                                | Inspire (12-DoF) | DG5F (26-DoF)            |
  |-------------------|----------------------------------------|------------------|--------------------------|
  | joint_pos         | DoF                                    | 12               | 26                       |
  | joint_vel         | DoF                                    | 12               | 26                       |
  | hand_tips_state_b | 6 bodies × 13                          | 78               | 78                       |
  | └ 구성            | 3 pos + 4 quat + 3 vel_lin + 3 vel_ang |                  |                          |
  | └ bodies          | 1 palm + 5 fingertips                  | palm, .*_tip     | rl_dg_palm, rl_dg_.*_tip |
  | contact           | 5 sensors × 3 (xyz)                    | 15               | 15                       |
  | └ clip            | Force range                            | (-20, 20) N      | (-50, 50) N              |
  | Single step       |                                        | 117              | 145                      |
  | With history ×5   |                                        | 585 dim          | 725 dim                  |

  Perception Group (history=5)

  | Component          | Formula             | Inspire | DG5F    |
  |--------------------|---------------------|---------|---------|
  | object_point_cloud | 64 points × 3 (xyz) | 192     | 192     |
  | └ frame            | Robot base frame    | base    | base    |
  | With history ×5    |                     | 960 dim | 960 dim |

  Total Observation Space

  Inspire:  115 (policy) + 585 (proprio) + 960 (perception) = 1660 dim
  DG5F:     185 (policy) + 725 (proprio) + 960 (perception) = 1870 dim
  차이:     +70            +140            +0              = +210 dim

  핵심 차이점:
  - Joint-dependent observations: 12-DoF vs 26-DoF로 인해 210 dim 차이
    - joint_pos/vel: (26-12) × 2 = 28 dim × 5 history = 140 dim
    - actions: (26-12) = 14 dim × 5 history = 70 dim
  - Hand-independent observations: 동일 (hand_tips_state_b=78, contact=15, point_cloud=192)
  - Contact clip 값만 다름: effort에 맞게 설정 (Inspire 1.0N → clip ±20, DG5F 5.0N → clip ±50)

  ---
  2. Action Space 비교

  | Component     | Inspire                        | DG5F                  |
  |---------------|--------------------------------|-----------------------|
  | DoF           | 12                             | 26                    |
  | └ Arm (UR10e) | 6                              | 6                     |
  | └ Hand        | 6 (underactuated)              | 20 (fully-actuated)   |
  | Action type   | RelativeJointPosition          | RelativeJointPosition |
  | Scale         | 0.1                            | 0.1                   |
  | Control freq  | 60 Hz (dt=1/120, decimation=2) | 60 Hz                 |

  Inspire Hand (6-DoF):
  # Thumb: 2 joints (spread + bend)
  right_thumb_1_joint  # spread
  right_thumb_2_joint  # bend

  # Other 4 fingers: 1 joint each (bend only)
  right_index_1_joint
  right_middle_1_joint
  right_ring_1_joint
  right_little_1_joint

  DG5F Hand (20-DoF):
  # 5 fingers × 4 joints each
  # Finger 1-5: rj_dg_[1-5]_[1-4]
  # _1: spread (abduction/adduction)
  # _2, _3, _4: bending joints
  rj_dg_1_1, rj_dg_1_2, rj_dg_1_3, rj_dg_1_4  # Finger 1 (Thumb)
  rj_dg_2_1, rj_dg_2_2, rj_dg_2_3, rj_dg_2_4  # Finger 2 (Index)
  ...

  ---
  3. State Space (Internal) 비교

  Isaac Lab에서 state는 observation과 별개입니다:

  | State Component      | Inspire                | DG5F             | 비고           |
  |----------------------|------------------------|------------------|----------------|
  | Robot joint state    | 12 × 2 (pos+vel)       | 26 × 2 (pos+vel) | Full state     |
  | Object state         | 13 (7 pose + 6 vel)    | 13               | Same           |
  | Contact forces       | 5 sensors × 6 (wrench) | 5 sensors × 6    | Same structure |
  | Total internal state | ~60 dim                | ~110 dim         | DoF-dependent  |

  ---
  4. Reward Structure 비교

  완전히 동일합니다 (둘 다 Kuka-Allegro baseline으로 revert됨):

  | Reward Term            | Weight | Formula                           | Params               |
  |------------------------|--------|-----------------------------------|----------------------|
  | action_l2              | -0.005 | ‖a‖² clamped                      |                      |
  | action_rate_l2         | -0.005 | ‖a_t - a_{t-1}‖²                  |                      |
  | fingers_to_object      | 1.0    | exp(-d²/σ²), σ=0.4                | 6 bodies: palm+5tips |
  | good_finger_contact    | 0.5    | contacts > 1.0N                   | 5 sensors            |
  | position_tracking      | 2.0    | tanh(‖p_obj - p_target‖/σ), σ=0.2 |                      |
  | success                | 10.0   | pos < 0.1m & rot < 0.5rad         |                      |
  | early_termination      | -1.0   | abnormal_robot penalty            |                      |
  | ground_contact_penalty | -0.3   | object-ground contact             | UR10e only           |

  차이점:
  - None (완전히 동일)
  - 참고: 2025-11-18에 UH035 실험 결과를 바탕으로 Inspire, DG5F 모두 Kuka baseline으로 revert됨

  ---
  5. Termination 조건 비교

  완전히 동일합니다:

  | Termination         | Condition                       | Inspire | DG5F |
  |---------------------|---------------------------------|---------|------|
  | time_out            | 4.0s (240 steps @ 60Hz)         | ✓       | ✓    |
  | object_out_of_bound | x∈(-0.5,2.0), y∈(-2,2), z∈(0,2) | ✓       | ✓    |
  | abnormal_robot      | vel > limit × 1000              | ✓       | ✓    |

  ---
  6. Domain Randomization 비교

  완전히 동일합니다 (보수적 설정):

  | Randomization     | Range             | Inspire      | DG5F                   |
  |-------------------|-------------------|--------------|------------------------|
  | Object scale      | [0.75, 1.5]       | ✓            | ✓                      |
  | Robot friction    | [0.9, 1.1]        | ✓            | ✓                      |
  | Joint stiffness   | [0.8, 1.2]        | ✓            | ✓                      |
  | Joint damping     | [0.8, 1.2]        | ✓            | ✓                      |
  | Object mass       | [0.2, 2.0] scale  | ✓            | ✓                      |
  | Arm joints reset  | [-0.5, 0.5] rad   | ✓            | ✓                      |
  | Hand joints reset | [-0.05, 0.05] rad | ✓ (6 joints) | ✓ (15 joints, _1 제외) |

  차이점:
  - Hand joints randomization 범위는 동일하지만, 대상 조인트 개수가 다름:
    - Inspire: 6 joints (thumb_2 + 4 fingers _1)
    - DG5F: 15 joints (5 fingers × _2,_3,_4, spread _1 제외)

  ---
  요약

  ❌ 다른 점

  1. Observation 차원: Inspire 1660 vs DG5F 1870 (+210 dim)
    - joint_pos/vel: 12 vs 26
    - actions: 12 vs 26
  2. Action 차원: 12-DoF vs 26-DoF
  3. Contact clip: (-20,20) vs (-50,50) N
  4. Object size: 0.5x vs 1.0x (환경 설정 차이)

  ✅ 같은 점

  1. Observation structure: Policy, Proprio, Perception 구조 동일
  2. Hand-independent obs: hand_tips_state_b (6 bodies), contact (5 sensors), point_cloud (64 points)
  3. Reward structure: 8개 항목, weights 완전히 동일
  4. Termination: 3개 조건 동일
  5. Domain randomization: 범위 동일
  6. Control frequency: 60 Hz
  7. Episode length: 240 steps (4.0s)

  결론: "Structure는 동일하지만, DoF 차이로 인해 observation/action 차원이 다릅니다."
  
  
  
  --------------------------------------------
  ● 완벽하게 점검했습니다! CRITICAL 문제를 발견했습니다 🚨

  🔍 전체 파라미터 점검 결과

  🔴 CRITICAL: 반드시 수정 필요

  1. Reward Std 값들 (이미 발견)

  | Parameter             | 현재 (동일) | DG5F 상대비 | Inspire 상대비 | 수정 필요 |
  |-----------------------|-------------|-------------|----------------|-----------|
  | position_tracking std | 0.2m        | 물체의 4배  | 물체의 8배     | 0.1m      |
  | success pos_std       | 0.1m        | 물체의 2배  | 물체의 4배     | 0.05m     |
  | fingers_to_object std | 0.4m        | 물체의 8배  | 물체의 16배    | 0.2m      |

  2. Object Mass (🚨 새로 발견!)

  현재 설정:
  # dexsuite_env_cfg.py:68, inspire:97 (둘 다 동일)
  mass_props=sim_utils.MassPropertiesCfg(mass=0.2)

  문제점:
  | Property           | DG5F (1.0x) | Inspire (0.5x) | 비교    |
  |--------------------|-------------|----------------|---------|
  | Object size (cube) | ~5cm        | ~2.5cm         | 절반    |
  | Volume             | 125 cm³     | 15.6 cm³       | 1/8     |
  | Mass               | 0.2 kg      | 0.2 kg         | 동일 ⚠️ |
  | Density            | 1.6 g/cm³   | 12.8 g/cm³     | 8배!    |

  의미:
  - 같은 재질이라면 질량도 1/8이어야 함 (부피에 비례)
  - 현재 Inspire 물체는 DG5F 대비 밀도 8배 → 상대적으로 훨씬 무거움!
  - 비유: 성인이 5kg 들기 vs 유아가 5kg 들기

  물리적 영향:
  1. Grasping force: 작은 손으로 무거운 물체 → 더 큰 힘 필요
  2. Lifting: 0.2kg을 6-DoF 손으로 들기 vs 20-DoF 손으로 들기
  3. Contact dynamics: 무거운 물체 → 더 큰 접촉력, 관성 → 제어 어려움
  4. Success 달성: 무거운 물체를 target까지 정밀하게 이동 → 매우 어려움

  해결책:
  # Inspire object mass 수정 필요
  mass=0.025  # 0.2kg / 8 = 0.025kg (25g)

  Mass randomization도 확인:
  # dexsuite_env_cfg.py:274 (둘 다 동일)
  "mass_distribution_params": [0.2, 2.0],  # scale operation
  - DG5F: 0.2kg × [0.2, 2.0] = [0.04, 0.4]kg
  - Inspire (현재): 0.2kg × [0.2, 2.0] = [0.04, 0.4]kg ⚠️
  - Inspire (수정 후): 0.025kg × [0.2, 2.0] = [0.005, 0.05]kg ✓

  밀도 비교:
  | Material      | Density (g/cm³) | Inspire 현재       | Inspire 수정 후 |
  |---------------|-----------------|--------------------|-----------------|
  | Aluminum      | 2.7             | ❌ 12.8 (5배 높음) | ✓ 1.6 (적절)    |
  | Wood          | 0.5-0.9         | ❌ 12.8            | ✓ 1.6           |
  | Plastic (ABS) | 1.0-1.4         | ❌ 12.8            | ✓ 1.6           |

  수정 후에야 realistic density가 됩니다!

  ---
  ✅ OK: 문제 없음

  3. Object Scale Randomization

  # dexsuite_env_cfg.py:208
  "scale_range": (0.75, 1.5)  # 상대값
  - DG5F: 1.0 × [0.75, 1.5] = [0.75, 1.5] (absolute)
  - Inspire: 0.5 × [0.75, 1.5] = [0.375, 0.75] (absolute)
  - 상대적으로 동일 ✓

  4. Command Target Ranges

  # dexsuite_env_cfg.py:119-121
  pos_x=(0.7, 0.9)   # 0.2m range
  pos_y=(-0.3, 0.3)  # 0.6m range
  pos_z=(0.3, 0.6)   # 0.3m range
  - 작업 공간(workspace) 범위이므로 물체 크기와 무관
  - 둘 다 같은 테이블 위에서 작업 → 동일해야 맞음 ✓

  5. Object Reset Position Ranges

  # dexsuite_env_cfg.py:295-297
  "x": [-0.1, 0.1]   # 0.2m range
  "y": [-0.2, 0.2]   # 0.4m range
  "z": [0.0, 0.2]    # 0.2m range
  - 절대 공간 범위이므로 동일해야 맞음 ✓

  6. Domain Randomization

  robot_friction: [0.9, 1.1]
  joint_stiffness: [0.8, 1.2]
  joint_damping: [0.8, 1.2]
  arm_joints_reset: [-0.5, 0.5] rad
  hand_joints_reset: [-0.05, 0.05] rad
  - 모두 동일 ✓

  7. Observation Clipping

  hand_tips_state_b: clip=(-2.0, 2.0)
  object_point_cloud: clip=(-2.0, 2.0)
  contact: clip=(-20, 20) vs (-50, 50)  # effort에 맞게 설정됨
  - Robot base frame 기준 절대 좌표 → 동일해야 맞음 ✓
  - Contact clip은 effort 차이로 인한 것 → OK ✓

  ---
  ⚠️ 추가 고려사항

  8. Contact Force Threshold

  # good_finger_contact threshold: 1.0N (둘 다 동일)
  good_finger_contact = RewTerm(
      func=mdp.contacts,
      weight=0.5,
      params={"threshold": 1.0},  # 1.0N
  )

  현재 상황:
  - DG5F: 0.2kg 물체, 1.0N threshold → 적절
  - Inspire: 0.2kg 물체, 1.0N threshold → 적절

  Mass 수정 후:
  - Inspire: 0.025kg 물체, 1.0N threshold → 너무 높을 수 있음
  - 0.025kg × 9.8 = 0.245N (중력)
  - 물체를 살짝만 잡아도 1.0N 넘을 수 있음

  수정 제안:
  - Inspire contact threshold: 0.5N (절반)
  - 이렇게 해야 상대적 난이도 동일

  ---
  📊 종합 수정 사항

  | Parameter             | 현재  | 수정 →  | 이유          |
  |-----------------------|-------|---------|---------------|
  | position_tracking std | 0.2   | 0.1     | Scale 1/2     |
  | success pos_std       | 0.1   | 0.05    | Scale 1/2     |
  | fingers_to_object std | 0.4   | 0.2     | Scale 1/2     |
  | object mass           | 0.2kg | 0.025kg | Volume 1/8    |
  | contact threshold     | 1.0N  | 0.5N    | Mass 1/8 고려 |

  💡 왜 Inspire가 실패하는가?

  3가지 복합 원인:

  1. Scale Trap (Reward std): 대충 가까이만 가도 reward 충분 → 정밀 파지 학습 안함
  2. Heavy Object Trap (Mass): 작은 손으로 8배 무거운 물체 → 물리적으로 불가능에 가까움
  3. Underactuation Gap: 6-DoF로는 무거운 물체 제어가 매우 어려움

  현재 DG5F가 잘 되는 이유:
  - Reward std가 적절히 까다로움 (물체 크기 대비)
  - 질량이 적절함 (밀도 realistic)
  - 20-DoF로 충분한 제어 자유도

  Inspire가 안 되는 근본 원인:
  - Reward는 너무 관대 (local optima)
  - Mass는 너무 무거움 (물리적 한계)
  - → 학습이 진행될수록 "불가능한 일"을 하라고 요구받는 상황
