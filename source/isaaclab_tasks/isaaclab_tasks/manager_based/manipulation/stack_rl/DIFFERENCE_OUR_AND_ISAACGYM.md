# IsaacGym vs 우리 Stack RL 환경 비교 분석

## 📚 분석 대상

### IsaacGym 환경 (성공적으로 학습됨)
- **경로**: `/home/dyros/IsaacLab/IsaacGymEnvsTwk-main`
- **태스크**: `FrankaCubeStack` (2-cube stacking)
- **파일**:
  - Task: `isaacgymenvs/tasks/franka_cube_stack.py`
  - Config: `isaacgymenvs/cfg/task/FrankaCubeStack.yaml`
  - PPO: `isaacgymenvs/cfg/train/FrankaCubeStackPPO.yaml`

### 우리 환경 (학습 안 됨)
- **경로**: `/home/dyros/IsaacLab/source/isaaclab_tasks/.../stack_rl`
- **태스크**: `Stack-RL` (3-cube stacking)
- **파일**:
  - Config: `stack_rl_env_cfg.py`
  - Rewards: `mdp/rewards.py`
  - PPO: `config/franka/agents/rl_games_ppo_cfg.yaml`

---

## 🔍 1. Observation Space 비교

### IsaacGym (19-dim for OSC)
```python
# Line 454-456 in franka_cube_stack.py
obs = ["cubeA_quat", "cubeA_pos", "cubeA_to_cubeB_pos", "eef_pos", "eef_quat"]
obs += ["q_gripper"] if self.control_type == "osc" else ["q"]

# Breakdown:
cubeA_quat = 4          # 하단 큐브의 orientation
cubeA_pos = 3           # 하단 큐브의 position
cubeA_to_cubeB_pos = 3  # 하단 큐브에서 상단 큐브로의 상대 벡터
eef_pos = 3             # End-effector position
eef_quat = 4            # End-effector orientation
q_gripper = 2           # Gripper joint positions
-------------------
Total: 19-dim
```

**특징**:
- ✅ **매우 간결**: 19차원만 사용
- ✅ **상대 좌표 사용**: `cubeA_to_cubeB_pos` - 절대 좌표가 아닌 상대 벡터
- ✅ **cubeB의 절대 위치 없음**: 상대 위치만 제공
- ✅ **Joint 정보 없음** (OSC 모드): IK를 내부적으로 처리하므로 joint info 불필요
- ✅ **Velocity 없음**: Position만 사용
- ✅ **Cube 크기 없음**: Observation에 포함 안 함 (reward에서만 사용)

### 우리 환경 (65-dim)
```python
# stack_rl_env_cfg.py ObservationsCfg.PolicyCfg
joint_pos = 7              # 7-DOF arm
joint_vel = 7              # Joint velocities
cube_positions = 9         # 3 cubes × xyz (world frame)
cube_orientations = 12     # 3 cubes × quaternion
eef_pos = 3               # End-effector position
eef_quat = 4              # End-effector orientation
gripper_pos = 2           # Gripper joint positions
actions = 12              # Last action (7 arm + 2 gripper assumed)
cube_dimensions = 9       # 3 cubes × xyz dimensions (20251210 추가)
-------------------
Total: 65-dim
```

**특징**:
- ❌ **매우 복잡**: 65차원 (IsaacGym의 3.4배)
- ❌ **절대 좌표 사용**: 모든 큐브의 world frame position
- ❌ **상대 위치 없음**: 큐브 간 관계를 policy가 학습해야 함
- ❌ **Joint-level 정보**: Joint pos/vel 포함 (IK-Rel 사용에도 불구하고)
- ❌ **Velocity 포함**: 추가 차원
- ❌ **Last action 포함**: 12차원 추가
- ❌ **Cube dimensions**: 9차원 추가 (최근 수정)

### ⚠️ **핵심 차이점**
1. **Observation 차원**: 19 vs **65** (3.4배 차이!)
2. **상대 vs 절대 좌표**: IsaacGym은 상대 벡터 사용, 우리는 절대 좌표
3. **정보 밀도**: IsaacGym은 task에 필요한 최소 정보만 제공

---

## 💰 2. Reward Structure 비교

### IsaacGym Reward (compute_franka_reward, line 707-758)

#### Reward Components (4개)
```python
# 1. Distance Reward (dist_reward)
d = torch.norm(states["cubeA_pos_relative"], dim=-1)
d_lf = torch.norm(states["cubeA_pos"] - states["eef_lf_pos"], dim=-1)
d_rf = torch.norm(states["cubeA_pos"] - states["eef_rf_pos"], dim=-1)
dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)

# 2. Lift Reward (lift_reward)
cubeA_height = states["cubeA_pos"][:, 2] - reward_settings["table_height"]
cubeA_lifted = (cubeA_height - cubeA_size) > 0.04
lift_reward = cubeA_lifted  # Binary: 0 or 1

# 3. Align Reward (align_reward)
offset = torch.zeros_like(states["cubeA_to_cubeB_pos"])
offset[:, 2] = (cubeA_size + cubeB_size) / 2
d_ab = torch.norm(states["cubeA_to_cubeB_pos"] + offset, dim=-1)
align_reward = (1 - torch.tanh(10.0 * d_ab)) * cubeA_lifted  # 들었을 때만

# 4. Stack Success (stack_reward)
cubeA_align_cubeB = (torch.norm(states["cubeA_to_cubeB_pos"][:, :2], dim=-1) < 0.02)
cubeA_on_cubeB = torch.abs(cubeA_height - target_height) < 0.02
gripper_away_from_cubeA = (d > 0.04)
stack_reward = cubeA_align_cubeB & cubeA_on_cubeB & gripper_away_from_cubeA

# Final Reward Composition (중요!)
rewards = torch.where(
    stack_reward,
    reward_settings["r_stack_scale"] * stack_reward,  # 성공 시: 16.0
    reward_settings["r_dist_scale"] * dist_reward +    # 실패 시: 0.1 * dist
    reward_settings["r_lift_scale"] * lift_reward +    #         + 1.5 * lift
    reward_settings["r_align_scale"] * align_reward,   #         + 2.0 * align
)
```

#### Reward Scales (FrankaCubeStack.yaml)
```yaml
distRewardScale: 0.1      # Reaching 보상
liftRewardScale: 1.5      # Lifting 보상
alignRewardScale: 2.0     # Alignment 보상
stackRewardScale: 16.0    # Success 보상
```

**특징**:
- ✅ **단순하고 명확**: 4개 reward terms만 사용
- ✅ **Max 대신 Where**: `dist_reward`와 `align_reward` 중 최대값 선택 후, 성공 여부로 분기
- ✅ **Sparse + Dense 혼합**: 성공하면 sparse (16.0), 실패하면 dense (0.1 + 1.5 + 2.0)
- ✅ **자동 Curriculum**:
  - 초기: dist_reward 학습 (큐브 잡기)
  - 중기: lift_reward 학습 (들기)
  - 후기: align_reward 학습 (정렬)
  - 최종: stack_reward (성공)
- ✅ **Gripper 조건**: 성공 시 gripper가 멀리 떨어져 있어야 함 (놓았는지 확인)

### 우리 환경 Reward (20251210 수정 후)

#### Reward Components (11개)
```python
# Phase 1: cube_2 stacking (4개)
reaching_cube_2 = 3.0              # EE to cube_2 distance
grasping_cube_2 = 10.0             # Grasping check
lifting_cube_2 = 5.0               # Lifted after grasp
stacking_cube_2_on_1 = 35.0        # Hybrid (5 dense + 30 sparse)

# Phase 2: cube_3 stacking (4개, masked)
reaching_cube_3 = 3.0              # Masked
grasping_cube_3 = 10.0             # Masked
lifting_cube_3 = 5.0               # Masked
stacking_cube_3_on_2 = 55.0        # Masked hybrid (5 + 50)

# Constraints (3개)
cube_1_stays_on_table = 2.0        # cube_1 안정성
cube_1_movement_penalty = -1.0     # cube_1 움직임 페널티
stack_stability = 1.0              # 그리퍼 열렸을 때 안정성

# Success (1개)
task_success = 100.0               # 3-cube stacking success

# Action penalties (2개)
action_rate = -1e-4
joint_vel = -1e-4
```

**총 reward terms**: **14개** (action penalties 제외 시 12개)

**특징**:
- ❌ **매우 복잡**: 14개 reward terms (IsaacGym의 3.5배)
- ❌ **Explicit Phases**: Phase 1, Phase 2로 명시적 분리 + masking
- ❌ **Hybrid Reward의 복잡성**: One-time flag 관리 필요
- ❌ **Constraints 추가**: cube_1 관련 제약 조건
- ❓ **3-cube 태스크**: IsaacGym은 2-cube, 우리는 3-cube (더 어려움)

### ⚠️ **핵심 차이점**

| 항목 | IsaacGym | 우리 환경 | 문제점 |
|------|----------|-----------|--------|
| **Reward terms** | 4개 | 14개 | 너무 많음 |
| **최대 보상** | 16.0 (stack) | 229 | 스케일 차이 큼 |
| **자동 Curriculum** | ✅ `torch.where` + `max` | ❌ Explicit masking | 복잡도 증가 |
| **Reward 분기** | 성공/실패로 2가지 | Phase별 분리 | 학습 어려움 |
| **Sparse vs Dense** | 명확한 분리 | 혼재 (hybrid) | 학습 혼란 |

---

## 🎮 3. Action Space 비교

### IsaacGym
```python
# controlType: "osc" (Operational Space Control)
# Line 106-107
self.cfg["env"]["numActions"] = 7 if self.control_type == "osc" else 8

# OSC 모드 (7-dim):
actions = [
    delta_x, delta_y, delta_z,           # EE position delta (3)
    delta_roll, delta_pitch, delta_yaw,  # EE orientation delta (3)
    gripper_command                      # Binary: open(+) / close(-) (1)
]
```

**특징**:
- ✅ **Cartesian control**: End-effector 공간에서 직접 제어
- ✅ **Internal IK**: OSC가 자동으로 joint torques 계산
- ✅ **Gripper binary**: +1 (open) / -1 (close)

### 우리 환경
```python
# controlType: IK-Rel (Inverse Kinematics Relative)
# joint_pos_env_cfg.py
actions.arm_action = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    scale=0.5,
    use_default_offset=True
)
actions.gripper_action = mdp.BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_finger.*"],
    open_command_expr={"panda_finger_.*": 0.04},
    close_command_expr={"panda_finger_.*": 0.0},
)

# Actions (9-dim?):
actions = [
    delta_joint_1, ..., delta_joint_7,  # 7-DOF arm joint deltas
    gripper_open/close                   # Binary gripper
]
```

**특징**:
- ✅ **Joint Position Control**: Joint space에서 제어
- ❓ **IK-Rel**: 상대적 joint position 제어
- ✅ **Gripper binary**: Open/close

### ⚠️ **차이점**
- **Control space**: IsaacGym은 Cartesian, 우리는 Joint
- **IK 위치**: IsaacGym은 OSC 내부, 우리는 IK-Rel action
- 둘 다 비슷한 접근이지만, OSC가 더 직관적일 수 있음

---

## ⚙️ 4. PPO Hyperparameters 비교

| Hyperparameter | IsaacGym | 우리 환경 | 차이점 분석 |
|----------------|----------|-----------|------------|
| **Network Architecture** | [256, 128, 64] | [256, 128, 64] | ✅ 동일 |
| **Activation** | `elu` | `elu` | ✅ 동일 |
| **Learning Rate** | **5e-4** | **1e-4** | ❌ IsaacGym이 **5배 높음** |
| **Gamma** | 0.99 | 0.99 | ✅ 동일 |
| **Tau (GAE)** | 0.95 | 0.95 | ✅ 동일 |
| **Horizon Length** | **32** | **64** | ❌ 우리가 **2배 길음** |
| **Minibatch Size** | **16384** | **32768** | ❌ 우리가 **2배 큼** |
| **Mini Epochs** | 5 | **8** | ❌ 우리가 더 많음 |
| **Entropy Coef** | **0.0** | **0.001** | ❌ 우리만 사용 |
| **Critic Coef** | 4 | 4 | ✅ 동일 |
| **E-clip** | 0.2 | 0.2 | ✅ 동일 |
| **Grad Norm** | 1.0 | 1.0 | ✅ 동일 |
| **Reward Scale** | **1.0** | **0.01** | ❌ IsaacGym이 **100배 높음** |
| **Normalize Input** | True | True | ✅ 동일 |
| **Normalize Value** | True | True | ✅ 동일 |
| **Value Bootstrap** | **True** | **False** | ❌ 다름 |
| **Episode Length** | **300** | **750** | ❌ 우리가 **2.5배 길음** |
| **Num Envs** | **8192** | **4096** | ❌ IsaacGym이 **2배 많음** |

### 🔥 **중요한 차이점**

#### 1. **Learning Rate** (가장 중요!)
- IsaacGym: **5e-4** (0.0005)
- 우리: **1e-4** (0.0001)
- **문제**: 우리 LR이 너무 낮아서 학습이 느릴 수 있음

#### 2. **Reward Scale**
- IsaacGym: **1.0** (원본 reward 사용)
- 우리: **0.01** (reward를 100배 축소)
- **문제**: 우리 최대 reward 229가 → 2.29로 축소됨
  - IsaacGym은 최대 16.0을 그대로 사용
  - **Reward scale 불일치!**

#### 3. **Horizon Length**
- IsaacGym: **32 steps**
- 우리: **64 steps**
- **문제**: Longer horizon = 더 긴 credit assignment
  - 3-cube stacking이 더 어렵긴 하지만, 2배는 과할 수 있음

#### 4. **Episode Length**
- IsaacGym: **300 steps** (30초 @ 100Hz → 실제로는 5초 @ 60Hz)
- 우리: **750 steps** (15초 @ 50Hz)
- **분석**:
  - IsaacGym: dt=0.01667 (60Hz), substeps=2
  - 우리: dt=0.01 (100Hz), decimation=2 → 50Hz
  - Episode 길이는 비슷할 수 있음

#### 5. **Minibatch Size & Mini Epochs**
- IsaacGym: 16384 batch × 5 epochs = **81,920 updates/iteration**
- 우리: 32768 batch × 8 epochs = **262,144 updates/iteration**
- **문제**: 우리가 3.2배 더 많은 업데이트
  - Overfitting 가능성 증가

#### 6. **Number of Envs**
- IsaacGym: **8192 envs**
- 우리: **4096 envs**
- **문제**: Sample efficiency 차이
  - IsaacGym이 2배 많은 experience 수집

---

## 🏗️ 5. MDP Settings 비교

### Episode Length
| 환경 | Steps | Real Time | 비고 |
|------|-------|-----------|------|
| IsaacGym | 300 | ~5초 | 짧고 빠른 학습 |
| 우리 | 750 | ~15초 | 2.5배 길음 |

**분석**:
- IsaacGym은 빠르게 실패/성공 → 빠른 iteration
- 우리는 긴 episode → 느린 학습

### Reset 조건
#### IsaacGym
```python
# Line 756
reset_buf = torch.where(
    (progress_buf >= max_episode_length - 1) | (stack_reward > 0),
    torch.ones_like(reset_buf),
    reset_buf
)
```
- **Success 시 즉시 reset**: 성공하면 episode 종료
- 효율적인 학습

#### 우리 환경
```python
# stack_rl_env_cfg.py TerminationsCfg
time_out = DoneTerm(func=mdp.time_out, time_out=True)
cube_1_dropping = DoneTerm(...)  # Cube 떨어지면 종료
cube_2_dropping = DoneTerm(...)
cube_3_dropping = DoneTerm(...)
success = DoneTerm(func=mdp.cubes_stacked)
```
- **Success 및 failure 조건 모두 구현**
- Cube 떨어지면 종료 (좋음)

---

## 📊 6. 핵심 문제점 요약

### 🔴 **Critical Issues** (즉시 수정 필요)

#### 1. **Reward Scale 불일치** ⚠️⚠️⚠️
```yaml
# 현재 설정
reward_shaper:
  scale_value: 0.01  # 229 → 2.29로 축소

# IsaacGym
reward_shaper:
  scale_value: 1.0   # 16 → 16 그대로
```
**문제**:
- Reward scale 0.01 때문에 최대 reward가 2.29밖에 안 됨
- PPO의 value network가 작은 값만 학습
- Gradient도 작아짐

**해결책**:
```yaml
reward_shaper:
  scale_value: 1.0  # 또는 제거
```

#### 2. **Learning Rate 너무 낮음** ⚠️⚠️
```yaml
# 현재: 1e-4
learning_rate: 1e-4

# IsaacGym: 5e-4
learning_rate: 5e-4
```
**문제**: 학습 속도가 너무 느림

**해결책**:
```yaml
learning_rate: 5e-4  # IsaacGym과 동일하게
```

#### 3. **Observation 차원 너무 큼** ⚠️⚠️
- **현재**: 65-dim
- **IsaacGym**: 19-dim (3.4배 차이)

**문제**:
- Policy가 배워야 할 것이 너무 많음
- Sample efficiency 감소

**해결책**:
1. **상대 위치 사용**:
   ```python
   # 절대 좌표 대신
   cube_2_to_cube_1_pos = 3    # cube_2에서 cube_1으로의 벡터
   cube_3_to_cube_2_pos = 3    # cube_3에서 cube_2로의 벡터
   ```

2. **불필요한 observation 제거**:
   - `joint_vel`: OSC/IK-Rel에서 불필요
   - `actions`: Last action 불필요
   - `cube_dimensions`: 상수이므로 불필요 (reward에서만 사용)

3. **간소화된 observation (추정 ~25-30 dim)**:
   ```python
   cube_1_pos = 3            # 기준점
   cube_2_to_cube_1 = 3      # 상대 위치
   cube_3_to_cube_2 = 3      # 상대 위치
   cube_1_quat = 4           # Orientation
   cube_2_quat = 4
   cube_3_quat = 4
   eef_pos = 3
   eef_quat = 4
   gripper_pos = 2
   -------------------
   Total: ~30-dim (절반으로 감소!)
   ```

### 🟡 **Important Issues** (개선 권장)

#### 4. **Reward Terms 너무 많음**
- **현재**: 14개 terms
- **IsaacGym**: 4개 terms

**문제**: Reward shaping이 복잡하면 학습 불안정

**해결책**: Reward 단순화
```python
# 제안: 6-8개로 축소
# Phase 1
reaching_cube_2 = 1.0           # Dist reward
grasping_lifting_cube_2 = 5.0   # Grasp + Lift 통합
stacking_cube_2_on_1 = 30.0     # Stack success

# Phase 2 (masked)
reaching_cube_3 = 1.0           # Masked
grasping_lifting_cube_3 = 5.0   # Masked
stacking_cube_3_on_2 = 50.0     # Masked

# Success
task_success = 100.0

# Constraints (선택적)
cube_1_stable = 2.0             # 간소화
```

#### 5. **Horizon Length & Minibatch Size**
```yaml
# 현재
horizon_length: 64
minibatch_size: 32768
mini_epochs: 8

# IsaacGym
horizon_length: 32
minibatch_size: 16384
mini_epochs: 5
```

**문제**: 너무 많은 updates → overfitting 가능

**해결책**:
```yaml
horizon_length: 32         # IsaacGym과 동일
minibatch_size: 16384      # IsaacGym과 동일
mini_epochs: 5             # IsaacGym과 동일
```

#### 6. **Number of Environments**
```yaml
# 현재: 4096 envs
scene: ObjectTableSceneCfg(num_envs=4096, ...)

# IsaacGym: 8192 envs
numEnvs: 8192
```

**문제**: Sample 수집 속도가 느림

**해결책**: GPU 메모리가 허용하면
```python
scene: ObjectTableSceneCfg(num_envs=8192, ...)
```

### 🟢 **Minor Issues** (선택적)

#### 7. **Value Bootstrap**
```yaml
# 현재: False
value_bootstrap: False

# IsaacGym: True
value_bootstrap: True
```

**해결책**:
```yaml
value_bootstrap: True
```

#### 8. **Entropy Coefficient**
```yaml
# 현재: 0.001
entropy_coef: 0.001

# IsaacGym: 0.0
entropy_coef: 0.0
```

**해결책**:
```yaml
entropy_coef: 0.0  # Exploration은 action noise로 충분
```

---

## 🎯 7. 우선순위별 수정 사항

### 🔥 **즉시 수정** (학습이 안 되는 주원인)

1. **Reward Scale 수정**
   ```yaml
   # rl_games_ppo_cfg.yaml
   reward_shaper:
     scale_value: 1.0  # 0.01 → 1.0
   ```

2. **Learning Rate 증가**
   ```yaml
   learning_rate: 5e-4  # 1e-4 → 5e-4
   ```

3. **Observation 간소화** (가장 중요!)
   - 상대 좌표 사용
   - 불필요한 obs 제거 (joint_vel, actions, cube_dimensions)
   - 목표: **65-dim → ~30-dim**

### ⚠️ **권장 수정** (학습 속도/안정성 개선)

4. **Hyperparameter 조정**
   ```yaml
   horizon_length: 32       # 64 → 32
   minibatch_size: 16384    # 32768 → 16384
   mini_epochs: 5           # 8 → 5
   entropy_coef: 0.0        # 0.001 → 0.0
   value_bootstrap: True    # False → True
   ```

5. **Reward 단순화**
   - 14개 terms → 6-8개로 축소
   - Reaching + Grasping + Lifting 통합 고려

6. **Env 수 증가** (메모리 허용 시)
   ```python
   num_envs=8192  # 4096 → 8192
   ```

### 📝 **선택적** (3-cube 특화)

7. **Episode Length 조정**
   ```python
   episode_length_s = 10.0  # 15.0 → 10.0
   ```

---

## 📈 8. 예상 효과

### Before (현재)
- Observation: 65-dim
- Reward Scale: 0.01 (최대 2.29)
- Learning Rate: 1e-4
- Horizon: 64
- **결과**: 학습 거의 안 됨

### After (수정 후)
- Observation: ~30-dim (상대 좌표)
- Reward Scale: 1.0 (최대 229)
- Learning Rate: 5e-4
- Horizon: 32
- **예상 결과**:
  - Epoch 3000-5000: cube_2 stacking 성공
  - Epoch 8000-12000: cube_3 stacking 시도
  - Epoch 15000: 3-cube stacking 일부 성공

---

## 🔬 9. IsaacGym의 핵심 성공 요인

### 1. **Minimal Observation**
- 19-dim만으로 충분한 정보 제공
- **상대 좌표** 사용으로 task-relevant info에 집중

### 2. **Simple Reward Structure**
- 4개 reward terms로 자연스러운 curriculum 형성
- `torch.where`와 `max`로 자동 phase 전환

### 3. **Aggressive Learning**
- High LR (5e-4)
- No reward scaling (1.0)
- 빠른 초기 학습

### 4. **Efficient Sampling**
- 8192 envs로 많은 experience 수집
- Shorter episodes (300 steps)로 빠른 iteration

### 5. **Operational Space Control**
- Cartesian space에서 직접 제어
- 더 직관적인 action space

---

## 💡 10. 결론 및 제안

### 왜 우리 환경이 학습 안 되는가?

1. **Reward Scale 문제**: 0.01 scaling → reward 너무 작음
2. **Observation 복잡도**: 65-dim은 너무 많음, 상대 좌표 없음
3. **Learning Rate 낮음**: 5배 차이로 학습 속도 느림
4. **Reward 복잡도**: 14개 terms는 과도함

### 최소 수정 (Quick Fix)

```yaml
# rl_games_ppo_cfg.yaml에서 3줄만 수정
reward_shaper:
  scale_value: 1.0           # 0.01 → 1.0

learning_rate: 5e-4          # 1e-4 → 5e-4

horizon_length: 32           # 64 → 32
```

**+ Observation 간소화 (stack_rl_env_cfg.py)**
- 상대 좌표 추가
- 불필요한 obs 제거

### 제대로 된 수정 (Recommended)

1. **Observation 재설계** (가장 중요!)
   - IsaacGym처럼 상대 벡터 사용
   - 65-dim → ~30-dim

2. **Reward 단순화**
   - 14 terms → 6-8 terms

3. **Hyperparameter 정렬**
   - IsaacGym 설정 따라가기

4. **2-cube부터 시작**
   - 3-cube는 2-cube 성공 후 시도
   - Curriculum learning

---

## 🚀 11. 다음 단계

### Phase 1: Quick Fix (1-2일)
1. PPO config 수정 (reward_scale, LR, horizon)
2. 학습 시작 → 개선 확인

### Phase 2: Observation 재설계 (3-5일)
1. 상대 좌표 observation 추가
2. 불필요한 obs 제거
3. 재학습 및 비교

### Phase 3: Reward 단순화 (선택적)
1. IsaacGym 스타일로 reward 재설계
2. 4-6개 terms로 축소

### Phase 4: 2-Cube 환경 (검증용)
1. 2-cube stacking 환경 구현
2. IsaacGym과 직접 비교
3. 성공 후 3-cube로 확장

---

## 📚 참고: IsaacGym Reward 철학

IsaacGym의 reward는 **"What, not How"** 철학을 따름:

```python
# BAD: Micromanaging (우리 방식)
reaching_reward    # "이렇게 가라"
grasping_reward    # "이렇게 잡아라"
lifting_reward     # "이렇게 들어라"

# GOOD: Goal-oriented (IsaacGym 방식)
dist_reward        # "가까이 가라" (어떻게든)
lift_reward        # "들어라" (어떻게든)
align_reward       # "정렬해라" (어떻게든)
stack_reward       # "쌓아라" (어떻게든)
```

→ RL agent가 "How"를 스스로 학습하도록 자유도 부여!

---

**파일 생성 완료**: `DIFFERENCE_OUR_AND_ISAACGYM.md`
