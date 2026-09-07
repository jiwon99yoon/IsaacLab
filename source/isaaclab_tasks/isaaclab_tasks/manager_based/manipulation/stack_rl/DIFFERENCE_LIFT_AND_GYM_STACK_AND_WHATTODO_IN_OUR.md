# 세 환경 종합 비교: IsaacLab Lift vs IsaacGym Stack vs 우리 Stack RL

## 📚 분석 대상 환경

### 1️⃣ IsaacLab Lift (✅ 학습 성공)
- **태스크**: 1-cube lifting to goal
- **경로**: `/home/dyros/IsaacLab/source/isaaclab_tasks/.../lift`
- **특징**: IsaacLab 공식 튜토리얼, 학습 잘 됨
- **우리 환경 설계 시 참고한 베이스**

### 2️⃣ IsaacGym Cube Stack (✅ 학습 성공)
- **태스크**: 2-cube stacking
- **경로**: `/home/dyros/IsaacLab/IsaacGymEnvsTwk-main`
- **특징**: IsaacGym 레퍼런스, 학습 잘 됨

### 3️⃣ 우리 Stack RL (❌ 학습 안 됨)
- **태스크**: **3-cube stacking** (cube_1 base, cube_2 on 1, cube_3 on 2)
- **경로**: 현재 작업 중
- **문제**: 학습이 거의 안 됨

---

## 🔍 1. Observation Space 철학 비교

### IsaacLab Lift: **Robot Root Frame (상대 좌표)**

```python
# 33-dim 총 구성
joint_pos = 7                    # Joint positions
joint_vel = 7                    # Joint velocities
object_position = 3              # 🔑 Robot root frame 기준 상대 위치!
target_object_position = 7       # Command (pos + quat)
actions = 9                      # Last action
-------------------
Total: 33-dim
```

**핵심 특징**:
- ✅ **Robot root frame 사용**: `object_position_in_robot_root_frame()`
  ```python
  # lift/mdp/observations.py
  object_pos_b, _ = subtract_frame_transforms(
      robot.data.root_pos_w,
      robot.data.root_quat_w,
      object_pos_w
  )
  ```
- ✅ **Command-based**: Target position이 command로 주어짐
- ✅ **Task-agnostic**: Robot 기준 상대 좌표 → generalization 좋음
- ✅ **중간 복잡도**: 33-dim (joint info 포함)

### IsaacGym Stack: **Object-to-Object 상대 좌표**

```python
# 19-dim 총 구성 (OSC mode)
cubeA_quat = 4                   # 하단 큐브 orientation
cubeA_pos = 3                    # 하단 큐브 position (world)
cubeA_to_cubeB_pos = 3           # 🔑 큐브 간 상대 벡터!
eef_pos = 3                      # End-effector position
eef_quat = 4                     # End-effector orientation
q_gripper = 2                    # Gripper state
-------------------
Total: 19-dim
```

**핵심 특징**:
- ✅ **Object-to-Object 상대 벡터**: `cubeA_to_cubeB_pos`
  ```python
  # franka_cube_stack.py line 434
  "cubeA_to_cubeB_pos": self._cubeB_state[:, :3] - self._cubeA_state[:, :3]
  ```
- ✅ **Task-specific**: Stacking task에 최적화된 observation
- ✅ **매우 간결**: 19-dim (joint info 없음, OSC 사용)
- ✅ **Direct relevance**: Policy가 필요한 정보만 제공

### 우리 Stack RL: **Absolute World Coordinates (절대 좌표)**

```python
# 65-dim 총 구성
joint_pos = 7                    # Joint positions
joint_vel = 7                    # Joint velocities
cube_positions = 9               # 🔴 World frame 절대 좌표 (3 cubes)
cube_orientations = 12           # World frame quaternions
eef_pos = 3                      # End-effector position
eef_quat = 4                     # End-effector orientation
gripper_pos = 2                  # Gripper state
actions = 12                     # Last action
cube_dimensions = 9              # Cube sizes (constant!)
-------------------
Total: 65-dim
```

**핵심 문제**:
- ❌ **Absolute world coordinates**: Policy가 cube 간 관계를 스스로 학습해야 함
- ❌ **No relative information**: 상대 위치/벡터 없음
- ❌ **Redundant info**: cube_dimensions는 상수 (불필요)
- ❌ **Too complex**: 65-dim (IsaacGym의 3.4배, IsaacLab의 2배)

### 📊 Observation 비교표

| 특징 | IsaacLab Lift | IsaacGym Stack | 우리 Stack RL |
|------|--------------|----------------|--------------|
| **차원** | 33-dim | 19-dim | **65-dim** ❌ |
| **좌표계** | Robot root frame | Object-to-object | World frame |
| **상대 정보** | ✅ Robot 기준 | ✅ Cube 간 상대 | ❌ 없음 |
| **Joint info** | ✅ 포함 (IK-Rel) | ❌ 없음 (OSC) | ✅ 포함 (IK-Rel) |
| **Last action** | ✅ 9-dim | ❌ 없음 | ✅ 12-dim |
| **Cube size** | ❌ 없음 | ❌ 없음 | ✅ 9-dim (상수!) |
| **설계 철학** | Generalization | Task-specific | ??? |

---

## 💰 2. Reward Structure 철학 비교

### IsaacLab Lift: **Command-based Goal Tracking**

```python
# 4개 Reward Terms (+ 2 penalties)
reaching_object = 1.0              # EE to object distance
lifting_object = 15.0              # Binary: lifted or not
object_goal_tracking = 16.0        # 🔑 Distance to COMMAND goal (coarse)
object_goal_tracking_fine = 5.0    # 🔑 Distance to COMMAND goal (fine)

# Penalties
action_rate = -1e-4
joint_vel = -1e-4

# 최대 보상: ~37 (1 + 15 + 16 + 5)
```

**특징**:
- ✅ **Command-based**: Goal이 매 episode 랜덤하게 변경됨
- ✅ **2-stage goal tracking**: Coarse (std=0.3) → Fine (std=0.05)
- ✅ **Simple curriculum**: reach → lift → track_coarse → track_fine
- ✅ **Generalization**: 다양한 target position 학습

**Reward 함수 구조**:
```python
# object_goal_tracking (rewards.py line 48-67)
def object_goal_distance(env, std, minimal_height, command_name):
    # Command에서 목표 위치 가져오기
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )

    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)

    # 들었을 때만 reward 지급
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
```

### IsaacGym Stack: **Hardcoded Goal + Max Composition**

```python
# 4개 Reward Terms (franka_cube_stack.py line 707-758)
dist_reward = 0.1                  # EE to cubeA distance (tanh)
lift_reward = 1.5                  # Binary: cubeA lifted
align_reward = 2.0                 # 🔑 cubeA to cubeB alignment (들었을 때만)
stack_reward = 16.0                # 🔑 Success bonus

# 최대 보상: 16.0 (stack success)
# 또는: 0.1 + 1.5 + 2.0 = 3.6 (not stacked)
```

**특징**:
- ✅ **Hardcoded target**: cubeB position은 고정 (테이블 위)
- ✅ **Max composition**: `dist_reward = max(dist_reward, align_reward)`
- ✅ **Automatic curriculum**: dist → lift → align 자동 전환
- ✅ **Binary switch**: 성공하면 다른 rewards 무시

**Reward 함수 구조**:
```python
# compute_franka_reward (line 707-758)
# 1. Dist reward: EE to cubeA
dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)

# 2. Lift reward: Binary
cubeA_lifted = (cubeA_height - cubeA_size) > 0.04

# 3. Align reward: cubeA to target position (들었을 때만)
d_ab = torch.norm(states["cubeA_to_cubeB_pos"] + offset, dim=-1)
align_reward = (1 - torch.tanh(10.0 * d_ab)) * cubeA_lifted

# 4. Max composition (중요!)
dist_reward = torch.max(dist_reward, align_reward)

# 5. Success check
stack_reward = cubeA_align_cubeB & cubeA_on_cubeB & gripper_away

# 6. Final reward (torch.where로 분기)
rewards = torch.where(
    stack_reward,
    r_stack_scale * stack_reward,      # 성공: 16.0
    r_dist_scale * dist_reward +        # 실패: 0.1~3.6
    r_lift_scale * lift_reward +
    r_align_scale * align_reward
)
```

### 우리 Stack RL: **Explicit Phases + Masked Rewards**

```python
# 12개 Reward Terms (+ 2 penalties)

# Phase 1: cube_2 stacking
reaching_cube_2 = 3.0                    # EE to cube_2
grasping_cube_2 = 10.0                   # Grasp check
lifting_cube_2 = 5.0                     # Lifted after grasp
stacking_cube_2_on_1 = 35.0              # 🔑 Hybrid (5 dense + 30 sparse)

# Phase 2: cube_3 stacking (MASKED)
reaching_cube_3 = 3.0                    # Masked
grasping_cube_3 = 10.0                   # Masked
lifting_cube_3 = 5.0                     # Masked
stacking_cube_3_on_2 = 55.0              # Masked hybrid (5 + 50)

# Constraints
cube_1_stays_on_table = 2.0              # cube_1 stability
cube_1_movement_penalty = -1.0           # cube_1 movement
stack_stability = 1.0                    # When gripper open

# Success
task_success = 100.0                     # All 3 cubes stacked

# 최대 보상: 229 (53 + 73 + 3 + 100)
```

**특징**:
- ❌ **Explicit masking**: Phase 2 rewards manually masked
- ❌ **Complex hybrid**: One-time flags per stacking action
- ❌ **Many terms**: 12개 main rewards (너무 많음)
- ❌ **Large scale**: 최대 229 (IsaacGym의 14배)

### 📊 Reward 비교표

| 항목 | IsaacLab Lift | IsaacGym Stack | 우리 Stack RL |
|------|--------------|----------------|--------------|
| **Reward terms** | 4 + 2 penalties | 4 | 12 + 2 penalties |
| **최대 보상** | ~37 | 16 | **229** |
| **설계 철학** | Command-based | Goal-based | Phase-based |
| **Curriculum** | Implicit (2-stage) | Implicit (max) | Explicit (masking) |
| **Composition** | Additive | Max + Where | Additive |
| **성공 판정** | Goal distance | Stack check | 3-cube stack |

---

## ⚙️ 3. PPO Hyperparameters 비교

### 🔥 Critical Differences

| Hyperparameter | IsaacLab Lift | IsaacGym Stack | 우리 Stack RL | 분석 |
|----------------|--------------|----------------|--------------|------|
| **Learning Rate** | **1e-4** | **5e-4** | **1e-4** | IsaacGym이 5배 높음 |
| **Reward Scale** | **0.01** | **1.0** | **0.01** | ⚠️ 우리/IsaacLab은 100배 축소 |
| **Horizon Length** | 24 | 32 | **64** | 우리가 2배 길음 (합리적?) |
| **Minibatch Size** | 24576 | 16384 | **32768** | 우리가 가장 큼 |
| **Mini Epochs** | 8 | 5 | **8** | IsaacGym이 적음 |
| **Episode Length** | **5초** | **5초** | **15초** | 우리가 3배 길음 |
| **Num Envs** | 4096 | **8192** | 4096 | IsaacGym이 2배 |

### 📊 상세 비교

```yaml
# IsaacLab Lift (lift/config/.../rl_games_ppo_cfg.yaml)
learning_rate: 1e-4
reward_shaper:
  scale_value: 0.01          # 37 → 0.37
horizon_length: 24
minibatch_size: 24576
mini_epochs: 8
episode_length_s: 5.0
entropy_coef: 0.001
value_bootstrap: False

# IsaacGym Stack (cfg/train/FrankaCubeStackPPO.yaml)
learning_rate: 5e-4          # ⭐️ 5배 높음!
reward_shaper:
  scale_value: 1.0           # ⭐️ Scaling 안 함!
horizon_length: 32
minibatch_size: 16384
mini_epochs: 5
episode_length: 300 steps (~5초)
entropy_coef: 0.0
value_bootstrap: True

# 우리 Stack RL (stack_rl/.../rl_games_ppo_cfg.yaml)
learning_rate: 1e-4
reward_shaper:
  scale_value: 0.01          # ⚠️ 229 → 2.29
horizon_length: 64           # ⭐️ 2배 길음 (3-cube라서?)
minibatch_size: 32768
mini_epochs: 8
episode_length_s: 15.0       # ⭐️ 3배 길음
entropy_coef: 0.001
value_bootstrap: False
```

---

## 🤔 4. 설계 철학 분석: 왜 각각 잘 작동했는가?

### IsaacLab Lift의 성공 요인

#### 1. **Command-based Generalization**
```python
# CommandsCfg에서 매 episode 새 goal 생성
object_pose = mdp.UniformPoseCommandCfg(
    resampling_time_range=(5.0, 5.0),
    ranges=mdp.UniformPoseCommandCfg.Ranges(
        pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), ...
    ),
)
```
→ **다양한 target position** 학습 → Generalization 좋음

#### 2. **Robot Root Frame Observation**
```python
# 상대 좌표 사용
object_pos_b, _ = subtract_frame_transforms(
    robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w
)
```
→ **Task-agnostic representation** → 다른 task에도 적용 가능

#### 3. **2-Stage Goal Tracking**
```python
object_goal_tracking = 16.0        # Coarse (std=0.3)
object_goal_tracking_fine = 5.0    # Fine (std=0.05)
```
→ **점진적 정밀도 향상** → Smooth learning curve

#### 4. **Reward Scale 0.01의 의미**
- 최대 reward: 37 → 0.37 (scaled)
- **작지만 충분**: Value network가 0~1 범위 학습
- **Stable gradient**: 큰 reward로 인한 불안정성 방지

#### 5. **짧은 Episode (5초)**
- 빠른 trial-and-error
- Sample efficiency 증가

### IsaacGym Stack의 성공 요인

#### 1. **Relative Vector Observation**
```python
"cubeA_to_cubeB_pos": cubeB_pos - cubeA_pos
```
→ **Direct task relevance** → Policy가 바로 사용 가능

#### 2. **Max Composition Reward**
```python
dist_reward = torch.max(dist_reward, align_reward)
```
→ **Automatic curriculum** → 자동으로 phase 전환

#### 3. **Binary Success Switch**
```python
rewards = torch.where(
    stack_reward,
    r_stack_scale * stack_reward,      # Success
    r_dist + r_lift + r_align          # In progress
)
```
→ **Clear goal**: 성공하면 다른 reward 무시

#### 4. **No Reward Scaling (1.0)**
- 최대 reward: 16 (scaled 그대로)
- **직관적**: 사람이 이해하기 쉬움
- **High LR (5e-4)와 조합**: 빠른 학습

#### 5. **Minimal Observation (19-dim)**
- Joint info 없음 (OSC 사용)
- Task에 필요한 정보만
- **Sample efficiency** 증가

#### 6. **Many Envs (8192)**
- 2배 많은 experience 수집
- Parallel exploration

### 우리 Stack RL이 안 되는 이유

#### 1. **Reward Scale Mismatch** ⚠️⚠️⚠️
```python
reward_shaper:
  scale_value: 0.01
# 최대 reward: 229 → 2.29
```
**문제**:
- IsaacLab은 37 → 0.37 (합리적)
- IsaacGym은 16 → 16 (scaling 안 함)
- **우리는 229 → 2.29** (너무 큼에서 너무 작게!)

**왜 문제인가?**:
- Value network가 학습해야 할 range가 애매함
- Gradient 작아짐
- LR 1e-4와 조합 시 학습 속도 매우 느림

#### 2. **Observation Too Complex (65-dim)**
- IsaacLab: 33-dim (robot frame)
- IsaacGym: 19-dim (relative vectors)
- **우리: 65-dim (absolute world)**

**문제**:
- Policy network 크기 증가
- Sample efficiency 감소
- 상대 위치 정보 없음

#### 3. **Explicit Masking Complexity**
```python
# Masked rewards require manual condition check
def reaching_cube_3_masked(env, std):
    base_reward = object_ee_distance(...)
    mask = check_cube_2_on_cube_1(env)  # 수동 체크
    return base_reward * mask.float()
```
**문제**:
- IsaacGym은 자동 (max composition)
- 우리는 수동 (explicit masking)
- Complexity 증가

#### 4. **Task Difficulty**
- IsaacLab: 1 cube (단순)
- IsaacGym: 2 cubes (중간)
- **우리: 3 cubes (복잡)**

**하지만**: 이것만으로는 학습 안 되는 이유 설명 불가

---

## 💡 5. 두 접근 방식 중 어느 것이 우리에게 맞는가?

### 🎯 결론: **IsaacGym 스타일 + IsaacLab 일부 요소**

### 이유:

#### 1. **Task Nature**: Stacking (목표가 명확)
- IsaacLab: Command-based (목표가 변함)
- IsaacGym: Hardcoded goal (목표 고정)
- **우리**: 3-cube stacking → **목표 고정** (IsaacGym 스타일)

#### 2. **Observation Philosophy**
- IsaacLab: Robot root frame (generalization)
- IsaacGym: Object-to-object (task-specific)
- **우리**: 3-cube stacking → **Object-to-object** (IsaacGym 스타일)

#### 3. **Reward Composition**
- IsaacLab: Additive (command tracking)
- IsaacGym: Max + Where (automatic curriculum)
- **우리**: 복잡한 phase → **Automatic curriculum** (IsaacGym 스타일)

#### 4. **하지만 IsaacLab에서 배울 점**:
- ✅ Reward scale 0.01 (합리적 scaling)
- ✅ 2-stage reward (coarse → fine)
- ✅ Joint info 포함 (IK-Rel 사용 시)

---

## 🎯 6. 우리 환경 수정 권장사항

### 🔥 **Priority 1: Critical (즉시 수정)**

#### 1. **Reward Scale 재조정** ⚠️⚠️⚠️

**현재 문제**:
```yaml
reward_shaper:
  scale_value: 0.01
# 최대 229 → 2.29 (너무 작음)
```

**옵션 A: IsaacGym 스타일 (No scaling)**
```yaml
reward_shaper:
  scale_value: 1.0
# 최대 229 → 229 (그대로)
```
✅ **추천**: 직관적, 큰 reward로 빠른 학습

**옵션 B: IsaacLab 스타일 (Moderate scaling)**
```yaml
reward_shaper:
  scale_value: 0.1
# 최대 229 → 22.9 (10배 감소)
```
✅ **추천**: 안정적, value network 학습 용이

**옵션 C: 현재 유지 + Reward 축소**
```yaml
# Reward weights 자체를 줄임
reaching_cube_2 = 0.3   # 3.0 → 0.3
grasping_cube_2 = 1.0   # 10.0 → 1.0
...
# 최대: ~23 → 0.23 (scale 0.01 적용)
```
❌ **비추천**: 복잡도 증가

**→ 권장: 옵션 A 또는 B** (A가 더 직관적)

#### 2. **Learning Rate 증가**

**사용자 의견**: "learning_rate의 경우는 바꿔도 될 거 같긴 한데"

✅ **동의합니다!**

```yaml
# 현재
learning_rate: 1e-4

# 권장
learning_rate: 5e-4  # IsaacGym과 동일
```

**이유**:
- IsaacGym: 5e-4로 성공
- 우리 reward scale 높이면 → 더 높은 LR 필요
- 빠른 학습

#### 3. **Observation 간소화** (상대 좌표 추가)

**현재 (65-dim)**:
```python
cube_positions = 9           # World frame (절대)
cube_orientations = 12
...
cube_dimensions = 9          # 상수!
```

**권장: IsaacGym 스타일 (상대 벡터)**
```python
# 간소화된 observation (~35-40 dim)
cube_1_pos = 3               # Base cube (world frame)
cube_2_to_cube_1 = 3         # 🔑 Relative vector!
cube_3_to_cube_2 = 3         # 🔑 Relative vector!
cube_1_quat = 4
cube_2_quat = 4
cube_3_quat = 4
eef_pos = 3
eef_quat = 4
gripper_pos = 2
actions = 9                  # 또는 제거
-------------------
Total: ~35-39 dim (40% 감소!)
```

**제거 가능**:
- ❌ `joint_vel`: IK-Rel에서 덜 중요
- ❌ `cube_dimensions`: 상수 (reward 함수에서만 사용)
- ❌ `actions` (선택적): 일부 환경에서는 불필요

### ⚠️ **Priority 2: Important (개선 권장)**

#### 4. **Reward Structure 단순화**

**현재**: 12 terms (너무 많음)

**권장**: IsaacGym 스타일로 단순화
```python
# 5-7 terms로 축소

# Phase 1 (cube_2)
dist_reward_2 = max(reaching_2, align_2) # 🔑 Max composition!
lift_reward_2 = lifting_2
stack_reward_2 = success_2               # Binary + Where

# Phase 2 (cube_3, masked)
dist_reward_3 = max(reaching_3, align_3) # Masked
lift_reward_3 = lifting_3                # Masked
stack_reward_3 = success_3               # Masked

# Final
task_success = 100.0

# Constraints (선택적)
cube_1_stable = 2.0
```

**Masking 자동화 (IsaacGym 스타일)**:
```python
# Explicit masking 대신 reward 함수 내부에서 처리
def dist_reward_cube_3_auto_masked(env, std):
    # Phase 2 조건 자동 체크
    cube_2_stacked = check_stack_condition(env, "cube_2", "cube_1")

    base_reward = object_ee_distance(env, std, "cube_3")

    # 자동 masking (내부에서)
    return base_reward * cube_2_stacked.float()
```

#### 5. **Horizon Length 유지 (사용자 의견 동의)**

**사용자 의견**: "horizon_length의 경우 우린 cube 2개를 쌓아야 하니까 2배 길어진 게 괜찮을 것 같아"

✅ **매우 합리적인 의견입니다!**

```yaml
# IsaacGym: 2-cube stacking, horizon=32
# 우리: 3-cube stacking (2개 쌓기), horizon=64

horizon_length: 64  # ✅ 유지 (2배 harder task)
```

**이유**:
- Phase 1: cube_2 stacking (~30 steps)
- Phase 2: cube_3 stacking (~30 steps)
- Total: ~60 steps 필요
- **64는 합리적**

#### 6. **Episode Length 단축 (고려)**

**현재**: 15초 (750 steps @ 50Hz)
**IsaacGym/IsaacLab**: 5초

**제안**: 10초로 단축
```python
episode_length_s = 10.0  # 15 → 10
# 500 steps @ 50Hz
```

**이유**:
- 3-cube는 2-cube보다 2배 오래 걸림
- 5초는 너무 짧음
- **10초가 적당** (2배)

### 🟢 **Priority 3: Optional (선택적)**

#### 7. **Minibatch Size & Mini Epochs 조정**

**현재**:
```yaml
minibatch_size: 32768
mini_epochs: 8
# Total updates: 262,144 per iteration
```

**IsaacGym**:
```yaml
minibatch_size: 16384
mini_epochs: 5
# Total updates: 81,920 per iteration
```

**제안**: IsaacGym 따라가기
```yaml
minibatch_size: 16384  # 32768 → 16384
mini_epochs: 5         # 8 → 5
```

**이유**: Overfitting 방지

#### 8. **Num Envs 증가 (메모리 허용 시)**

```yaml
scene: ObjectTableSceneCfg(num_envs=8192, ...)  # 4096 → 8192
```

#### 9. **Entropy Coef 제거**

```yaml
entropy_coef: 0.0  # 0.001 → 0.0 (IsaacGym 스타일)
```

#### 10. **Value Bootstrap 활성화**

```yaml
value_bootstrap: True  # False → True (IsaacGym 스타일)
```

---

## 📊 7. Reward Weight 비교 및 조정

### 현재 우리 Reward Weights

```python
# Phase 1
reaching_cube_2 = 3.0
grasping_cube_2 = 10.0
lifting_cube_2 = 5.0
stacking_cube_2_on_1 = 35.0 (5 dense + 30 sparse)

# Phase 2
reaching_cube_3 = 3.0
grasping_cube_3 = 10.0
lifting_cube_3 = 5.0
stacking_cube_3_on_2 = 55.0 (5 + 50)

# Constraints
cube_1_stable = 2.0
cube_1_penalty = -1.0
stack_stability = 1.0

# Success
task_success = 100.0

# 최대: 229
```

### IsaacGym Reward Weights (yaml에서)

```yaml
distRewardScale: 0.1      # Reaching
liftRewardScale: 1.5      # Lifting
alignRewardScale: 2.0     # Alignment
stackRewardScale: 16.0    # Success

# 최대: 16
```

### IsaacLab Lift Reward Weights

```python
reaching_object = 1.0
lifting_object = 15.0
object_goal_tracking = 16.0
object_goal_tracking_fine = 5.0

# 최대: ~37
```

### 🎯 Reward Scale 재조정 (사용자 질문 답변)

**사용자 질문**: "또 yaml file의 경우 우리의 현재 reward weight 값도 고려해야지? ... gym에서의 reward weight와 비교해서 yaml file 수정해야 한다고 한 거 맞니?"

✅ **네, 맞습니다!**

**핵심 이슈**: Reward scale과 reward weight의 상호작용

| 환경 | Max Reward | Reward Scale | Scaled Max | 분석 |
|------|-----------|--------------|-----------|------|
| IsaacGym | 16 | 1.0 | **16** | Scaling 안 함 |
| IsaacLab | 37 | 0.01 | **0.37** | Moderate |
| 우리 (현재) | 229 | 0.01 | **2.29** | 너무 큼 |

**문제**:
1. 우리 reward weights가 IsaacGym의 14배 (229 vs 16)
2. 하지만 scale 0.01 적용 → 2.29 (IsaacGym 16의 1/7)
3. **결과**: 너무 작아짐!

### 🔧 해결책 (3가지 옵션)

#### 옵션 1: Reward Scale만 수정 (추천 ⭐️)

```yaml
# yaml file
reward_shaper:
  scale_value: 0.1  # 0.01 → 0.1

# 효과: 229 → 22.9 (IsaacGym 16과 비슷)
```

**장점**:
- 간단
- Reward weight 그대로 유지 (의미 보존)
- IsaacGym과 비슷한 scale

#### 옵션 2: Reward Weight 감소 + Scale 유지

```python
# stack_rl_env_cfg.py
# 모든 weight를 1/10으로
reaching_cube_2 = 0.3    # 3.0 → 0.3
grasping_cube_2 = 1.0    # 10.0 → 1.0
...
task_success = 10.0      # 100.0 → 10.0

# 최대: ~23
# Scale 0.01 적용 → 0.23 (IsaacLab과 비슷)
```

**단점**:
- 코드 수정 많음
- 의미가 애매해짐

#### 옵션 3: IsaacGym 스타일 (No scaling)

```yaml
# yaml file
reward_shaper:
  scale_value: 1.0  # 0.01 → 1.0

# 효과: 229 → 229 (큰 reward)
```

**장점**:
- 직관적
- IsaacGym과 동일한 철학

**단점**:
- Large reward → gradient 불안정 가능성
- 하지만 IsaacGym은 16으로 성공했으니 229도 가능할 듯

### 📊 권장 순서

1. **먼저 시도**: 옵션 1 (scale=0.1) + LR 증가
2. **안 되면**: 옵션 3 (scale=1.0) + LR 증가
3. **마지막**: 옵션 2 (weight 감소)

---

## 🎯 8. 최종 권장 수정사항 (우선순위별)

### 🔴 **즉시 적용** (필수)

#### 1. Reward Scale 수정
```yaml
# stack_rl/.../rl_games_ppo_cfg.yaml
reward_shaper:
  scale_value: 0.1  # 0.01 → 0.1 (또는 1.0)
```

#### 2. Learning Rate 증가
```yaml
learning_rate: 5e-4  # 1e-4 → 5e-4
```

#### 3. Observation 간소화 (중기 목표)
```python
# stack_rl_env_cfg.py ObservationsCfg
# 상대 벡터 추가, 불필요한 obs 제거
# 목표: 65-dim → ~35-40-dim
```

### 🟡 **강력 권장** (개선)

#### 4. Horizon Length 유지
```yaml
horizon_length: 64  # ✅ 유지 (사용자 의견 동의)
```

#### 5. Episode Length 단축
```python
episode_length_s = 10.0  # 15 → 10
```

#### 6. Minibatch 조정
```yaml
minibatch_size: 16384  # 32768 → 16384
mini_epochs: 5         # 8 → 5
```

### 🟢 **선택적** (Fine-tuning)

#### 7. Reward Structure 단순화
- 12 terms → 6-8 terms
- Max composition 사용
- Automatic masking

#### 8. Hyperparameter 정렬
```yaml
entropy_coef: 0.0           # 0.001 → 0.0
value_bootstrap: True       # False → True
```

---

## 📝 9. Quick Start (바로 적용 가능한 수정)

### Step 1: yaml 파일 3줄 수정 (5분)

```yaml
# stack_rl/config/franka/agents/rl_games_ppo_cfg.yaml

# Line 60 수정
reward_shaper:
  scale_value: 0.1           # 0.01 → 0.1 ⭐️⭐️⭐️

# Line 64 수정
learning_rate: 5e-4          # 1e-4 → 5e-4 ⭐️⭐️

# Line 77-78 수정
horizon_length: 64           # ✅ 유지 (합리적)
minibatch_size: 16384        # 32768 → 16384 ⭐️
```

### Step 2: 학습 시작

```bash
./isaaclab.sh -p scripts/train.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 15000
```

### Step 3: Tensorboard 모니터링

```bash
tensorboard --logdir logs/rl_games/stack_cube_franka
```

**기대 효과**:
- Epoch 1000: Reaching reward 증가 시작
- Epoch 3000: cube_2 stacking 일부 성공
- Epoch 8000: cube_3 phase 활성화
- Epoch 15000: 3-cube stacking 성공률 증가

---

## 🔬 10. 왜 이 수정이 효과적인가?

### IsaacGym의 성공 비결 3가지

1. **Minimal Observation (19-dim)** → 우리는 65-dim (나중에 개선)
2. **Simple Reward (4 terms)** → 우리는 12 terms (나중에 개선)
3. **Proper Scaling (reward_scale=1.0, LR=5e-4)** → **우리가 지금 고칠 부분!**

### IsaacLab의 성공 비결

1. **Reasonable Scaling (0.01)** → max 37 → 0.37 (합리적)
2. **Robot Root Frame** → Generalization
3. **Command-based** → Variety

### 우리가 지금 고치는 것

✅ **Reward scale 0.01 + reward 229 = 2.29 (너무 작음)**
→ Scale 0.1로 → **22.9** (IsaacGym 16과 비슷)

✅ **LR 1e-4 (너무 낮음)**
→ 5e-4로 → **IsaacGym과 동일**

✅ **Horizon 64 유지**
→ **3-cube task에 합리적** (사용자 의견 동의)

---

## 💬 11. 요약 (TL;DR)

### 🔴 핵심 문제 3가지

1. **Reward Scale**: 0.01 × 229 = 2.29 (너무 작음)
2. **Learning Rate**: 1e-4 (IsaacGym의 1/5)
3. **Observation**: 65-dim (IsaacGym의 3.4배)

### ✅ 즉시 해결 (yaml 3줄)

```yaml
reward_shaper:
  scale_value: 0.1     # 0.01 → 0.1
learning_rate: 5e-4    # 1e-4 → 5e-4
minibatch_size: 16384  # 32768 → 16384
```

### 🎯 철학 선택

- **IsaacGym 스타일**: Task-specific, simple, effective
- **IsaacLab 스타일**: General, command-based, moderate
- **우리 선택**: **IsaacGym 기반 + IsaacLab scaling**

### 📊 Horizon Length (사용자 질문)

✅ **64 유지 권장**
- 이유: 3-cube (2개 쌓기) = 2× harder than 2-cube
- IsaacGym (2-cube): horizon=32
- 우리 (3-cube): horizon=64 (2배, 합리적!)

### 🚀 예상 결과

- **Before**: 학습 거의 안 됨, reward ~0.1
- **After**: Epoch 5000-10000에 cube_2 stacking 성공, Epoch 15000에 일부 3-cube 성공

---

## 📚 참고: 각 환경의 설계 의도

### IsaacLab Lift
**의도**: General-purpose manipulation framework
- Command-based → Variety
- Robot frame → Generalization
- Moderate complexity → Tutorial-friendly

### IsaacGym Stack
**의도**: Fast, efficient RL for specific task
- Task-specific observation → Sample efficiency
- Simple reward → Easy to tune
- High throughput (8192 envs) → Fast iteration

### 우리 Stack RL
**의도**: 3-cube stacking (더 어려운 task)
- 문제: IsaacLab 기반으로 시작했으나, IsaacGym 스타일이 더 적합
- 해결: Hybrid approach (IsaacGym observation + IsaacLab scaling)

---

**파일 생성 완료**: `DIFFERENCE_LIFT_AND_GYM_STACK_AND_WHATTODO_IN_OUR.md`

**다음 단계**: yaml 파일 3줄 수정 후 학습 시작!
