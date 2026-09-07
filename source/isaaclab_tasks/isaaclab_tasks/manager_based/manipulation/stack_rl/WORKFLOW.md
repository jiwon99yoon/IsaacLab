# Stack-RL Workflow Documentation

## Project Overview

This document describes the development of **Stack-RL**, a pure Reinforcement Learning environment for 3-cube sequential stacking tasks in IsaacLab. This environment serves as a baseline for future integration with **CuRobo** motion planning.

### Project Context
- **Parent Project**: Hybrid RL-CuRobo approach for manipulation tasks
- **Reference**: CuRobo SkillGen (`generate_dataset.py`) for Imitation Learning
- **Goal**: Compare pure RL approach vs. CuRobo-assisted RL approach

---

## Project Structure

```
stack_rl/
├── WORKFLOW.md                          # This file
├── stack_rl_env_cfg.py                  # Base environment configuration
├── __init__.py                          # Package initialization
├── mdp/                                 # MDP components
│   ├── __init__.py                      # Imports stack observations/terminations + custom rewards
│   └── rewards.py                       # Custom reward functions for stacking
└── config/                              # Robot-specific configurations
    ├── __init__.py
    └── franka/                          # Franka Panda robot configs
        ├── __init__.py                  # Environment registration
        ├── joint_pos_env_cfg.py        # Joint position control
        ├── ik_rel_env_cfg.py           # IK relative control
        └── agents/
            └── rl_games_ppo_cfg.yaml   # PPO hyperparameters
```

---

## What We've Built

### 1. Pure RL Framework for 3-Cube Stacking

Created a complete Manager-based RL environment for training a Franka Panda robot to stack 3 cubes sequentially.

#### Task Description
- **Phase 1**: Reach, grasp, and lift cube_1 (blue, base cube)
- **Phase 2**: Reach, grasp cube_2 (red), and stack it on cube_1
- **Phase 3**: Reach, grasp cube_3 (green), and stack it on cube_2
- **Success**: All 3 cubes stacked + gripper open (released)

#### Key Features
- **3 Cubes**: blue_block (cube_1), red_block (cube_2), green_block (cube_3)
- **Randomized Start Positions**: x ∈ [0.4, 0.6], y ∈ [-0.10, 0.10], with minimum 0.1m separation
- **Dense Reward Shaping**: Progressive rewards for reaching → grasping → stacking
- **Episode Length**: 15 seconds (750 steps @ 50Hz control)
- **Parallel Environments**: 4096 environments for fast learning

---

## Implementation Details

### 1. Environment Configuration (`stack_rl_env_cfg.py`)

#### Scene Setup
- Robot: Franka Panda (7-DOF arm + parallel gripper)
- Table: Seattle Lab Table
- Cubes: 3x rigid body cubes with physics properties
- Frame Transformer: Tracks end-effector pose

#### Observations (56-dim)
- Joint positions (relative)
- Joint velocities (relative)
- Cube positions (3x 3D = 9-dim)
- Cube orientations (3x 4D quaternions = 12-dim)
- End-effector position (3D)
- End-effector orientation (4D quaternion)
- Gripper position (2-dim for parallel gripper)
- Previous actions

#### Actions
- **Joint Position Control**: 7 arm joints + 1 binary gripper (8-dim)
- **IK Relative Control**: 6-DOF pose delta + binary gripper (7-dim)

#### Rewards (9 terms)
```python
# Phase 1: Lift cube_1
reaching_cube_1     = 1.0   # Tanh kernel distance reward
grasping_cube_1     = 10.0  # Binary grasp reward
lifting_cube_1      = 15.0  # Height-based reward

# Phase 2: Stack cube_2 on cube_1
reaching_cube_2     = 1.0
grasping_cube_2     = 10.0
stacking_cube_2_on_1 = 30.0  # XY alignment + height + gripper open

# Phase 3: Stack cube_3 on cube_2
reaching_cube_3     = 1.0
grasping_cube_3     = 10.0
stacking_cube_3_on_2 = 50.0  # Higher weight for final stack

# Success bonus
task_success        = 100.0  # All 3 cubes stacked
```

#### Terminations
- Time out (15 seconds)
- Any cube drops below table height (-0.05m)
- Success (all 3 cubes stacked)

#### PhysX Configuration (Important!)
```python
gpu_total_aggregate_pairs_capacity = 32 * 1024  # Increased for 3-cube stacking
```
**Note**: This was increased from 16K to 32K to prevent PhysX errors during training with 3 cubes.

---

### 2. Reward Functions (`mdp/rewards.py`)

Implemented custom reward functions:

- `object_ee_distance()`: Smooth reaching reward using tanh kernel
- `object_is_lifted()`: Binary reward for lifting object above threshold
- `object_grasped_reward()`: Detects successful grasp (EE-object distance + gripper closed)
- `object_stacked_reward()`: Checks XY alignment, height, and gripper open
- `cubes_stacked_success()`: Success bonus for 2-cube stack (legacy)
- `three_cubes_stacked_success()`: Success bonus for 3-cube sequential stack
  - Verifies cube_2 on cube_1
  - Verifies cube_3 on cube_2
  - Ensures gripper is open

---

### 3. Franka Robot Configuration (`config/franka/joint_pos_env_cfg.py`)

#### Cube Spawning
```python
cube_1: blue_block  @ [0.4, 0.0, 0.0203]
cube_2: red_block   @ [0.55, 0.05, 0.0203]
cube_3: green_block @ [0.60, -0.1, 0.0203]
```

#### Randomization Events
- Initial arm pose: Home position with slight randomization (Gaussian noise σ=0.02)
- Cube positions: Randomized within workspace with minimum separation
- Yaw rotation: Random ∈ [-1.0, 1.0] radians

#### End-Effector Frame Transformer
Tracks 3 frames:
1. `end_effector`: Center of panda_hand + 10.34cm offset
2. `tool_rightfinger`: Right finger tip
3. `tool_leftfinger`: Left finger tip

---

### 4. PPO Hyperparameters (`agents/rl_games_ppo_cfg.yaml`)

```yaml
max_epochs: 100000           # Extended from 10000 for thorough training
horizon_length: 64           # Longer than lift (24) for multi-phase task
minibatch_size: 32768        # Large batches for stable learning
mini_epochs: 8               # PPO update iterations
learning_rate: 1e-4          # Conservative for long training
gamma: 0.99                  # Discount factor
tau: 0.95                    # GAE parameter
entropy_coef: 0.001          # Encourage exploration
network: [256, 128, 64]      # 3-layer MLP
activation: elu              # Smooth activation
```

**Key Change**: `max_epochs` increased to **100000** to allow sufficient training time for the complex 3-cube stacking task.

---

### 5. Registered Environments

Four environments registered in `config/franka/__init__.py`:

1. **Isaac-Stack-RL-Cube-Franka-v0** (Joint Position Control)
2. **Isaac-Stack-RL-Cube-Franka-Play-v0** (Joint Pos, 50 envs for visualization)
3. **Isaac-Stack-RL-Cube-Franka-IK-Rel-v0** (IK Relative Control)
4. **Isaac-Stack-RL-Cube-Franka-IK-Rel-Play-v0** (IK Rel, 50 envs for visualization)

---

## How to Run

### Training (Headless, Fast)
```bash
cd /home/dyros/IsaacLab

# Joint Position Control
python scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Stack-RL-Cube-Franka-v0 \
  --num_envs 4096 \
  --headless

# IK Relative Control
python scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Stack-RL-Cube-Franka-IK-Rel-v0 \
  --num_envs 4096 \
  --headless
```

### Training (With Visualization)
```bash
# Fewer environments for better visualization
python scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Stack-RL-Cube-Franka-v0 \
  --num_envs 512
```

### Playing Trained Policy
```bash
python scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Stack-RL-Cube-Franka-Play-v0 \
  --num_envs 50 \
  --checkpoint /path/to/checkpoint.pth
```

### Expected Training Time
- **Hardware**: RTX 3090/4090 or similar
- **FPS**: ~85k-90k steps/sec (with 4096 envs)
- **Training Steps**: 100k epochs × 262144 frames/epoch = 26.2B frames
- **Wall Clock Time**: ~7-10 hours for full training

---

## Development History

### Initial Creation (2025-12-09)
1. Created `stack_rl` folder structure based on `lift` environment format
2. Initially implemented 2-cube stacking (cube_1 + cube_2)
3. Registered environments and tested basic functionality

### Bug Fix: Missing cube_3
**Problem**: Environment crashed with `KeyError: "Scene entity with key 'cube_3' not found"`

**Root Cause**: Original stack environment uses 3 cubes, but our initial implementation only created 2.

**Solution**: Updated all files to support 3-cube sequential stacking:
- Added `cube_3` RigidObjectCfg (green_block) to `joint_pos_env_cfg.py`
- Updated reward terms for Phase 3 (reach/grasp/stack cube_3)
- Added `three_cubes_stacked_success()` function
- Added `cube_3_dropping` termination

### Bug Fix: PhysX Error
**Problem**: Training crashed around epoch 11-12 with:
```
PhysX error: The application needs to increase
PxGpuDynamicsMemoryConfig::totalAggregatePairsCapacity to 17365
```

**Root Cause**: 3 cubes create more physics interactions than GPU memory allocation (16K pairs).

**Solution**: Increased `gpu_total_aggregate_pairs_capacity` from 16K → 32K in `stack_rl_env_cfg.py`

---

## Known Issues & Warnings (Safe to Ignore)

### Training Warnings
These warnings appear during training but don't affect performance:

```
FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated
```
→ PyTorch API deprecation warning in RL Games library. No action needed.

```
[Warning] [carb] Client gpu.foundation.plugin has acquired... 100 times
```
→ Omniverse internal optimization suggestion. No impact on training.

```
imDefLkup.c,355: The key event is already fabricated
```
→ X11 display warning. Harmless.

### Pylance Type Warnings
```
"gripper_joint_names" attribute unknown
"gripper_open_val" attribute unknown
```
→ These attributes are dynamically added in robot-specific configs. Runtime behavior is correct.

---

## Future Work: CuRobo Integration

### Current Status: Pure RL Baseline
The current `stack_rl` environment is a **pure Reinforcement Learning** approach without any motion planning assistance. This serves as our **baseline** for comparison.

### Planned: Hybrid RL-CuRobo Approach

#### Inspiration: CuRobo SkillGen
Based on `isaaclab_mimic/curobo/generate_dataset.py`:
- CuRobo generates expert trajectories for Imitation Learning
- Uses motion planning for smooth, collision-free paths
- Fast GPU-accelerated trajectory optimization

#### Integration Strategy

We will create a **hybrid approach** that combines:

1. **CuRobo Motion Planning** (for reaching/approaching phases)
   - Phase 1a: Plan trajectory to approach cube_1
   - Phase 2a: Plan trajectory to approach cube_2
   - Phase 3a: Plan trajectory to approach cube_3

2. **RL Fine-Tuning** (for contact-rich manipulation)
   - Phase 1b: RL learns to grasp and lift cube_1
   - Phase 2b: RL learns to grasp and stack cube_2
   - Phase 3b: RL learns to grasp and stack cube_3

#### Proposed Architecture

```
┌─────────────────────────────────────────────────────────┐
│  High-Level Policy (RL)                                 │
│  Decides: Which cube to target? Grasp or release?      │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌───────────────┐   ┌──────────────────┐
│ CuRobo Planner│   │ RL Low-Level     │
│ (Reaching)    │   │ (Manipulation)   │
│               │   │                  │
│ - Collision-  │   │ - Grasp control  │
│   free paths  │   │ - Force control  │
│ - Smooth      │   │ - Contact-rich   │
│   trajectories│   │   interactions   │
└───────────────┘   └──────────────────┘
```

#### Implementation Plan

1. **Create `stack_curobo` Environment**
   - Copy `stack_rl` as starting point
   - Add CuRobo motion planner interface
   - Implement mode switching (planning vs. RL control)

2. **Modify Action Space**
   ```python
   # Option 1: Residual RL on CuRobo trajectories
   action = curobo_trajectory + rl_residual

   # Option 2: Mode switching
   if mode == "reach":
       action = curobo_planner.plan(target_pose)
   elif mode == "manipulate":
       action = rl_policy(obs)
   ```

3. **Modify Reward Structure**
   - Reduce penalty for CuRobo-guided reaching
   - Focus RL rewards on manipulation quality
   - Add reward for following CuRobo suggestions

4. **Training Procedure**
   - Pre-train RL policy on reaching tasks
   - Gradually enable CuRobo assistance
   - Curriculum: CuRobo → Hybrid → Pure RL

#### Expected Benefits
- **Faster Training**: CuRobo handles reaching, RL focuses on manipulation
- **Better Sample Efficiency**: Guided exploration reduces random flailing
- **Higher Success Rate**: Motion planning ensures collision-free paths
- **Smoother Trajectories**: Combines optimality of planning + adaptability of RL

---

## Comparison Analysis Plan

### Metrics to Compare

#### 1. Training Efficiency
- **Sample Efficiency**: Frames to reach 50% success rate
- **Training Time**: Wall-clock hours to convergence
- **Computational Cost**: GPU hours, memory usage

#### 2. Task Performance
- **Success Rate**: % of episodes with all 3 cubes stacked
- **Episode Length**: Average steps to completion
- **Stack Quality**:
  - XY alignment error (mm)
  - Height alignment error (mm)
  - Stack stability (time until collapse)

#### 3. Trajectory Quality
- **Smoothness**: Jerk (third derivative of position)
- **Energy Efficiency**: Cumulative joint torques
- **Collision Avoidance**: Self-collision and cube-collision events
- **Execution Time**: Time to complete successful stacks

#### 4. Robustness
- **Generalization**: Performance on randomized start positions
- **Perturbation Recovery**: Success rate with external disturbances
- **Sim-to-Real Transfer**: (Future) Real robot performance

### Experiment Design

| Experiment | Environment | CuRobo? | Num Envs | Max Epochs | Notes |
|------------|-------------|---------|----------|------------|-------|
| **Baseline** | stack_rl | ❌ No | 4096 | 100000 | Pure RL (current) |
| **Hybrid-1** | stack_curobo | ✅ Full guidance | 4096 | 50000 | CuRobo for reaching |
| **Hybrid-2** | stack_curobo | ⚠️ Partial | 4096 | 75000 | Curriculum: CuRobo→RL |
| **IL Baseline** | stack (original) | ✅ Expert demos | 4096 | N/A | Pure Imitation Learning |

### Evaluation Protocol

1. **Train Each Method**
   - Same hardware (NVIDIA RTX GPU)
   - Same random seeds (42, 123, 456)
   - Same environment randomization

2. **Evaluate on Test Set**
   - 1000 episodes with fixed random seeds
   - Record all metrics (success rate, trajectory quality, etc.)

3. **Statistical Analysis**
   - Mean ± standard deviation across 3 seeds
   - Paired t-tests for significance
   - Learning curves with confidence intervals

4. **Qualitative Analysis**
   - Video recordings of successful/failed episodes
   - Failure mode analysis (what goes wrong?)
   - Trajectory visualization (path through workspace)

---

## Results Tracking

### Current Status: Training Pure RL Baseline

**Training Started**: 2025-12-09 22:35:51
**Configuration**: Isaac-Stack-RL-Cube-Franka-v0, 4096 envs, 100k max epochs
**Expected Completion**: ~10 hours

#### Preliminary Results (First 16 epochs)
- **FPS**: ~85k-90k steps/sec (after PhysX fix)
- **Status**: Training stable, no crashes
- **Observations**:
  - Initial random policy, low rewards
  - PhysX error fixed at epoch 12 (GPU memory increase)
  - Need to monitor convergence over full training

### Results Summary (To Be Updated)

| Method | Success Rate | Avg Episode Length | Training Time | Sample Efficiency |
|--------|--------------|-------------------|---------------|-------------------|
| Pure RL (stack_rl) | TBD | TBD | TBD | TBD |
| Hybrid CuRobo | TBD | TBD | TBD | TBD |
| Pure IL (stack) | TBD | TBD | N/A | N/A |

**Update this table after each experiment completes!**

---

## References

### IsaacLab Environments
- **Lift**: `isaaclab_tasks/manager_based/manipulation/lift/`
  - Single cube, simpler task
  - Used as reference for RL Games integration
- **Stack**: `isaaclab_tasks/manager_based/manipulation/stack/`
  - 3 cubes, IL-only (no rewards defined)
  - Used as reference for scene setup and observations

### CuRobo Integration
- **SkillGen Dataset Generation**: `isaaclab_mimic/curobo/generate_dataset.py`
  - CuRobo motion planning for IL data collection
  - Example of CuRobo API usage in IsaacLab

### RL Games
- **Documentation**: https://github.com/Denys88/rl_games
- **PPO Implementation**: `rl_games.algos_torch.a2c_continuous`

---

## Troubleshooting

### Environment Won't Load
```bash
# Check if environment is registered
python -c "import gymnasium as gym; print(gym.spec('Isaac-Stack-RL-Cube-Franka-v0'))"

# Check imports
python -c "from isaaclab_tasks.manager_based.manipulation.stack_rl import *"
```

### PhysX Errors During Training
If you see `totalAggregatePairsCapacity` errors:
1. Open `stack_rl_env_cfg.py`
2. Increase `gpu_total_aggregate_pairs_capacity` (currently 32K)
3. Rule of thumb: `capacity >= (num_envs * num_objects * 2)`

### Low FPS / Slow Training
- Reduce `num_envs` (try 2048 instead of 4096)
- Use `--headless` flag
- Check GPU utilization: `nvidia-smi`
- Reduce `horizon_length` in PPO config

### Training Doesn't Converge
- Increase `max_epochs` (currently 100k)
- Tune reward weights in `stack_rl_env_cfg.py`
- Try different learning rates: 5e-5, 1e-4, 5e-4
- Enable reward shaping curriculum

---

## Contact & Collaboration

This is an ongoing research project. Key decisions and changes should be documented in this file.

**Last Updated**: 2025-12-09
**Status**: ✅ Pure RL baseline training in progress
**Next Steps**: Monitor training completion → Implement CuRobo integration → Comparative analysis

---

## Appendix: File Locations

All files in: `/home/dyros/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack_rl/`

```
stack_rl/
├── WORKFLOW.md                                    # This document
├── stack_rl_env_cfg.py                            # Base environment
├── __init__.py
├── mdp/
│   ├── __init__.py
│   └── rewards.py                                 # Reward functions
└── config/
    ├── __init__.py
    └── franka/
        ├── __init__.py                            # Environment registration
        ├── joint_pos_env_cfg.py                   # Joint control config
        ├── ik_rel_env_cfg.py                      # IK control config
        └── agents/
            └── rl_games_ppo_cfg.yaml              # PPO hyperparameters
```

**Related Files (Outside stack_rl)**:
- Original Stack IL: `isaaclab_tasks/manager_based/manipulation/stack/`
- Lift RL Reference: `isaaclab_tasks/manager_based/manipulation/lift/`
- CuRobo SkillGen: `isaaclab_mimic/curobo/generate_dataset.py`
- Training Script: `scripts/reinforcement_learning/rl_games/train.py`

---

## 📅 20251211 - Major Hyperparameter and Observation Optimization

### 배경: 학습 실패 원인 분석

#### 문제 발견
- **Epoch 22000까지 학습했으나 성능 거의 없음** (reward ~0.1)
- IsaacLab의 `lift` 환경 (1-cube)과 IsaacGym의 `franka_cube_stack` 환경 (2-cube)은 성공했으나, 우리 환경 (3-cube)은 실패

#### 종합 분석 실시
다음 세 환경을 비교 분석:
1. **IsaacLab Lift** (33-dim obs, 학습 성공)
2. **IsaacGym Cube Stack** (19-dim obs, 학습 성공)
3. **우리 Stack RL** (65-dim obs, 학습 실패)

**분석 문서**:
- `DIFFERENCE_OUR_AND_ISAACGYM.md` (IsaacGym vs 우리)
- `DIFFERENCE_LIFT_AND_GYM_STACK_AND_WHATTODO_IN_OUR.md` (세 환경 종합 비교)

### 발견된 문제점

#### 🔴 Critical Issues (학습 안 되는 주원인)

1. **Reward Scale 불일치** ⚠️⚠️⚠️
   ```yaml
   # 현재
   reward_shaper:
     scale_value: 0.01
   # 효과: 최대 229 → 2.29 (너무 작음!)

   # 비교
   IsaacGym: 최대 16 × 1.0 = 16
   IsaacLab: 최대 37 × 0.01 = 0.37
   우리: 최대 229 × 0.01 = 2.29 (IsaacGym의 1/7!)
   ```
   **문제**: Value network가 학습할 범위가 너무 작음, gradient 미약

2. **Learning Rate 너무 낮음**
   ```yaml
   # 현재: 1e-4
   # IsaacGym: 5e-4 (5배 차이!)
   ```
   **문제**: 학습 속도가 너무 느림

3. **Observation 복잡도**
   ```
   IsaacLab: 33-dim (robot root frame, 상대 좌표)
   IsaacGym: 19-dim (cube-to-cube 상대 벡터)
   우리: 65-dim (world frame 절대 좌표)
   ```
   **문제**:
   - Absolute world coordinates 사용 → Policy가 관계를 학습해야 함
   - 상대 벡터 없음 → Task-relevant info가 implicit
   - Sample efficiency 저하

#### 🟡 Important Issues

4. **Minibatch Size 과다**
   - 현재: 32768 × 8 epochs = 262,144 updates
   - IsaacGym: 16384 × 5 epochs = 81,920 updates
   - **문제**: Overfitting 가능성

5. **Episode Length 과다**
   - 현재: 15초 (750 steps)
   - IsaacGym/IsaacLab: 5초
   - **문제**: 학습 iteration 속도 느림

### 수정 사항 (20251211)

#### 1. Reward Scale 증가 (0.01 → 0.1)

**파일**: `config/franka/agents/rl_games_ppo_cfg.yaml`

```yaml
reward_shaper:
  scale_value: 0.1  # 0.01 → 0.1 (10배 증가)
```

**효과**:
- 229 × 0.1 = 22.9 (IsaacGym의 16과 유사)
- Value network 학습 범위 적절화
- Gradient 크기 증가

**근거**:
- IsaacGym: 16 (scale 1.0) → 성공
- IsaacLab: 37 → 0.37 (scale 0.01) → 성공
- 우리: 229 → 22.9 (scale 0.1) → **IsaacGym과 유사한 범위**

#### 2. Learning Rate 증가 (1e-4 → 5e-4)

**파일**: `config/franka/agents/rl_games_ppo_cfg.yaml`

```yaml
learning_rate: 5e-4  # 1e-4 → 5e-4 (5배 증가)
```

**효과**:
- 빠른 학습 속도
- 증가된 reward scale과 조합하여 적절한 gradient step

**근거**:
- IsaacGym: 5e-4로 성공
- 낮은 LR(1e-4) + 작은 reward(2.29) = 매우 느린 학습
- 높은 LR(5e-4) + 적절한 reward(22.9) = 빠른 학습

#### 3. Observation 간소화 (65-dim → 46-dim)

**파일**:
- `mdp/observations.py` (새로 생성)
- `stack_rl_env_cfg.py` (ObservationsCfg 수정)

**변경사항**:
```python
# 제거 (19-dim)
- joint_vel (7-dim)           # IK-Rel에서 덜 중요
- cube_positions (9-dim)      # World frame 절대 좌표
- cube_dimensions (9-dim)     # 상수 (불필요)

# 추가 (9-dim)
+ cube_1_position (3-dim)     # Base cube 절대 위치
+ cube_relative_positions (6-dim)  # 상대 벡터!
  - cube_2_to_cube_1 (3-dim)  # Phase 1에 직접 사용
  - cube_3_to_cube_2 (3-dim)  # Phase 2에 직접 사용

# 총 감소: 19 - 9 = 10-dim → 65 - 19 = 46-dim
```

**새 Observation 구성 (46-dim)**:
```
joint_pos: 7
cube_1_position: 3 (base)
cube_relative_positions: 6 (상대 벡터 ⭐️)
cube_orientations: 12
eef_pos: 3
eef_quat: 4
gripper_pos: 2
actions: 9
---
Total: 46-dim (29% 감소)
```

**효과**:
1. **Task-specific information**: 상대 벡터를 policy가 바로 사용
2. **Sample efficiency**: 학습해야 할 관계가 명시적
3. **Dimension reduction**: 65 → 46 (policy network 크기 감소)

**근거 (IsaacGym 성공 요인)**:
```python
# IsaacGym (19-dim)
"cubeA_to_cubeB_pos": cubeB_pos - cubeA_pos  # 상대 벡터!

# 우리 (새로운 방식)
cube_2_to_cube_1 = cube_1_pos - cube_2_pos  # 동일한 철학
cube_3_to_cube_2 = cube_2_pos - cube_3_pos
```

**새로 추가된 함수 (`mdp/observations.py`)**:
- `cube_relative_positions()`: 큐브 간 상대 벡터 (6-dim)
- `cube_1_position()`: Base cube 절대 위치 (3-dim)
- `cube_orientations_compact()`: Orientation (12-dim, 기존과 동일)

#### 4. Episode Length 단축 (15초 → 10초)

**파일**: `stack_rl_env_cfg.py`

```python
self.episode_length_s = 10.0  # 15 → 10 (33% 감소)
```

**효과**:
- 500 steps @ 50Hz (decimation=2)
- 빠른 trial-and-error iteration
- Sample throughput 증가

**근거**:
- IsaacGym/IsaacLab: 5초 (1-2 cube)
- 우리: 10초 (3-cube, **2배**로 충분한 시간 확보)
- 15초는 과도하게 김

#### 5. Minibatch Size & Mini Epochs 조정

**파일**: `config/franka/agents/rl_games_ppo_cfg.yaml`

```yaml
minibatch_size: 16384  # 32768 → 16384 (절반)
mini_epochs: 5         # 8 → 5 (감소)
```

**효과**:
- Updates per iteration: 262,144 → 81,920 (69% 감소)
- Overfitting 위험 감소
- 안정적인 학습

**근거**:
- IsaacGym: 16384 × 5 = 81,920 (성공)
- 너무 많은 updates는 같은 data로 overfitting 유발

#### 6. Horizon Length 유지 (64 유지)

**파일**: `config/franka/agents/rl_games_ppo_cfg.yaml`

```yaml
horizon_length: 64  # ✅ 유지 (변경 없음)
```

**유지 근거** (사용자 의견 반영):
- IsaacGym (2-cube stacking): horizon=32
- 우리 (3-cube stacking): horizon=64
- **2배 harder task → 2배 longer horizon (합리적)**
- Phase 1 (cube_2) + Phase 2 (cube_3) = ~60 steps 필요

### 설계 철학 선택: IsaacGym vs IsaacLab

#### 분석 결과

**IsaacLab 철학**:
- Robot root frame (상대 좌표)
- Command-based (generalization)
- Moderate reward scale (0.01)
- 33-dim observation

**IsaacGym 철학**:
- Object-to-object 상대 벡터
- Task-specific (hardcoded goal)
- No reward scaling (1.0) or minimal
- 19-dim observation

#### 우리의 선택: Hybrid Approach

**Observation**: IsaacGym 스타일
- 이유: 3-cube stacking은 목표가 고정된 task-specific 문제
- 상대 벡터가 더 직접적 (cube_2 → cube_1, cube_3 → cube_2)

**Reward Scale**: IsaacLab 스타일
- 이유: 우리 최대 reward 229는 IsaacGym 16보다 14배 큼
- Scale 0.1 적용 → 22.9 (적절한 범위)

**Learning Rate**: IsaacGym 스타일
- 이유: 5e-4가 빠른 학습에 효과적
- IsaacGym 성공 사례

**Horizon Length**: 2배 확장 (사용자 의견)
- 이유: 3-cube는 2-cube보다 2배 어려움
- 64 = 32 × 2 (합리적)

### 예상 효과

#### Before (20251210 이전)
```
Reward Scale: 0.01 (max 2.29)
Learning Rate: 1e-4
Observation: 65-dim (절대 좌표)
Episode: 15초
---
결과: Epoch 22000, reward ~0.1 (학습 안 됨)
```

#### After (20251211 수정 후)
```
Reward Scale: 0.1 (max 22.9)
Learning Rate: 5e-4
Observation: 46-dim (상대 좌표)
Episode: 10초
---
예상 결과:
- Epoch 1000: Reaching reward 증가
- Epoch 3000-5000: cube_2 stacking 일부 성공
- Epoch 8000-10000: cube_3 phase 활성화
- Epoch 15000: 3-cube stacking 성공률 증가
```

### 참고 문서

1. **DIFFERENCE_OUR_AND_ISAACGYM.md**
   - IsaacGym franka_cube_stack 환경 분석
   - Observation, Reward, Hyperparameter 비교

2. **DIFFERENCE_LIFT_AND_GYM_STACK_AND_WHATTODO_IN_OUR.md**
   - IsaacLab Lift, IsaacGym Stack, 우리 Stack RL 종합 비교
   - 설계 철학 분석 및 권장사항
   - 왜 각 환경이 성공했는지 분석

3. **mdp/observations.py** (새로 추가)
   - 상대 좌표 observation 함수들
   - 상세한 주석으로 설계 의도 설명

### 수정된 파일 목록

```
✏️ 수정된 파일:
config/franka/agents/rl_games_ppo_cfg.yaml
  - reward_shaper.scale_value: 0.01 → 0.1
  - learning_rate: 1e-4 → 5e-4
  - minibatch_size: 32768 → 16384
  - mini_epochs: 8 → 5
  - horizon_length: 64 (유지)

stack_rl_env_cfg.py
  - episode_length_s: 15.0 → 10.0
  - ObservationsCfg 완전 재구성 (65-dim → 46-dim)

mdp/__init__.py
  - observations.py import 추가

📝 새로 생성된 파일:
mdp/observations.py
  - cube_relative_positions()
  - cube_1_position()
  - cube_orientations_compact()

📚 문서:
DIFFERENCE_OUR_AND_ISAACGYM.md
DIFFERENCE_LIFT_AND_GYM_STACK_AND_WHATTODO_IN_OUR.md
WORKFLOW.md (본 섹션 추가)
```

### 다음 단계

1. **학습 시작**
   ```bash
   ./isaaclab.sh -p scripts/train.py \
       --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --headless \
       --num_envs 4096 \
       --max_iterations 15000
   ```

2. **모니터링**
   ```bash
   tensorboard --logdir logs/rl_games/stack_cube_franka
   ```

3. **주요 체크포인트**
   - Epoch 1000: reaching_cube_2 증가?
   - Epoch 3000: stacking_cube_2_on_1 증가?
   - Epoch 8000: reaching_cube_3 (masked) 활성화?
   - Epoch 15000: task_success 증가?

4. **추가 튜닝 (필요 시)**
   - Reward scale: 0.1 → 0.05 or 0.2 조정
   - Observation: actions 제거 고려 (46 → 37-dim)
   - Reward structure 단순화 (현재는 유지)

---

## 📅 20251211 (2차) - IsaacGym 방식 Reward 완전 재설계

### 배경: Continuous Reward Exploitation 발견

**발견 시점**: Epoch 7309, Reward 148
**관찰된 현상**:
```
✅ cube_2를 집는 건 잘해
❌ cube_2를 cube_1 위에 놓지 않음
❌ 그리퍼를 열지 않음
⚠️ cube_2를 들고 cube_1 위에서 계속 이동만 함
```

**원인 분석**:
1. **Continuous Approach Reward Exploitation**
   ```python
   # 기존 (문제)
   reaching(3.0) + grasping(10.0) + lifting(5.0) + approach(5.0) = 23.0/step
   # → 잡고 이동만 해도 계속 23.0/step 받음!
   # → Episode(500 steps): 23 × 500 = 11,500
   # → Success(30, one-time)보다 훨씬 큼!

   # Policy 학습:
   "들고 이동하면 계속 보상 → 왜 놓아야 하는지 모르겠어"
   ```

2. **Gripper Opening Incentive 없음**
   ```python
   # Success 조건: gripper_open
   # 하지만: gripper 여는 보상 없음
   # 결과: Policy가 왜 열어야 하는지 모름
   ```

3. **Mutual Exclusion 부재**
   ```python
   # IsaacGym (정답)
   rewards = torch.where(
       stacked,
       success_reward_only,    # 성공 → sparse만
       approach_rewards        # 실패 → dense만
   )

   # 기존 stack_rl (문제)
   rewards = approach + success  # 둘 다 활성!
   # → 성공 후에도 approach 계속 받음
   ```

### IsaacGym vs 기존 Stack-RL 비교

| Feature | IsaacGym Stack | 기존 Stack-RL (20251210) | 문제점 |
|---------|----------------|--------------------------|--------|
| **Reward 철학** | Mutual exclusion | All active | Exploit 가능 |
| **Approach reward** | `max(dist, align)` | `approach_weight × exp(...)` | Continuous |
| **Success reward** | `16.0` (exclusive) | `30.0` (one-time) | Approach와 겹침 |
| **성공 후 approach** | ❌ 사라짐 (0) | ✅ 계속 활성 | **Exploit!** |
| **Gripper 조건** | `gripper_away (d > 0.04)` | `gripper_open` | 유사 |
| **Gripper opening reward** | ❌ 없음 | ❌ 없음 | **문제!** |
| **Max reward/step** | `16.0` (success) OR `3.6` (approach) | `53+` (모두 동시) | 너무 높음 |

**IsaacGym 핵심 코드**:
```python
# franka_cube_stack.py line 748-753
rewards = torch.where(
    stack_reward,
    reward_settings["r_stack_scale"] * stack_reward,  # 16.0
    reward_settings["r_dist_scale"] * dist_reward +   # 0.1
    reward_settings["r_lift_scale"] * lift_reward +   # 1.5
    reward_settings["r_align_scale"] * align_reward,  # 2.0
)
# 핵심: torch.where()로 mutual exclusion!
# 성공하면 approach 사라짐 → exploit 불가능
```

### 해결책: IsaacGym 방식 완전 도입

#### 1. **새로운 Reward 함수 추가** (`mdp/rewards.py`)

**6개의 새로운 함수** (line 706-1135):

1. **`reaching_or_aligning_isaacgym()`** - max(reach, align) 방식
   ```python
   # Reaching: EE → object (집기 전)
   reach_reward = 1 - tanh(10.0 * reach_dist)

   # Aligning: object → target (들고 난 후, lifted일 때만)
   align_reward = (1 - tanh(10.0 * align_dist)) * lifted

   # Max (한 번에 하나만)
   return torch.max(reach_reward, align_reward)
   ```
   - **IsaacGym 참고**: `dist_reward = max(dist_reward, align_reward)` (line 735)
   - **효과**: 동시에 두 개 받는 것 방지

2. **`object_is_lifted_simple()`** - Binary lift reward
   ```python
   # Binary (0 or 1), continuous 아님
   lifted = (object_height - table_height) > minimal_height
   return lifted.float()
   ```
   - **IsaacGym 참고**: `cubeA_lifted = (cubeA_height - cubeA_size) > 0.04` (line 725)
   - **효과**: Continuous exploit 방지

3. **`gripper_opening_at_target()`** - Gripper opening reward (새로운 개념!)
   ```python
   # 올바른 위치 체크
   at_target = xy_dist < 0.03 AND z_dist < 0.01

   # 그리퍼 열림 체크
   gripper_open = isclose(gripper_pos, gripper_open_val)

   # 조건 결합
   return (at_target * gripper_open).float()
   ```
   - **IsaacGym에는 없음**: 우리만의 추가 요소
   - **철학**: Opening 행동 자체를 보상 → 학습 유도
   - **효과**: Policy가 "왜 그리퍼를 열어야 하는지" 학습

4. **`stacking_success_isaacgym()`** - Success with gripper away
   ```python
   # Position aligned
   position_aligned = xy_dist < 0.02 AND z_dist < 0.02

   # Gripper away (IsaacGym 핵심!)
   gripper_away = ee_to_object_dist > 0.04

   # Success
   success = position_aligned AND gripper_away
   return success.float()
   ```
   - **IsaacGym 참고**: `gripper_away_from_cubeA = (d > 0.04)` (line 740)
   - **효과**: 놓고 떨어져야 성공

5. **`phase1_mutual_exclusive_reward()`** - Phase 1 torch.where()
   ```python
   # Approach rewards
   approach_rewards = (
       2.0 * reach_align +
       3.0 * lift +
       5.0 * gripper_open
   )  # Max 10.0

   # Success reward
   success_reward = 16.0 * success

   # Mutual Exclusion (핵심!)
   total_reward = torch.where(
       cube_2_stacked,
       success_reward,      # Stacked → only 16.0
       approach_rewards     # Not stacked → only 10.0
   )
   ```
   - **IsaacGym 참고**: `torch.where(stack_reward, ...)` (line 748)
   - **효과**: 성공하면 approach 사라짐, exploit 불가능

6. **`phase2_mutual_exclusive_reward()`** - Phase 2 (동일 구조 + masking)
   ```python
   # Phase 1과 동일한 구조
   # 차이점:
   # - cube_2 on cube_1일 때만 활성화 (masking)
   # - Success reward 더 높음 (20.0)

   return total_reward * mask.float()
   ```

#### 2. **RewardsCfg 완전 재설계** (`stack_rl_env_cfg.py`)

**변경 전 (20251210)**:
```python
# Phase 1: 개별 rewards
reaching_cube_2 = RewTerm(..., weight=3.0)
grasping_cube_2 = RewTerm(..., weight=10.0)
lifting_cube_2 = RewTerm(..., weight=5.0)
stacking_cube_2_on_1 = RewTerm(
    func=mdp.stacking_hybrid_reward,  # approach + success
    params={"approach_weight": 5.0, "success_weight": 30.0},
)

# 문제: 모두 동시 활성, continuous exploit
```

**변경 후 (20251211)**:
```python
# Phase 1: 하나의 mutual exclusive reward
phase1_stacking = RewTerm(
    func=mdp.phase1_mutual_exclusive_reward,
    params={"robot_cfg": SceneEntityCfg("robot")},
    weight=1.0,
)

# 내부 구성:
# - Approach (not stacked): 10.0 = 2.0 + 3.0 + 5.0
# - Success (stacked): 16.0
# - torch.where()로 mutual exclusion
```

**Phase 2도 동일**:
```python
phase2_stacking = RewTerm(
    func=mdp.phase2_mutual_exclusive_reward,
    params={"robot_cfg": SceneEntityCfg("robot")},
    weight=1.0,
)

# + cube_2 on cube_1일 때만 활성화 (masking)
# + Success reward 20.0 (더 높음)
```

**삭제된 Rewards**:
- ❌ `reaching_cube_2`, `reaching_cube_3` (개별)
- ❌ `grasping_cube_2`, `grasping_cube_3` (개별)
- ❌ `lifting_cube_2`, `lifting_cube_3` (개별)
- ❌ `stacking_cube_2_on_1`, `stacking_cube_3_on_2` (hybrid)
- ❌ `cube_1_movement_penalty` (영향도 너무 낮음)
- ❌ `stack_stability` (gripper opening 학습 후 자동 활성화)

**유지된 Rewards**:
- ✅ `cube_1_stays_on_table` (잘 작동 중)
- ✅ `task_success` (최종 성공 보상)
- ✅ `action_rate`, `joint_vel` (표준 penalties)

#### 3. **새로운 Reward 구조 요약**

| Phase | Before (20251210) | After (20251211) | 변화 |
|-------|-------------------|------------------|------|
| **Phase 1 approach** | 23.0 (continuous!) | 10.0 (max) | ✅ -13.0 |
| **Phase 1 success** | 30.0 (one-time, 못 받음) | 16.0 (exclusive) | ✅ Exclusive |
| **Phase 2 approach** | 23.0 (continuous!) | 10.0 (max, masked) | ✅ -13.0 |
| **Phase 2 success** | 50.0 (one-time, 못 받음) | 20.0 (exclusive) | ✅ Exclusive |
| **Gripper opening** | ❌ 없음 | ✅ 5.0 (new!) | ✅ 추가 |
| **Constraints** | 3.0 | 2.0 | ✅ 단순화 |
| **Final success** | 100.0 | 100.0 | 유지 |

**최대 보상 비교**:
```python
# Before (문제)
Max per step = 23.0 (approach, continuous!)
Episode (500 steps) = 11,500 (exploit!)

# After (해결)
Max per step = 16.0 (success, exclusive) OR 10.0 (approach)
Episode success = 16.0 (한 번만, 그 후 다시 approach로 전환 못 함)
```

### 예상 효과

#### 1. **Continuous Exploitation 완전 제거**
```python
# Before: 들고 이동만 해도 계속 보상
# → Policy: "이동만 하자!" (148 reward)

# After: 성공하면 approach 사라짐
# → Policy: "빨리 놓고 다음 단계로!" (16 reward → 0)
```

#### 2. **Gripper Opening 학습**
```python
# Before: gripper_open 조건만 있음, 유인 없음
# → Policy: "왜 열어야 하는지 모르겠어"

# After: 올바른 위치 + gripper open = 5.0 보상
# → Policy: "아, 여기서 열면 보상 받는구나!"
```

#### 3. **One-Way Transition**
```python
# Before: 성공 후에도 approach 계속 받음
# → Success 의미 없음

# After: torch.where()로 mutual exclusion
# → Success = irreversible state change
# → Exploit 불가능
```

#### 4. **학습 속도 향상 예상**
```
Before:
- Epoch 7309: Reward 148 (exploit)
- Behavior: 들고 계속 이동

After (예상):
- Epoch 3000-5000: cube_2 집기 학습
- Epoch 5000-8000: cube_2 놓기 학습 (gripper opening!)
- Epoch 8000-12000: Phase 1 성공 → Phase 2 활성화
- Epoch 12000-18000: cube_3 쌓기 학습
- Epoch 18000+: 3-cube 성공
```

### 수정된 파일 목록

#### 1. **`mdp/rewards.py`**
- **추가**: 6개의 새로운 IsaacGym 스타일 함수 (line 706-1135)
- **변경 사항**:
  - `reaching_or_aligning_isaacgym()` - max(reach, align)
  - `object_is_lifted_simple()` - Binary lift
  - `gripper_opening_at_target()` - Gripper opening reward (new!)
  - `stacking_success_isaacgym()` - Success with gripper away
  - `phase1_mutual_exclusive_reward()` - torch.where() wrapper
  - `phase2_mutual_exclusive_reward()` - torch.where() + masking

#### 2. **`stack_rl_env_cfg.py`**
- **변경**: `RewardsCfg` 완전 재설계 (line 176-298)
- **변경 사항**:
  - 개별 rewards 삭제 (reaching, grasping, lifting 등)
  - Hybrid rewards 삭제 (stacking_hybrid_reward)
  - 2개의 mutual exclusive rewards로 통합
  - Gripper opening 보상 추가
  - 불필요한 constraints 삭제
  - 최대 보상: 229 → 118 (단순화, 현실적)

#### 3. **`WORKFLOW.md`** (이 파일)
- **추가**: 이 섹션 (20251211 2차 수정 문서화)

### 다음 단계

1. **학습 재시작**
   ```bash
   ./isaaclab.sh -p scripts/train.py \
       --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --headless \
       --max_iterations 20000
   ```

2. **모니터링 포인트**
   - Epoch 3000: cube_2 grasping 학습?
   - Epoch 5000: **gripper opening 학습?** (새로운 체크포인트!)
   - Epoch 8000: Phase 1 success → Phase 2 활성화?
   - Epoch 12000: cube_3 grasping?
   - Epoch 18000: 3-cube success?

3. **Tensorboard 체크**
   ```bash
   tensorboard --logdir logs/rl_games/franka_stack_rl
   ```
   - **새로운 metrics**:
     - `phase1_stacking`: 10.0 (approach) or 16.0 (success)
     - `phase2_stacking`: 0 (masked) → 10.0 → 20.0
     - `task_success`: 0 → 100 (최종)

4. **예상 Reward 곡선**
   ```
   Before (문제):
   Reward: 148 (high, stuck)

   After (예상):
   Epoch 0-3000:    5-10  (approaching)
   Epoch 3000-5000: 10-14 (gripper opening 학습!)
   Epoch 5000-8000: 14-18 (Phase 1 success)
   Epoch 8000+:     18-40 (Phase 2 active, 증가)
   ```

### 설계 철학 변경

**Before (20251210)**:
```
IsaacLab Lift 참고 → 모든 reward 동시 활성
+ IsaacGym Success 참고 → One-time flag
= Hybrid approach (실패!)
```

**After (20251211)**:
```
IsaacGym Stack 완전 차용 → torch.where() mutual exclusion
+ Gripper opening reward 추가 → 우리만의 개선
= Pure IsaacGym style + Opening incentive (성공 예상!)
```

**핵심 차이**:
- ❌ Before: "Dense + Sparse 조합" → Exploit 발생
- ✅ After: "Mutually exclusive" → Exploit 불가능

---

## Contact & Collaboration

This is an ongoing research project. Key decisions and changes should be documented in this file.

**Last Updated**: 2025-12-11 (3차 수정)
**Status**: 🔧 Tanh Saturation 문제 해결 (Aggressive Fix) - 학습 재시작 필요
**Next Steps**:
1. ✅ IsaacGym 방식 reward 구현 완료 (2차)
2. ✅ Tanh saturation 문제 해결 (3차)
3. 🔄 학습 재시작 (새 설정으로)
4. 🔍 Policy가 observation 사용하는지 확인 (Epoch 3000)
5. 🎯 Phase 1 → Phase 2 전환 관찰
6. 📊 성공 시 CuRobo 통합 진행

**주요 변경점 (20251211 - 3차)**:
- 🔴 **Critical**: Tanh coefficient 10.0 → 3.0 (gradient 284배 증가!)
- 🟡 **Critical**: Align weight 2.0 → 10.0 (lift와 comparable)
- 🟢 **Fixed**: Policy가 observation 무시하던 문제 해결
- 🔵 **Apply**: Phase 1, Phase 2 모두 동일하게 적용

**주요 변경점 (20251211 - 2차)**:
- 🔴 **Critical**: Mutual exclusion 도입 (torch.where)
- 🟢 **New**: Gripper opening reward 추가
- 🔵 **Redesign**: 전체 reward 구조 재설계
- ⚪ **Removed**: Continuous exploit rewards 완전 제거

---

# 📅 20251211 (3차) - Tanh Saturation 문제 해결 (Aggressive Fix)

## 배경: Policy가 Observation 무시하는 문제 발견

### **문제 발견 경위**

**Epoch 2500 시점 관찰:**
- ✅ Cube_2 grasping & lifting 완벽하게 학습
- ❌ Cube_2를 cube_1 방향이 아닌 **반대 방향(오른쪽-뒤쪽)**으로 이동
- ❌ 모든 environment에서 동일한 방향으로 이동 (cube_1 위치 무관!)

**스크린샷 분석 결과:**
```
Screenshot 1 (초기 위치):
  - Cube_1 (파란색): 테이블 왼쪽
  - Cube_2 (빨간색): 테이블 오른쪽
  - 올바른 행동: 빨간 cube를 왼쪽(←)으로 이동

Screenshot 2 (실제 행동):
  - Robot: 빨간 cube를 오른쪽(→)으로 이동 중!
  - 파란 cube와 점점 멀어짐!
  - 완전히 반대 방향!
```

**핵심 발견:**
- Cube 위치는 environment마다 randomize됨 (확인 완료)
- 하지만 robot은 항상 "오른쪽-뒤쪽"으로 이동
- → **Policy가 `cube_relative_positions` observation을 사용하지 않음!**
- → **Hard-coded 방향 학습 (initialization bias에 의존)**

---

## 근본 원인 분석: Credit Assignment Failure

### **1. Align Reward Gradient가 너무 약함**

```python
# 현재 상황 (거리 0.3m):
align_reward = (1 - tanh(10.0 × 0.3)) × 2.0 × 0.1
             = (1 - 0.995) × 0.2
             = 0.001 per step

# 10cm 이동 시 reward 변화:
왼쪽으로 (올바른 방향): 0.001 → 0.002 (차이 +0.001)
오른쪽으로 (반대 방향): 0.001 → 0.0007 (차이 -0.0003)

# 문제:
PPO exploration noise ≈ 0.1 per action
Signal difference ≈ 0.001
Noise / Signal = 100배!

→ Policy가 noise와 signal 구별 불가능!
→ Observation → Action 연결 학습 실패!
```

### **2. Lift Reward가 너무 Dominant**

```python
# Episode 진행 시 reward:
Lift: 3.0 (계속 유지, 거리 무관)
Align: 0.001 (거리 멀 때)

Total: 3.001
→ Lift가 99.97%
→ Align은 0.03% (완전히 무시됨)

Policy 학습 결과:
  "Cube_2를 들고 있으면 reward 3.0!"
  "어디로 가든 상관없음 (차이 0.001)"
  → Network initialization bias 따라감
  → "오른쪽-뒤쪽" hard-code됨!
```

### **3. Policy가 Observation을 학습하려면**

```python
# Credit Assignment이 작동하려면:
1. Observation: cube_relative_positions = [-0.15, -0.05, 0]
                (cube_1이 왼쪽-앞에 있다는 정보)

2. Action: "왼쪽으로 이동"

3. Reward 증가: align 0.001 → 0.020 (차이 0.019)
                ↑
                최소 0.01 이상 차이 필요 (PPO noise의 10%)

4. Gradient 계산: "이 observation을 보고 왼쪽으로 가면 좋구나!"

# 현재는 3번에서 실패 (차이 0.001 = PPO noise의 1%)
```

---

## 해결책: Aggressive Fix (Option 1)

### **변경 사항 요약**

| 항목 | Before | After | 증가율 |
|------|--------|-------|--------|
| **Tanh coefficient** | 10.0 | **3.0** | 3.33배 완화 |
| **Reach/Align weight** | 2.0 | **10.0** | 5배 증가 |
| **Gradient strength** | 0.001 | **2.84** | **284배 증가!** |

### **정량적 효과 (거리 0.3m 기준)**

```python
# === Before (기존) ===
tanh(10.0 × 0.3) = tanh(3.0) = 0.995
align_reward = (1 - 0.995) × 2.0 = 0.01
After scale (0.1): 0.001 per step

# 10cm 가까워질 때:
0.001 → 0.002 (차이 0.001)

# === After (수정 후) ===
tanh(3.0 × 0.3) = tanh(0.9) = 0.716
align_reward = (1 - 0.716) × 10.0 = 2.84
After scale (0.1): 0.284 per step ← 284배 증가!

# 10cm 가까워질 때:
0.284 → 0.452 (차이 0.168) ← 168배 증가!

# 비교:
Lift reward: 3.0
Align reward: 2.84
→ 이제 comparable! Policy가 무시 불가능!
```

### **예상 학습 패턴**

```python
# Epoch 3000-5000: Observation 사용 시작
- Policy가 cube_relative_positions 관찰
- 올바른 방향으로 이동 시작 (왼쪽 ← cube_1 방향)
- Align reward 서서히 증가 (0.3 → 1.5)

# Epoch 5000-7000: 정렬 학습
- Cube_2가 cube_1 위 근처까지 이동
- Align reward 최대화 (2.84)
- At_target 조건 만족 시작

# Epoch 7000-10000: Gripper opening
- Gripper_opening reward 활성화
- Phase 1 success 달성

# Epoch 10000+: Phase 2
- Phase 2도 동일한 메커니즘으로 빠르게 학습
```

---

## 코드 변경 내역

### **1. mdp/rewards.py**

#### **Line 764: Reaching reward tanh coefficient**
```python
# Before
reach_reward = 1 - torch.tanh(10.0 * reach_dist)

# After
reach_reward = 1 - torch.tanh(3.0 * reach_dist)
```

**이유:**
- Tanh(10.0 × dist)는 0.3m 이상에서 급격히 포화 (gradient ≈ 0)
- 3.0으로 감소 시 saturation 완화, gradient 범위 3배 확장
- IsaacGym도 10.0 사용하지만 초기 거리가 더 가까움 (0.1-0.15m)

#### **Line 783: Aligning reward tanh coefficient**
```python
# Before
align_reward = (1 - torch.tanh(10.0 * align_dist)) * lifted.float()

# After
align_reward = (1 - torch.tanh(3.0 * align_dist)) * lifted.float()
```

**이유:**
- **핵심 수정**: Policy가 observation 무시하는 주범!
- 거리 0.3m: tanh(3.0) = 0.995 → reward 0.005 (너무 작음)
- 거리 0.3m: tanh(0.9) = 0.716 → reward 0.284 (57배 증가!)
- Policy가 cube_relative_positions 사용하여 방향 학습 가능

#### **Line 1026: Phase 1 reach_align weight**
```python
# Before
approach_rewards = (
    2.0 * reach_align +
    3.0 * lift +
    5.0 * gripper_open
)

# After
approach_rewards = (
    10.0 * reach_align +  # 2.0 → 10.0
    3.0 * lift +
    5.0 * gripper_open
)
```

**이유:**
- Lift (3.0)가 너무 dominant → align signal 묻힘
- 10.0으로 증가: align (최대 2.84) vs lift (3.0) - 이제 comparable!
- Policy가 "어디로 가야 할지" 학습 가능 (credit assignment 작동)

#### **Line 1113: Phase 2 reach_align weight**
```python
# Before
approach_rewards = (
    2.0 * reach_align +
    3.0 * lift +
    5.0 * gripper_open
)

# After
approach_rewards = (
    10.0 * reach_align +  # 2.0 → 10.0 (Phase 1과 동일)
    3.0 * lift +
    5.0 * gripper_open
)
```

**이유:**
- Phase 2도 Phase 1과 동일한 문제 겪을 것
- 미리 같이 수정하여 Phase 2 진입 시 바로 학습 가능

---

## 예상 효과

### **즉각적인 효과 (Epoch 3000-5000)**

1. **Policy가 Observation 사용 시작**
   - cube_relative_positions 벡터 방향으로 이동
   - Hard-coded "오른쪽-뒤쪽" 행동 사라짐
   - Cube_1 위치에 따라 adaptive하게 이동

2. **Reward 급격히 증가**
   ```python
   Before: 5.0 (lift 3.0 + align 0.001 + cube_1 2.0)
   After:  7.84 (lift 3.0 + align 2.84 + cube_1 2.0)
   Scaled: 0.784 per step (기존 0.5의 1.6배)
   ```

3. **Visualization 확인사항**
   - Robot이 cube_1 방향으로 이동하는지
   - Environment마다 다른 방향 이동하는지
   - Cube_2가 cube_1에 점점 가까워지는지

### **중기 효과 (Epoch 5000-10000)**

1. **Phase 1 Success 달성**
   - Aligning 완료 → gripper opening 학습
   - Cube_2 on cube_1 성공률 증가

2. **Phase 2 활성화**
   - 동일한 strong gradient로 빠르게 학습
   - Phase 1보다 빠를 수 있음 (mechanism 동일)

### **Risk & Mitigation**

**Risk 1: Align이 너무 dominant**
- Align (최대 2.84) > Lift (3.0) 될 수 있음
- Mitigation: Max(reach, align) 구조로 자동 조절
  - Lifting 전: reach_reward active
  - Lifting 후: align_reward active
  - 둘이 겹치지 않음!

**Risk 2: 초기 학습 불안정**
- Strong gradient가 초기에 혼란 줄 수 있음
- Mitigation: Reaching/Grasping/Lifting은 이미 학습됨 (Epoch 2500)
  - 새로운 학습만 필요 (aligning)
  - 기존 학습 유지됨

**Risk 3: Overfitting to close distances**
- Tanh(3.0)이 가까운 거리에서 덜 정확할 수 있음
- Mitigation: Success condition은 별도 (xy < 0.02m, z < 0.02m)
  - Reward는 guidance, termination은 정확한 조건

---

## 모니터링 가이드

### **Epoch 3000 체크리스트**

```python
# Terminal 확인:
Reward: 7.0-8.0 (기존 5.0-5.5에서 증가)

# Visualization 확인:
1. Robot이 cube_1 방향으로 이동하는가? ✅/❌
2. Environment마다 다른 방향 이동하는가? ✅/❌
3. Cube_2가 cube_1에 가까워지는가? ✅/❌

# 만약 모두 ❌라면:
- Gradient 여전히 부족 (드물지만 가능)
- 추가 조치: weight 10.0 → 15.0 고려
```

### **Epoch 5000 체크리스트**

```python
Reward: 9.0-11.0 (aligning 진행)

Visualization:
- Cube_2가 cube_1 위 5-10cm 이내? ✅/❌
- Gripper opening 시도? ✅/❌
```

### **Epoch 7000 체크리스트**

```python
Reward: 12.0-16.0 (Phase 1 success 시작)

Visualization:
- Cube_2 on cube_1 성공 environment 수 증가? ✅/❌
- Phase 2 reward 활성화? (0이 아닌 값) ✅/❌
```

---

## 변경 파일 목록

1. **mdp/rewards.py**
   - Line 764: `reach_reward` tanh coefficient (10.0 → 3.0)
   - Line 783: `align_reward` tanh coefficient (10.0 → 3.0)
   - Line 1026: Phase 1 `reach_align` weight (2.0 → 10.0)
   - Line 1113: Phase 2 `reach_align` weight (2.0 → 10.0)

2. **WORKFLOW.md** (this section)

---

## 다음 단계

1. **즉시**: 학습 재시작
   ```bash
   python scripts/rl_games/train.py --task Isaac-Stack-RL-Franka-IK-Rel-v0
   ```

2. **Epoch 3000**: 첫 체크포인트 확인
   - Robot 행동 시각화
   - Observation 사용 여부 확인

3. **Epoch 5000-7000**: Phase 1 success 모니터링

4. **Epoch 10000+**: Phase 2 진행 및 3-cube 완성

---

## 참고 자료

- **분석 문서**: `DIFFERENCE_LIFT_AND_GYM_STACK_AND_WHATTODO_IN_OUR.md`
- **IsaacGym 코드**: `IsaacGymEnvsTwk-main/isaacgymenvs/tasks/franka_cube_stack.py`
- **Tanh saturation 분석**: Session 2025-12-11 conversation
- **Screenshot 증거**:
  - `스크린샷 2025-12-11 16-29-12_초기위치.png`
  - `스크린샷 2025-12-11 16-29-44_cube1위가 아닌 다른 곳 탐색.png`

---

---

# 📅 20251211 (4차) - Excessive Height Lifting 문제 해결

## 배경: Robot이 Cube를 너무 높이 들어올림

### **문제 발견 경위**

**관찰된 현상** (Screenshot: `moving_up_motion_after_pickup_cube2.png`):
```
✅ Cube_2 grasping 성공
✅ Cube_2 lifting 성공
❌ Cube_2를 40-60cm까지 들어올림 (비정상!)
❌ 쌓기에 불필요한 높이 (10-20cm면 충분)
```

**사용자 피드백**:
> "계속 cube2를 집고 난 뒤에 완전 위로 들고 올라가... 이런 방향으로 학습되는 것도 지양해야할 것 같은데"

**핵심 문제**:
- 쌓기 작업은 10-20cm 높이면 충분
- Robot은 50-60cm까지 들어올림
- 불필요한 workspace 사용, 비효율적

---

## 근본 원인 분석

### **1. Lift Reward가 Binary (높이 무관)**

```python
# mdp/rewards.py - object_is_lifted_simple()
lifted = (object_height - table_height) > 0.04  # 4cm threshold
lift_reward = lifted.float()  # 0 or 1

# 문제:
height = 5cm  → lift_reward = 1.0 → internal 3.0
height = 50cm → lift_reward = 1.0 → internal 3.0 (동일!)
height = 100cm → lift_reward = 1.0 → internal 3.0 (동일!)

→ 높이에 대한 penalty 없음!
→ Policy가 "높을수록 안전" 학습 가능성
```

### **2. Align Reward가 아직 약함 (3차 수정 직후)**

```python
# 거리 0.5m (높이 50cm, 매우 높음)
align = 0.12 × 10.0 = 1.2
lift = 3.0
Total = 4.2 per step

# 거리 0.1m (높이 10cm, 적절)
align = 0.54 × 10.0 = 5.4
lift = 3.0
Total = 8.4 per step

# 차이: 4.2 vs 8.4 (4.2 차이)
# 문제:
- 차이는 존재하지만, policy가 높은 곳에서 local optimum에 갇힘
- "낮춰서 5.4 받기 vs 높은 곳에서 안전하게 1.2 받기"
- Exploration이 부족하면 후자 선택 가능
```

### **3. Workspace Constraint 없음**

```python
# 현재 reward structure:
- Lift: 높이 > 4cm이면 항상 +3.0
- Align: 거리 기반 (높이 간접적으로만 영향)
- Height penalty: ❌ 없음!

# 결과:
Robot이 높이 제한 없이 exploration
→ 높은 곳에서도 reward 받음
→ Local optimum에 갇힐 수 있음
```

---

## 해결책 논의

### **사용자 제안 (초기)**
- `max_reasonable_height = 0.4m` (40cm)
- `penalty_scale = 0.6`

**분석**:
```python
# 50cm에서 penalty:
penalty = -0.6 × (0.50 - 0.40)² = -0.006

# 문제:
Align reward: 1.2
Lift reward: 3.0
Height penalty: -0.006
Total: 4.194 (penalty가 너무 약함!)

→ Policy가 무시 가능한 수준
```

### **Claude 제안 (최종 채택)**
- `max_reasonable_height = 0.25m` (25cm)
- `penalty_scale = 5.0`

**효과**:
```python
# === 높이별 Penalty ===

# 10cm (최적):
excess = 0
penalty = 0
→ ✅ 아무 penalty 없음

# 20cm (적절):
excess = 0
penalty = 0
→ ✅ 여전히 괜찮음

# 25cm (경계):
excess = 0
penalty = 0
→ ⚠️ 경계선

# 30cm (약간 높음):
excess = 0.05
penalty = -5.0 × 0.05² = -0.0125
Total reward: 6.0 (align) + 3.0 (lift) - 0.0125 = 8.9875
→ ⚠️ 약한 penalty

# 40cm (높음):
excess = 0.15
penalty = -5.0 × 0.15² = -0.1125
Total: 3.0 + 1.8 - 0.1125 = 4.69
→ ⚠️ 중간 penalty

# 50cm (매우 높음):
excess = 0.25
penalty = -5.0 × 0.25² = -0.3125
Total: 3.0 + 1.2 - 0.3125 = 3.89
→ ❌ 강한 penalty!

# 60cm (비정상):
excess = 0.35
penalty = -5.0 × 0.35² = -0.6125
Total: 3.0 + 0.8 - 0.6125 = 3.19
→ ❌ 매우 강한 penalty!
```

**왜 더 효과적인가?**
1. **Lower threshold (40cm → 25cm)**:
   - 쌓기에 25cm면 충분
   - 30cm+ 부터 penalty 시작 → 더 빠른 유도

2. **Higher scale (0.6 → 5.0)**:
   - Quadratic penalty가 실제로 영향을 줌
   - 50cm: -0.006 → -0.3125 (52배 강화!)

3. **Align과 Balance**:
   - 낮은 곳 (10cm): align 5.4, penalty 0 → total 8.4
   - 높은 곳 (50cm): align 1.2, penalty -0.3125 → total 3.89
   - 차이: 4.51 (더 명확한 gradient!)

---

## 해결책 2: Align Weight 추가 증가 (10.0 → 15.0)

### **추가 문제 발견**

```python
# 높이 50cm에서도:
Lift: 3.0 (guaranteed)
Align: 1.2 (변동)
Total: 4.2

# 높이 10cm에서:
Lift: 3.0 (guaranteed)
Align: 5.4 (변동)
Total: 8.4

# 차이는 4.2지만, Lift가 여전히 dominant
# → Policy가 "lift 유지" 전략 선호 가능성
```

### **해결: Align Weight 15.0으로 증가**

**변경 전 (3차 수정)**:
```python
approach_rewards = (
    10.0 * reach_align +  # Max: 2.84 × 10 = 28.4
    3.0 * lift +          # Max: 3.0
    5.0 * gripper_open    # Max: 5.0
)
```

**변경 후 (4차 수정)**:
```python
approach_rewards = (
    15.0 * reach_align +  # Max: 2.84 × 15 = 42.6
    3.0 * lift +          # Max: 3.0
    5.0 * gripper_open    # Max: 5.0
)
```

**효과**:
```python
# 높이 50cm (높음, 거리 0.5m):
align = 0.12 × 15.0 = 1.8
lift = 3.0
penalty = -0.3125
Total = 1.8 + 3.0 - 0.3125 = 4.49

# 높이 10cm (적절, 거리 0.1m):
align = 0.54 × 15.0 = 8.1
lift = 3.0
penalty = 0
Total = 8.1 + 3.0 = 11.1

# 차이: 6.61 (기존 4.2의 1.57배!)
# → 훨씬 강한 signal!

# 이제 Align이 Lift보다 훨씬 강함:
Align (max): 8.1
Lift: 3.0
→ Policy가 "낮은 곳에서 align하는 게 훨씬 이득" 학습
```

---

## 코드 변경 내역

### **1. mdp/rewards.py**

#### **새 함수 추가: cube_height_penalty() (Lines 1168-1237)**

```python
def cube_height_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    max_reasonable_height: float = 0.25,
    penalty_scale: float = 5.0,
) -> torch.Tensor:
    """
    20251211 추가 (4차): Cube 높이 제한 penalty

    Problem:
        Robot이 cube를 40-60cm까지 들어올림 (비정상적으로 높음)

    Cause:
        Lift reward가 binary (>4cm = 1.0, 높이 무관)
        → 50cm든 100cm든 동일한 reward
        → Policy가 높이 제한 학습 못 함

    Solution:
        Quadratic penalty for excessive height
        penalty = -scale × (excess)²

    Args:
        max_reasonable_height: 합리적인 최대 높이 (m)
            - 0.25 (25cm): 쌓기 작업에 충분한 높이
            - 이상 들면 penalty 시작

        penalty_scale: Penalty 강도
            - 5.0: Quadratic하게 증가
            - 높이가 높을수록 penalty 급격히 증가

    Examples (max_height=0.25, scale=5.0):
        height=10cm: excess=0     → penalty=0         (✅ good!)
        height=25cm: excess=0     → penalty=0         (boundary)
        height=30cm: excess=0.05  → penalty=-0.0125   (weak)
        height=50cm: excess=0.25  → penalty=-0.3125   (strong!)
        height=60cm: excess=0.35  → penalty=-0.6125   (very strong!)

    Returns:
        Penalty (≤ 0)
    """
    object: RigidObject = env.scene[object_cfg.name]

    # Cube height above table
    cube_height = object.data.root_pos_w[:, 2]
    table_height = 0.0203  # Seattle Lab Table
    height_above_table = cube_height - table_height

    # Excess height (only penalize if above threshold)
    excess_height = torch.clamp(height_above_table - max_reasonable_height, min=0.0)

    # Quadratic penalty
    penalty = -penalty_scale * excess_height ** 2

    return penalty
```

**설계 철학**:
1. **Quadratic Penalty**: 높이가 높아질수록 penalty 급격히 증가
2. **Threshold 기반**: 25cm 이하는 penalty 없음 (자유)
3. **Continuous**: Differentiable gradient 제공

#### **Line 1044: Phase 1 reach_align weight (10.0 → 15.0)**

```python
# Before (3차 수정)
approach_rewards = (
    10.0 * reach_align +
    3.0 * lift +
    5.0 * gripper_open
)

# After (4차 수정)
approach_rewards = (
    15.0 * reach_align +  # 10.0 → 15.0 (align이 lift보다 강함!)
    3.0 * lift +
    5.0 * gripper_open
)
```

**주석 추가**:
```python
# 20251211 수정 (4차): 10.0 → 15.0
# 이유:
#   - Lift (3.0)가 여전히 dominant
#   - 높은 곳에서 local optimum 발생 가능성
#   - 15.0으로 증가 → align이 lift보다 훨씬 강함
#   - 낮은 곳에서 align하도록 강하게 유도
# 효과:
#   - 높이 10cm: align 8.1 vs lift 3.0 → align 우위!
#   - 높이 50cm: align 1.8 vs lift 3.0 → 감소 명확
```

#### **Line 1135: Phase 2 reach_align weight (10.0 → 15.0)**

```python
# Before (3차 수정)
approach_rewards = (
    10.0 * reach_align +
    3.0 * lift +
    5.0 * gripper_open
)

# After (4차 수정)
approach_rewards = (
    15.0 * reach_align +  # 10.0 → 15.0 (Phase 1과 동일)
    3.0 * lift +
    5.0 * gripper_open
)
```

**이유**: Phase 1과 일관성 유지

---

### **2. stack_rl_env_cfg.py**

#### **Lines 300-337: Height Penalty Terms 추가**

```python
# ========================================
# Height Penalties (4th modification - 20251211)
# ========================================

cube_height_penalty_cube_2 = RewTerm(
    func=mdp.cube_height_penalty,
    params={
        "object_cfg": SceneEntityCfg("cube_2"),
        "max_reasonable_height": 0.25,  # 25cm
        "penalty_scale": 5.0,            # Quadratic penalty
    },
    weight=1.0,
)

cube_height_penalty_cube_3 = RewTerm(
    func=mdp.cube_height_penalty,
    params={
        "object_cfg": SceneEntityCfg("cube_3"),
        "max_reasonable_height": 0.25,  # 25cm
        "penalty_scale": 5.0,            # Quadratic penalty
    },
    weight=1.0,
)
```

---

## 새로운 Reward 구조 (4차 수정 후)

### **Phase 1 Approach Rewards**

| Component | Weight | Max Internal | Scaled (×0.1) | Notes |
|-----------|--------|--------------|---------------|-------|
| **Reach/Align** | 15.0 | 2.84 × 15 = 42.6 | 4.26 | ⬆️ 10.0→15.0 |
| **Lift** | 3.0 | 1.0 × 3 = 3.0 | 0.3 | 유지 |
| **Gripper Open** | 5.0 | 1.0 × 5 = 5.0 | 0.5 | 유지 |
| **Height Penalty** | 1.0 | -0.6125 (max) | -0.06125 | ✅ 새로 추가 |
| **Total** | - | **50.6 or 50.6 - penalty** | **5.06 - penalty** | - |

### **높이별 Total Reward (거리 0.1m 가정)**

```python
# 10cm (최적):
align = 0.54 × 15.0 = 8.1
lift = 3.0
penalty = 0
Total = 11.1 → Scaled: 1.11

# 20cm (적절):
align = 0.48 × 15.0 = 7.2
lift = 3.0
penalty = 0
Total = 10.2 → Scaled: 1.02

# 30cm (약간 높음):
align = 0.40 × 15.0 = 6.0
lift = 3.0
penalty = -0.0125
Total = 8.99 → Scaled: 0.899

# 50cm (매우 높음):
align = 0.12 × 15.0 = 1.8
lift = 3.0
penalty = -0.3125
Total = 4.49 → Scaled: 0.449

# 차이: 1.11 vs 0.449 = 2.47배!
# → Policy가 낮은 곳 선호하도록 강하게 유도
```

---

## 예상 효과

### **즉각적인 효과 (Epoch 3000-5000)**

1. **Robot이 낮은 높이 유지**
   - 현재: 40-60cm 들어올림
   - 예상: 10-25cm 범위 내 유지
   - Height penalty가 즉시 작용

2. **Align Reward 증가**
   - 낮은 곳에서 align하면 더 높은 reward
   - Policy가 "빨리 낮춰서 align" 학습

3. **Workspace Efficiency**
   - 불필요한 높이 탐색 감소
   - 쌓기에 최적화된 높이 학습

### **중기 효과 (Epoch 5000-8000)**

1. **Phase 1 Success Rate 증가**
   - 적절한 높이에서 aligning 완료
   - Gripper opening 더 빠르게 학습
   - Cube_2 on cube_1 성공률 향상

2. **Learning Stability**
   - Local optimum (높은 곳에서 안전) 회피
   - Consistent한 낮은 높이 학습
   - Phase 2로 빠른 전환

### **장기 효과 (Epoch 8000+)**

1. **Phase 2도 동일한 mechanism**
   - Cube_3도 적절한 높이에서 쌓기
   - 전체 task 완성 시간 단축

2. **3-Cube Stacking Success**
   - 모든 phase에서 최적 높이 유지
   - Task completion 향상

---

## Risk Analysis & Mitigation

### **Risk 1: Penalty가 너무 강할 수 있음**

**현상**:
- 50cm: penalty -0.3125 (강함)
- Policy가 4cm 근처에서만 머물 수 있음

**Mitigation**:
- Penalty는 25cm부터 시작 (10-20cm은 자유)
- Quadratic이라 초반(25-30cm)은 약함 (-0.0125)
- Lift reward (3.0)가 작은 penalty 상쇄 가능
- → 적절한 balance

### **Risk 2: Align이 너무 dominant (15.0)**

**현상**:
- Align (max 8.1) >> Lift (3.0)
- Policy가 lift 무시하고 align만 추구?

**Mitigation**:
- Max(reach, align) 구조로 자동 조절
  - Lifting 전: reach active (align = 0, lifted=False)
  - Lifting 후: align active
- Lift는 binary로 항상 3.0 보장
- → 둘이 겹치지 않음!

### **Risk 3: 초기 학습 혼란**

**현상**:
- 새로운 penalty로 reward 감소
- Policy가 혼란스러울 수 있음

**Mitigation**:
- Reaching/Grasping/Lifting 이미 학습됨 (Epoch 2500)
- Aligning 학습 시작 단계 (3차 수정)
- Height penalty는 추가 guidance일 뿐
- → 기존 학습 유지, 새로운 constraint만 추가

---

## 모니터링 가이드

### **Epoch 3000 체크리스트**

```python
# Visualization 확인:
1. Cube_2 높이: 10-30cm 범위 내? ✅/❌
   (기존: 40-60cm)

2. Robot이 불필요하게 높이 들지 않는가? ✅/❌

3. Align reward 증가? (기존 0.2 → 예상 0.8+) ✅/❌

# Tensorboard:
height_penalty_cube_2: -0.01 ~ -0.05 (약한 penalty 발생)
phase1_stacking: 0.8-1.0 (증가 추세)
```

### **Epoch 5000 체크리스트**

```python
# Visualization:
1. Cube_2가 cube_1 위 10-20cm에 위치? ✅/❌

2. Gripper opening 시도? ✅/❌

3. Phase 1 success 환경 수 증가? ✅/❌

# Tensorboard:
height_penalty_cube_2: 0 ~ -0.01 (거의 0, 좋은 신호!)
phase1_stacking: 1.2-1.6 (success 시작)
```

### **Epoch 7000 체크리스트**

```python
# Phase 2 활성화 확인:
1. Phase 2 reward > 0? ✅/❌

2. Cube_3도 적절한 높이? ✅/❌

# Tensorboard:
phase2_stacking: 0.5-1.0 (활성화!)
task_success: 10-50 (최종 성공 시작)
```

---

## 수정된 파일 목록

### **1. mdp/rewards.py**
- **Line 1168-1237**: `cube_height_penalty()` 함수 추가
  - Quadratic height penalty
  - max_reasonable_height=0.25, penalty_scale=5.0
  - 상세한 docstring 및 한글 주석

- **Line 1044**: Phase 1 reach_align weight (10.0 → 15.0)
  - Align이 lift보다 강하도록
  - 주석으로 이유 설명

- **Line 1135**: Phase 2 reach_align weight (10.0 → 15.0)
  - Phase 1과 일관성 유지

### **2. stack_rl_env_cfg.py**
- **Lines 300-337**: Height penalty reward terms 추가
  - `cube_height_penalty_cube_2`
  - `cube_height_penalty_cube_3`
  - 상세한 주석으로 배경, 원인, 해결책 설명

### **3. WORKFLOW.md**
- **This section**: 4차 수정 문서화
  - 문제 배경
  - 근본 원인 분석
  - 해결책 논의 (사용자 vs Claude)
  - 코드 변경 내역
  - 예상 효과
  - 모니터링 가이드

---

## 다음 단계

1. **학습 재시작**
   ```bash
   python scripts/rl_games/train.py \
       --task Isaac-Stack-RL-Franka-IK-Rel-v0 \
       --headless \
       --num_envs 8192
   ```

2. **Epoch 3000**: 높이 체크
   - Cube height visualization
   - Height penalty 작동 확인

3. **Epoch 5000**: Phase 1 완성도
   - Gripper opening 학습
   - Success rate 증가

4. **Epoch 7000+**: Phase 2 & Task completion
   - 3-cube stacking 성공

---

## 설계 철학 (4차 수정 추가)

**3차 수정 후**:
```
✅ Policy가 observation 사용 (tanh 3.0, weight 10.0)
⚠️ 하지만 높이 제한 없음 → 40-60cm까지 들어올림
⚠️ Align vs lift balance 부족 → local optimum 가능
```

**4차 수정**:
```
✅ Height penalty 추가 → 25cm 이상 제한
✅ Align weight 15.0 → lift보다 훨씬 강함
✅ 낮은 곳에서 align하도록 강하게 유도
= 최적 높이 (10-20cm) 학습 예상!
```

**종합**:
- 3차: Observation 사용 문제 해결 (방향 학습)
- 4차: Workspace 최적화 문제 해결 (높이 학습)
- → Policy가 **어디로**(방향) + **어느 높이에서**(workspace) 학습 완성!

---

# 5차 수정 (20251212): Option B - Stage-based Curriculum Learning with Anca et al. Rewards

## 수정 사항 (20251212)

### 배경 및 문제

**4차 수정 후 상황**:
```
✅ Observation 사용 문제 해결 (3차)
✅ Height penalty로 workspace 최적화 (4차)
⚠️ 하지만 여전히 gripper opening 학습 안 됨
⚠️ Stage 구분 없이 모든 cube 동시에 학습 시도
⚠️ Mutual exclusion이 sub-optimal policy 유도 가능
```

**GPT-5.1 분석 (Option A vs Option B)**:
- **Option A** (기존 접근): IsaacGym 스타일 mutual exclusion + 점진적 조정
  - 장점: 간단함, 작은 변경
  - 단점: 근본적 학습 구조는 동일, 성공률 향상 한계

- **Option B** (새 접근): **Curriculum Learning + Anca et al. (2023) Reward Structure**
  - 장점: 검증된 방법론, 단계별 학습, 명확한 sub-goal
  - 논문: "Achieving Goals using Reward Shaping and Curriculum Learning" (arXiv:2206.02462)
  - **사용자 선택**: "Option B로 가자!"

### 근본 원인

**IsaacGym 스타일의 한계**:
1. **Mutual Exclusion의 문제**:
   - Max(reach_1, reach_2, reach_3) → 한 번에 하나만 학습
   - 하지만 "놓는 행위" (gripper open)는 sparse signal
   - 성공 순간의 가치를 충분히 알려주지 못함

2. **Stage 구분 없음**:
   - 초기부터 3개 cube 모두 고려
   - 복잡한 state space → 학습 느림
   - Sub-goal 달성의 명확한 신호 부족

3. **Sub-goal Bonus 부재**:
   - Cube_2 on cube_1 달성 시 특별한 보상 없음
   - 단지 align reward만 증가
   - "성공"의 가치를 policy가 모름

### 해결책: Anca et al. (2023) Curriculum + Reward Shaping

**핵심 아이디어**:
```python
# 1. Curriculum Stages (Epoch 기반 자동 전환)
Epoch 0-1000:    Stage 1 - Cube_2 → Cube_1만 학습
Epoch 1000-5000: Stage 2 - Cube_2 완료 + Cube_3 → Cube_2 학습
Epoch 5000+:     Stage 3 - 모든 cube 학습

# 2. Anca Reward Structure
r_total = λ₁·r_dense + λ₂·r_guide + λ₃·r_subgoal + λ₄·r_success + penalties

# 3. Sub-goal One-time Bonuses
Cube_2 on Cube_1 달성 → +150 (한 번만!)
Cube_3 on Cube_2 달성 → +150 (한 번만!)
All stacked → +150 (global success!)
```

**Anca et al. λ Weights (Table 1)**:
| Reward Term | λ (Weight) | 설명 |
|------------|-----------|------|
| r_dense (box-to-goal) | 5.0 | Cube → Goal 거리 (continuous) |
| r_guide (ee-to-box) | 5.0 | EE → Target cube 거리 |
| r_subgoal | 150.0 | **Sub-goal 달성 순간 huge spike!** |
| r_success | 150.0 | **Global success huge spike!** |
| r_action | 0.01 | Action smoothness |
| r_table | 5.0 | Table collision penalty |
| r_orientation | 0.1 | EE orientation maintenance |

**After scaling (×0.1)**:
- Dense rewards: ~1.0 per step
- **Sub-goal bonus: 15.0** (한 순간!)
- **Global success: 15.0** (한 순간!)
- 이게 "놓는 행위"를 policy에게 알려주는 핵심!

---

## 구현 내용

### 1. Curriculum Stage Management

**새 함수**: `mdp/rewards.py` line 1250-1310

```python
def get_curriculum_stage(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    현재 curriculum stage 반환 (1, 2, or 3)

    Epoch 기반 자동 전환:
        - 0-1000: Stage 1 (cube_2 → cube_1만)
        - 1000-5000: Stage 2 (cube_2 완료 + cube_3 → cube_2)
        - 5000+: Stage 3 (all cubes)

    Epoch 추정:
        total_steps / (num_envs × 500)
        → PPO epoch ≈ steps per rollout
    """
```

**장점**:
- 자동 전환: Manual buffer swap 불필요
- 점진적 난이도 증가
- 초기 학습 속도 향상 (간단한 task부터)

### 2. Anca Reward Functions

**새 함수들**: `mdp/rewards.py` line 1250-1745

#### Dense Shaping Rewards (Continuous Guidance)

1. **`box_to_goal_distance_anca()`** (line 1370-1490)
   ```python
   # r_dense in Anca et al.
   r = -Σ ||s_goal - s_obj||² / 2

   # Curriculum gating:
   - Stage 1: cube_2만 계산
   - Stage 2+: cube_2 + cube_3
   ```

2. **`ee_to_box_distance_anca()`** (line 1493-1565)
   ```python
   # r_guide in Anca et al.
   r = -||s_ee - s_target_cube||² / 2

   # Target selection:
   - Stage 1: EE → cube_2
   - Stage 2+: EE → cube_3
   ```

#### Sparse Bonuses (One-time Rewards)

3. **`subgoal_sparse_bonus_anca()`** (line 1663-1743)
   ```python
   # Sub-goal one-time bonus
   # 각 cube가 목표에 도달한 순간 한 번만!

   # Tracking in env.extras:
   - subgoal_cube_2_done: Bool[num_envs]
   - subgoal_cube_3_done: Bool[num_envs]

   # XY + Z alignment 체크 후 newly_done 환경에만 +1
   # Weight: λ=150.0 → scaled 15.0 huge spike!
   ```

4. **`global_success_bonus_anca()`** (line 1633-1660)
   ```python
   # All 3 cubes stacked → +150 (한 번!)
   # Episode 종료 신호로도 활용 가능
   ```

#### Penalties

5. **`action_penalty_anca()`** (line 1568-1590)
   - Action magnitude + joint deviation from home
   - λ=0.01 (약함, smoothness만)

6. **`table_collision_penalty_anca()`** (line 1593-1612)
   - EE height < table_height → -1
   - λ=5.0 (강함, 충돌 방지)

7. **`ee_orientation_penalty_anca()`** (line 1615-1630)
   - 초기 orientation 대비 deviation
   - λ=0.1 (약함, stable grasping 유도)

#### Helper Functions

8. **`init_subgoal_tracking()`** (line 1313-1341)
   - env.extras에 tracking buffers 초기화
   - subgoal_cube_2_done, subgoal_cube_3_done

9. **`reset_subgoal_tracking()`** (line 1344-1367)
   - Episode reset 시 tracking buffers 리셋
   - 중복 보상 방지

### 3. RewardsCfg 완전 재구성

**파일**: `stack_rl_env_cfg.py` line 175-331

**Before (4차 수정)**:
```python
# IsaacGym mutual exclusion style
reaching_cube_1 = max(...)
reaching_cube_2 = max(...)
reaching_cube_3 = max(...)
grasping_cube_X = 10.0
stacking = align rewards
+ height_penalty (new in 4차)
```

**After (5차 수정)**:
```python
@configclass
class RewardsCfg:
    """
    Anca et al. Style with Curriculum Learning
    arXiv:2206.02462
    """

    # Dense Shaping (λ=5.0)
    box_to_goal_distance = RewTerm(
        func=mdp.box_to_goal_distance_anca,
        weight=5.0
    )

    ee_to_box_distance = RewTerm(
        func=mdp.ee_to_box_distance_anca,
        weight=5.0
    )

    # Sparse Bonuses (λ=150.0)
    subgoal_bonus = RewTerm(
        func=mdp.subgoal_sparse_bonus_anca,
        weight=150.0  # 매우 큼!
    )

    global_success = RewTerm(
        func=mdp.global_success_bonus_anca,
        weight=150.0
    )

    # Penalties
    action_penalty = RewTerm(
        func=mdp.action_penalty_anca,
        weight=0.01
    )

    table_collision_penalty = RewTerm(
        func=mdp.table_collision_penalty_anca,
        weight=5.0
    )

    orientation_penalty = RewTerm(
        func=mdp.ee_orientation_penalty_anca,
        weight=0.1
    )

    # Existing constraint (유지)
    cube_1_stays_on_table = RewTerm(
        func=mdp.object_stays_on_table,
        weight=2.0
    )
```

**주요 변경점**:
1. ❌ **제거**: IsaacGym 스타일 mutual exclusion (reaching_X, grasping_X)
2. ❌ **제거**: Height penalty (Anca dense reward가 대체)
3. ✅ **추가**: Anca dense shaping (box_to_goal, ee_to_box)
4. ✅ **추가**: Anca sparse bonuses (subgoal, global_success)
5. ✅ **추가**: Anca penalties (action, table, orientation)
6. ✅ **유지**: cube_1_stays_on_table (기존 constraint)

---

## 수정된 파일 목록

### **1. mdp/rewards.py**

**새 함수 (Line 1250-1745)**:
- `get_curriculum_stage()`: Epoch 기반 stage 관리
- `init_subgoal_tracking()`: One-time bonus tracking 초기화
- `reset_subgoal_tracking()`: Episode reset 시 tracking 리셋
- `box_to_goal_distance_anca()`: Dense shaping (λ=5.0)
- `ee_to_box_distance_anca()`: EE guidance (λ=5.0)
- `action_penalty_anca()`: Action smoothness (λ=0.01)
- `table_collision_penalty_anca()`: Table collision (λ=5.0)
- `ee_orientation_penalty_anca()`: Orientation (λ=0.1)
- `subgoal_sparse_bonus_anca()`: Sub-goal bonus (λ=150.0)
- `global_success_bonus_anca()`: Global success (λ=150.0)

**주석**:
- 모든 함수에 상세한 docstring
- Anca 논문 식 참조
- Curriculum gating 설명
- 한글 주석으로 의도 명확화

### **2. stack_rl_env_cfg.py**

**RewardsCfg 재구성 (Line 175-331)**:
- Anca et al. (2023) 논문 기반 완전 재설계
- λ weights from Anca Table 1
- Curriculum timeline 명시
- 최대 보상 분석 (before/after scaling)
- 각 reward term마다 상세 주석
- 배경, 문제, 해결책 설명

### **3. WORKFLOW.md**

**This section**: 5차 수정 문서화
- GPT-5.1 분석 배경
- Option A vs B 비교
- Anca et al. 논문 요약
- Curriculum learning 철학
- 모든 코드 변경사항
- 예상 학습 곡선
- 모니터링 가이드

---

## 예상 효과

### 학습 곡선 예측

#### Stage 1 (Epoch 0-1000): Cube_2 → Cube_1
```
예상 진행:
- Epoch 0-300: EE → cube_2 도달 학습 (ee_to_box reward 증가)
- Epoch 300-600: Grasping 학습 (contact force + gripper close)
- Epoch 600-800: Cube_2 → Cube_1 stacking 학습 (box_to_goal reward 증가)
- Epoch 800-1000: Sub-goal 달성! (subgoal_bonus spike 관찰)

Tensorboard 신호:
- ee_to_box_distance: -5.0 → -1.0 (개선)
- box_to_goal_distance: -5.0 → -1.0 (개선)
- subgoal_bonus: 0 → 15.0 spikes 관찰! ← 핵심!
- Success rate (Stage 1): 0% → 60-80%
```

#### Stage 2 (Epoch 1000-5000): + Cube_3 → Cube_2
```
예상 진행:
- Epoch 1000-1500: Cube_2 skill 유지하면서 cube_3 추가
- Epoch 1500-3000: Cube_3 grasping + stacking 학습
- Epoch 3000-5000: 2-cube stack 달성 (2nd subgoal bonus!)

Tensorboard 신호:
- ee_to_box_distance: Target이 cube_3로 전환
- subgoal_bonus: 2nd spike 등장! (cube_3 on cube_2)
- Success rate (2-stack): 0% → 40-60%
```

#### Stage 3 (Epoch 5000+): Full 3-Cube Stack
```
예상 진행:
- Epoch 5000-7000: 모든 cube 동시 최적화
- Epoch 7000-10000: Global success bonus 달성!

Tensorboard 신호:
- global_success: Rare spikes → Frequent!
- Task success rate: 0% → 20-50%
```

### 핵심 개선 포인트

**1. "놓는 행위" 학습 가능!**
```
Before (4차):
- Align reward만 증가 → sparse signal
- Policy가 "성공"의 가치를 모름

After (5차):
- Sub-goal bonus: +150 (huge spike!)
- Policy가 "이게 성공이구나!" 명확히 학습
- Gripper opening 자연스럽게 학습 예상
```

**2. 점진적 난이도 증가**
```
Before (4차):
- 처음부터 3개 cube 모두 고려
- 복잡한 state space

After (5차):
- Stage 1: Cube_2만 (simple!)
- Stage 2: + Cube_3
- Stage 3: 전체 최적화
- 초기 학습 속도 ↑↑
```

**3. 검증된 방법론**
```
Anca et al. (2023):
- Multi-object manipulation 성공
- Curriculum + sparse bonus 효과 입증
- 우리 task와 유사 (sequential stacking)
```

---

## 모니터링 가이드

### Tensorboard Metrics

#### Reward Terms (중요!)
```python
# Dense shaping (continuous)
Rewards/box_to_goal_distance: -5.0 ~ -0.5
Rewards/ee_to_box_distance: -5.0 ~ -0.5

# Sparse bonuses (spikes!)
Rewards/subgoal_bonus: 0 → 15.0 spikes ← 이게 보이면 성공!
Rewards/global_success: 0 → 15.0 spikes ← 최종 목표!

# Penalties
Rewards/action_penalty: -0.01 ~ 0
Rewards/table_collision_penalty: -5.0 ~ 0
Rewards/orientation_penalty: -0.1 ~ 0

# Total
Rewards/total_reward: 평소 ~1.0, spike 시 ~15.0
```

#### Curriculum Stage
```python
# env.extras에 저장된 stage 확인
print(f"Current Stage: {env.extras['curriculum_stage'][0].item()}")
print(f"Estimated Epoch: {env.extras['curriculum_epoch']}")

# 예상 전환 시점:
- Epoch 1000: Stage 1 → 2
- Epoch 5000: Stage 2 → 3
```

#### Success Rates
```python
# Stage별 success 정의:
Stage 1 Success: Cube_2 on Cube_1 (XY < 3cm, Z aligned)
Stage 2 Success: + Cube_3 on Cube_2
Stage 3 Success: All 3 cubes stacked

# Tensorboard:
Episode/success_rate: 0 → 0.8 (Stage 1)
                      0 → 0.6 (Stage 2)
                      0 → 0.3 (Stage 3, optimistic!)
```

### Visualization Checkpoints

#### Epoch 1000 (Stage 1 → 2 전환)
```bash
# Visualize Stage 1 performance
python scripts/rl_games/play.py \
    --task Isaac-Stack-RL-Franka-IK-Rel-v0 \
    --checkpoint logs/rl_games/.../nn/last_StackRL_ep_1000.pth

확인 사항:
✅ Cube_2가 cube_1 위에 안정적으로 stacking?
✅ Gripper opening 학습?
✅ Subgoal bonus spike 있었는지 Tensorboard 확인
```

#### Epoch 5000 (Stage 2 → 3 전환)
```bash
# Visualize Stage 2 performance
python scripts/rl_games/play.py \
    --checkpoint logs/rl_games/.../nn/last_StackRL_ep_5000.pth

확인 사항:
✅ 2-cube stack 성공률?
✅ Cube_3 grasping + stacking 학습?
✅ 2nd subgoal bonus spike 관찰?
```

#### Epoch 10000+ (Final)
```bash
# Full 3-cube stacking
python scripts/rl_games/play.py \
    --checkpoint logs/rl_games/.../nn/last_StackRL_ep_10000.pth

확인 사항:
✅ 3-cube stack 성공?
✅ Global success bonus 달성?
✅ 전체 task completion rate?
```

---

## 다음 단계

### 1. 학습 재시작 (Clean Start)
```bash
# 기존 logs 백업
mv logs/rl_games/stack_rl logs/rl_games/stack_rl_4th_mod_backup

# 새 학습 시작 (5차 수정 - Anca style)
python scripts/rl_games/train.py \
    --task Isaac-Stack-RL-Franka-IK-Rel-v0 \
    --headless \
    --num_envs 8192
```

**Why clean start?**
- Reward 구조가 완전히 바뀜 (mutual exclusion → Anca)
- Old policy가 new reward에 맞지 않음
- Curriculum은 Epoch 0부터 시작해야 함

### 2. Epoch 300 (Early Check)
```bash
# Tensorboard 확인
tensorboard --logdir logs/rl_games/stack_rl

확인 사항:
- ee_to_box_distance 감소 추세?
- Policy가 cube_2로 이동 시작?
- Action penalty 적당한 수준? (너무 크면 움직임 억제)
```

### 3. Epoch 1000 (Stage 1 완성)
```bash
# Visualization + Tensorboard
확인 사항:
✅ Subgoal bonus spike 관찰 (가장 중요!)
✅ Cube_2 stacking success rate 50%+
✅ Gripper opening 학습
✅ Stage 2 전환 확인 (curriculum_stage=2)
```

### 4. Epoch 5000 (Stage 2 완성)
```bash
확인 사항:
✅ 2nd subgoal bonus spike (cube_3 on cube_2)
✅ 2-cube stack success 40%+
✅ Stage 3 전환 확인
```

### 5. Epoch 10000+ (Full Task)
```bash
확인 사항:
✅ Global success bonus spike
✅ 3-cube stack 성공!
✅ Task completion rate 20-50%
```

---

## 설계 철학 (1-5차 수정 종합)

### 1차 수정 (Initial Implementation)
```
✅ IsaacGym 스타일 기본 구현
✅ Mutual exclusion reach/grasp rewards
⚠️ Policy가 observation 무시
⚠️ Gripper opening 학습 안 됨
```

### 2차 수정 (Observation Fixing Attempt)
```
✅ Observation space 점검
⚠️ 여전히 policy가 무시
→ Reward가 문제였음!
```

### 3차 수정 (Reward Kernel Change)
```
✅ Tanh kernel (sigma=3.0) 도입
✅ Policy가 observation 사용 시작!
✅ 방향 학습 가능
⚠️ 높이 제한 없음 (40-60cm까지 들어올림)
⚠️ Gripper opening 여전히 안 됨
```

### 4차 수정 (Height Penalty)
```
✅ Height penalty 추가 (max 25cm)
✅ Align weight 증가 (10.0 → 15.0)
✅ Workspace 최적화
⚠️ 여전히 gripper opening 안 됨
⚠️ Mutual exclusion의 근본적 한계
```

### 5차 수정 (Curriculum + Anca Rewards)
```
✅ 검증된 방법론 (Anca et al. 2023)
✅ Curriculum learning (stage-based)
✅ Sub-goal sparse bonuses (λ=150.0)
✅ Dense shaping + sparse bonus 조합
✅ "놓는 행위" 학습 가능!
✅ 점진적 난이도 증가
= 근본적 해결 시도!
```

### 종합 비교

| 수정 | 핵심 문제 | 해결책 | 효과 |
|-----|---------|-------|------|
| 1차 | 기본 구현 | IsaacGym style | Baseline |
| 2차 | Obs 무시 | Obs space 점검 | ❌ 효과 없음 |
| 3차 | Reward gradient | Tanh kernel + weight | ✅ 방향 학습 |
| 4차 | 높이 제한 | Height penalty + align ↑ | ✅ Workspace |
| 5차 | **Gripper opening** | **Curriculum + Sparse bonus** | ✅ 기대! |

### 철학적 변화

**Before (1-4차)**:
```
철학: "Policy에게 어떻게 가야 하는지 알려주자"
방법: Dense reward shaping (reach, grasp, align)
문제: "성공"의 가치를 알려주지 못함
결과: Gripper opening 학습 실패
```

**After (5차)**:
```
철학: "Policy에게 성공의 가치를 알려주자"
방법: Dense shaping + Sparse bonuses + Curriculum
핵심: Sub-goal 달성 순간 huge reward spike!
기대: Policy가 "아, 이게 성공이구나!" 학습
```

**Anca et al. 인사이트**:
> "Dense rewards guide the policy, but sparse bonuses teach the value of success."

- Dense: 어떻게 움직일지 (continuous guidance)
- Sparse: 무엇이 성공인지 (one-time huge signal)
- Curriculum: 언제 무엇을 배울지 (staged learning)

---

## 참고 자료

### Anca et al. (2023) Paper
- **Title**: "Achieving Goals using Reward Shaping and Curriculum Learning"
- **arXiv**: 2206.02462
- **Key Contributions**:
  - Curriculum learning for multi-object manipulation
  - Sparse sub-goal bonuses (λ=150)
  - Dense shaping (λ=5) + sparse bonus combination
  - Successful 3-object stacking in simulation

### GPT-5.1 Analysis
- **File**: `toward_more_cube_stack.md`
- **Content**:
  - Option A vs B 상세 비교
  - Anca 논문 요약
  - Hyperparameter 분석
  - Implementation 제안

### 코드 참조
- **rewards.py**: Line 1250-1745 (Anca functions)
- **stack_rl_env_cfg.py**: Line 175-331 (RewardsCfg)
- **WORKFLOW.md**: This section (5차 수정 문서)

---

## 버그 수정 기록

### Bug #1: CurriculumCfg ValueError (20251212)

**에러 메시지**:
```
ValueError: Reward term 'action_rate' not found.
```

**발생 원인**:
- 5차 수정에서 `RewardsCfg`를 완전히 재구성 (Anca style)
- 기존 reward terms (`action_rate`, `joint_vel`) 제거됨
- 하지만 `CurriculumCfg`는 여전히 이들을 참조
- 학습 시작 시 curriculum manager가 존재하지 않는 reward term 찾으려 함

**Stack Trace**:
```python
File "isaaclab/managers/curriculum_manager.py", line 197, in _prepare_terms
    self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)
...
File "isaaclab/envs/mdp/curriculums.py", line 32, in __init__
    self._term_cfg = env.reward_manager.get_term_cfg(term_name)  # ← 여기서 에러!
ValueError: Reward term 'action_rate' not found.
```

**해결 방법**:
```python
# Before (Lines 359-369)
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )

# After (Lines 359-383)
@configclass
class CurriculumCfg:
    """
    Curriculum terms for the MDP.

    20251212 (5차 수정): Empty curriculum config

    Curriculum은 이제 reward 함수 내부에서 자동 관리:
    - get_curriculum_stage()가 Epoch 기반으로 자동 전환
    - Stage 1 (0-1000) → Stage 2 (1000-5000) → Stage 3 (5000+)
    """

    pass  # Empty - curriculum은 reward 함수 내부에서 관리
```

**수정 파일**:
- `stack_rl_env_cfg.py`: Lines 359-383 (CurriculumCfg empty로 변경)

**교훈**:
- 대규모 구조 변경 시 연관된 모든 config 확인 필요
- RewardsCfg 변경 → CurriculumCfg, TerminationsCfg 등도 체크
- Anca 방식: Curriculum이 reward 함수 내부에서 관리되므로 외부 curriculum manager 불필요

---

**Last Updated**: 2025-12-12 (5차 수정 - Curriculum Learning with Anca et al. Rewards)

---

**End of Documentation**





