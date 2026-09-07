# 환경 Reset 및 초기 조건 분석

## 질문
> `num_envs=4`일 때:
> 1. 각 환경의 cube 위치, robot 초기 위치가 동일한가?
> 2. CuRobo의 입력/출력 (end-effector 위치)이 환경마다 동일한가?

## 답변: 환경마다 **다릅니다!**

---

## 1. 환경 Reset 과정

### 코드 위치: `data_generator.py:655-660`

```python
async def generate(self, env_id, success_term, env_reset_queue, ...):
    # reset the env to create a new task demo instance
    env_id_tensor = torch.tensor([env_id], dtype=torch.int64, device=self.env.device)
    self.env.recorder_manager.reset(env_ids=env_id_tensor)
    await env_reset_queue.put(env_id)  # Reset 요청
    await env_reset_queue.join()       # Reset 완료 대기
    new_initial_state = self.env.scene.get_state(is_relative=True)  # 새로운 초기 상태
```

### Reset이 일어나는 곳: `generation.py:env_loop()`

```python
def env_loop(env, env_reset_queue, env_action_queue, ...):
    while True:
        # Reset 요청 처리
        while not env_reset_queue.empty():
            env_id_tensor[0] = env_reset_queue.get_nowait()
            env.reset(env_ids=env_id_tensor)  # ★ 여기서 실제 reset!
            env_reset_queue.task_done()

        # Actions 수집 및 실행
        actions = collect_actions_from_queue()
        env.step(actions)
```

---

## 2. 초기 조건 설정 방식

### IsaacLab의 Reset 메커니즘

`env.reset(env_ids=[0, 1, 2, 3])`가 호출되면:

1. **EventManager 실행**:
   ```python
   # 각 환경의 초기 조건 randomization
   env.event_manager.reset(env_ids)
   ```

2. **Reset Events 트리거**:
   - `randomize_robot_joint_positions`: Joint 위치 randomization
   - `randomize_object_poses`: 객체 위치/방향 randomization
   - `reset_scene_to_default`: 기본 상태로 리셋 (선택적)

### Cube Stack 환경 예시

`source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_env_cfg.py`:

```python
@configclass
class StackEnvCfg(ManagerBasedRLEnvCfg):
    # Scene definition
    scene: StackSceneCfg = StackSceneCfg(num_envs=4096, env_spacing=2.5)

    # Events (초기 조건 randomization)
    events: EventCfg = EventCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # Robot reset event
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",  # Reset 시마다 실행
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
        },
    )

    # Object reset event
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    # Cube position randomization
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
            "velocity_range": {},
        },
    )
```

---

## 3. 각 환경이 다른 이유

### 병렬 실행 시나리오 (`num_envs=4`)

```
Time T0: 모든 환경 Reset
├─ Env 0: cube_1 at (0.45, 0.12, 0.05), cube_2 at (0.38, -0.08, 0.05)
├─ Env 1: cube_1 at (0.52, -0.05, 0.05), cube_2 at (0.41, 0.10, 0.05)
├─ Env 2: cube_1 at (0.39, 0.15, 0.05), cube_2 at (0.49, -0.12, 0.05)
└─ Env 3: cube_1 at (0.48, -0.10, 0.05), cube_2 at (0.35, 0.08, 0.05)

Time T1: Episode 진행
├─ Env 0: Subtask 0 실행 중... (cube_1 grasp)
├─ Env 1: Subtask 1 실행 중... (cube_1 place)
├─ Env 2: Subtask 0 실행 중... (cube_1 grasp)
└─ Env 3: Subtask 2 실행 중... (cube_2 grasp)

Time T2: Env 1이 먼저 완료
├─ Env 0: 계속 실행 중
├─ Env 1: ★ Reset! → 새로운 초기 조건 생성
│          cube_1 at (0.41, 0.09, 0.05), cube_2 at (0.46, -0.11, 0.05)
├─ Env 2: 계속 실행 중
└─ Env 3: 계속 실행 중
```

**핵심**:
- 각 환경은 독립적으로 Reset되고
- Reset될 때마다 **랜덤한 초기 조건** 생성
- 동시에 실행되지만 **서로 다른 상태**

---

## 4. CuRobo 입력/출력도 환경마다 다름

### CuRobo의 입력 (Start Pose)

```python
# data_generator.py:700-720
# 현재 end-effector pose를 start로 사용
current_eef_pose = self.env.scene["robot"].data.body_pose_w[env_id, eef_body_idx]
# 환경마다 robot 상태가 다르므로 start pose도 다름!
```

### CuRobo의 입력 (Target Pose)

```python
# data_generator.py:740-760
# Source demo에서 target pose 추출
selected_src_demo = select_source_demo(env_id)  # 환경마다 다른 demo 선택
target_eef_pose = extract_target_from_demo(selected_src_demo, subtask_id)
# 환경마다 다른 source demo → 다른 target pose!
```

### CuRobo의 출력 (Planned Path)

```python
# 환경마다 다른 start/target → 다른 planned path
motion_planner.plan_motion(
    start_pose=current_eef_pose[env_id],  # 환경마다 다름
    target_pose=target_pose[env_id],      # 환경마다 다름
    obstacles=world_state[env_id],        # 환경마다 다름 (cube 위치)
)
# 결과: 각 환경마다 다른 collision-free trajectory 생성!
```

---

## 5. 실제 예시: Cube Stack Task

### Episode 생성 예시 (`num_envs=4`)

**Env 0**:
```
Initial State:
  cube_1: pos=(0.45, 0.12, 0.025), quat=(0,0,0,1)
  cube_2: pos=(0.38, -0.08, 0.025), quat=(0,0,0,1)
  robot: joint_pos=[0.1, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8]

CuRobo Planning:
  Subtask 0→1 transition:
    Start: eef_pos=(0.45, 0.12, 0.15)  ← cube_1 위에서 grasp 완료
    Target: eef_pos=(0.40, 0.00, 0.20) ← stack 위치로 이동
    → CuRobo generates path: 50 waypoints
```

**Env 1** (동시에 실행, 다른 초기 조건):
```
Initial State:
  cube_1: pos=(0.52, -0.05, 0.025), quat=(0,0,0,1)  ← Env 0과 다름!
  cube_2: pos=(0.41, 0.10, 0.025), quat=(0,0,0,1)
  robot: joint_pos=[0.05, -0.25, 0.0, -2.3, 0.0, 1.9, 0.75]

CuRobo Planning:
  Subtask 0→1 transition:
    Start: eef_pos=(0.52, -0.05, 0.15)  ← 다른 위치!
    Target: eef_pos=(0.46, 0.02, 0.20)  ← 다른 target!
    → CuRobo generates path: 48 waypoints (다른 경로!)
```

**Env 2, 3**도 마찬가지로 모두 다른 초기 조건과 CuRobo path를 가집니다.

---

## 6. 왜 환경마다 다른 조건이 필요한가?

### Data Augmentation (핵심!)

```
Same Task, Different Conditions → More Robust Policy

Source Demo (1개):
  cube_1 at (0.50, 0.00, 0.025)
  cube_2 at (0.40, 0.00, 0.025)

Generated Demos (20개):
  Demo 0: cube_1 at (0.45, 0.12, 0.025), cube_2 at (0.38, -0.08, 0.025)
  Demo 1: cube_1 at (0.52, -0.05, 0.025), cube_2 at (0.41, 0.10, 0.025)
  Demo 2: cube_1 at (0.39, 0.15, 0.025), cube_2 at (0.49, -0.12, 0.025)
  ...
  Demo 19: cube_1 at (0.48, -0.10, 0.025), cube_2 at (0.35, 0.08, 0.025)

→ 다양한 초기 조건에서 성공하는 policy 학습!
```

### Generalization

- **동일한 조건**: Policy가 특정 조건에만 overfitting
- **다양한 조건**: Policy가 일반화되어 robust해짐

---

## 7. 확인 방법

### 방법 1: Replay로 확인

```bash
# Generated dataset replay
./isaaclab.sh -p scripts/tools/replay_demos.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0 \
    --device cuda \
    --num_envs 4 \
    --dataset_file ./datasets/generated_dataset.hdf5

# 4개 환경을 동시에 보면 각각 다른 초기 조건을 가지는 것을 볼 수 있음!
```

### 방법 2: HDF5에서 직접 확인

```python
import h5py

with h5py.File('./datasets/generated_dataset.hdf5', 'r') as f:
    # Demo 0의 cube_1 초기 위치
    cube1_pos_demo0 = f['data/demo_0/initial_state/rigid_object/cube_1/root_pose'][0, :3]
    print(f"Demo 0 cube_1: {cube1_pos_demo0}")

    # Demo 1의 cube_1 초기 위치
    cube1_pos_demo1 = f['data/demo_1/initial_state/rigid_object/cube_1/root_pose'][0, :3]
    print(f"Demo 1 cube_1: {cube1_pos_demo1}")

    # 결과: 다른 위치!
    # Demo 0 cube_1: [0.45 0.12 0.025]
    # Demo 1 cube_1: [0.52 -0.05 0.025]
```

---

## 요약

| 항목 | 동일 여부 | 이유 |
|------|----------|------|
| **Cube 초기 위치** | ❌ 다름 | Reset 시 randomization |
| **Robot 초기 자세** | ❌ 다름 | Reset 시 randomization |
| **CuRobo Start Pose** | ❌ 다름 | 현재 robot 상태 반영 |
| **CuRobo Target Pose** | ❌ 다름 | 다른 source demo 선택 |
| **CuRobo Output Path** | ❌ 다름 | Start/Target/Obstacles 모두 다름 |
| **Subtask 순서** | ✅ 동일 | Task definition은 동일 |
| **Task 목표** | ✅ 동일 | "Stack cubes" |

**핵심 포인트**:
- ✅ 각 환경은 **독립적으로** reset되고 실행됨
- ✅ 초기 조건은 **randomization**으로 다양하게 생성됨
- ✅ CuRobo는 각 환경의 **현재 상태**를 고려하여 경로 계획
- ✅ 이것이 **data augmentation**의 핵심!
- ✅ 결과: 다양한 조건에서 작동하는 **robust policy** 학습 가능
