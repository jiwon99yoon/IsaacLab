# generate_dataset.py 상세 코드 분석

## 개요
이 스크립트는 **Imitation Learning을 위한 데이터셋 생성**을 담당합니다.
- **Input**: Source demonstrations (HDF5)
- **Output**: Generated demonstrations (HDF5)
- **핵심 기능**: Source demo를 참고하여 새로운 초기 조건에서 성공적인 trajectory 생성
- **CuRobo 역할**: Subtask 간 collision-free transition 생성 (Skillgen 모드)

---

## Line-by-Line 분석

### 1-11: 헤더 및 문서화
```python
# Copyright (c) 2024-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: Apache-2.0
"""
Main data generation script.
"""
"""Launch Isaac Sim Simulator first."""
```
**설명**: 라이선스 정보 및 스크립트 목적 명시

---

### 13-51: Command Line Arguments 설정

```python
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab environments.")
```

**주요 Arguments:**

| Argument | Type | Default | 설명 |
|----------|------|---------|------|
| `--task` | str | None | 환경 이름 (예: Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0) |
| `--generation_num_trials` | int | None | 생성할 데모 개수 (override) |
| `--num_envs` | int | 1 | 병렬 실행할 환경 개수 |
| `--input_file` | str | **Required** | Source dataset HDF5 파일 |
| `--output_file` | str | ./datasets/output_dataset.hdf5 | 출력 HDF5 파일 |
| `--pause_subtask` | bool | False | 디버깅용: subtask마다 일시정지 |
| `--enable_pinocchio` | bool | False | Pinocchio IK 라이브러리 활성화 |
| `--use_skillgen` | bool | False | **CuRobo 사용 여부 (핵심!)** |

**중요**: `--use_skillgen` 플래그가 CuRobo 활성화를 결정합니다!

---

### 53-60: Isaac Sim 시뮬레이터 초기화

```python
if args_cli.enable_pinocchio:
    import pinocchio  # AppLauncher 전에 import 필요

# Isaac Sim 앱 런처 시작
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```

**설명**:
- Isaac Sim 시뮬레이터를 먼저 시작해야 합니다 (omniverse 초기화)
- Pinocchio가 필요하면 AppLauncher 전에 import (버전 충돌 방지)

---

### 64-82: 나머지 라이브러리 Import

```python
import asyncio        # 비동기 병렬 실행
import gymnasium as gym  # 환경 인터페이스
import torch          # PyTorch

from isaaclab.envs import ManagerBasedRLMimicEnv  # Mimic 환경 기반 클래스
import isaaclab_mimic.envs  # Mimic 환경 등록
from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation, setup_env_config
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths
import isaaclab_tasks  # Task 환경 등록
```

**설명**:
- **Asyncio**: 여러 환경을 병렬로 실행하기 위한 비동기 프레임워크
- **ManagerBasedRLMimicEnv**: Mimic API를 구현한 환경 클래스
- **generation 모듈**: 실제 데이터 생성 로직 (여기서 CuRobo 사용!)

---

### 85-93: main() 함수 시작 - 환경 이름 결정

```python
def main():
    num_envs = args_cli.num_envs

    # 출력 경로 설정
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    # 예: "./datasets/output.hdf5" → output_dir="./datasets", output_file_name="output"

    # Task 이름 추출
    task_name = args_cli.task
    if task_name:
        task_name = args_cli.task.split(":")[-1]  # "Isaac-Lab:Task-v0" → "Task-v0"

    # Task 이름이 없으면 input dataset에서 추출
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)
```

**Input/Output 흐름**:
- **Input**: Command line args
- **Processing**: 경로 파싱, 환경 이름 결정
- **Output**: `env_name`, `output_dir`, `output_file_name`

---

### 96-103: 환경 설정 구성

```python
# 환경 config 설정
env_cfg, success_term = setup_env_config(
    env_name=env_name,
    output_dir=output_dir,
    output_file_name=output_file_name,
    num_envs=num_envs,
    device=args_cli.device,
    generation_num_trials=args_cli.generation_num_trials,
)
```

**setup_env_config() 내부 동작** (generation.py:137-192):
1. 환경 config 파싱
2. Success termination term 추출
3. **Recorder 설정** (HDF5 저장 담당):
   ```python
   env_cfg.recorders = ActionStateRecorderManagerCfg()
   env_cfg.recorders.dataset_export_dir_path = output_dir
   env_cfg.recorders.dataset_filename = output_file_name

   # Failed demos 저장 여부 결정
   if env_cfg.datagen_config.generation_keep_failed:
       env_cfg.recorders.dataset_export_mode = EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
   else:
       env_cfg.recorders.dataset_export_mode = EXPORT_SUCCEEDED_ONLY
   ```

**Output**:
- `env_cfg`: 환경 설정 (recorder 포함)
- `success_term`: Success 판정 함수

---

### 106-109: 환경 생성 및 타입 검증

```python
# Gymnasium 환경 생성
env = gym.make(env_name, cfg=env_cfg).unwrapped

# Mimic 환경인지 확인
if not isinstance(env, ManagerBasedRLMimicEnv):
    raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")
```

**설명**:
- `gym.make()`: Gymnasium API로 환경 생성
- `.unwrapped`: Wrapper 제거, 실제 환경 객체 접근
- **ManagerBasedRLMimicEnv**: Mimic API 필수 (`target_eef_pose_to_action` 등)

---

### 112-116: API 버전 체크

```python
# Deprecated API 체크
if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
    omni.log.warn(
        f'The "noise" parameter in the "{env_name}" environment\'s mimic API...'
    )
```

**설명**: `target_eef_pose_to_action` API의 signature 검증 (하위 호환성)

---

### 119-124: 시드 설정 및 환경 초기화

```python
# Reproducibility를 위한 시드 설정
random.seed(env.cfg.datagen_config.seed)
np.random.seed(env.cfg.datagen_config.seed)
torch.manual_seed(env.cfg.datagen_config.seed)

# 환경 초기화
env.reset()
```

**설명**: 동일한 시드로 재현 가능한 데이터 생성

---

### 127-152: **CuRobo 초기화 (핵심!)**

```python
motion_planners = None
if args_cli.use_skillgen:  # --use_skillgen 플래그가 True일 때만
    from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
    from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

    # 환경마다 독립적인 motion planner 생성
    motion_planners = {}
    for env_id in range(num_envs):
        print(f"Initializing motion planner for environment {env_id}")

        # Task에 맞는 CuRobo config 생성
        planner_config = CuroboPlannerCfg.from_task_name(env_name)

        # Visualization은 첫 번째 환경만 (성능 이유)
        if env_id != 0:
            planner_config.visualize_spheres = False
            planner_config.visualize_plan = False

        # CuroboPlanner 인스턴스 생성
        motion_planners[env_id] = CuroboPlanner(
            env=env,                      # 환경 객체
            robot=env.scene["robot"],     # 로봇 articulation
            config=planner_config,        # CuRobo 설정
            env_id=env_id,               # 환경 ID
        )

    # Skillgen 모드 활성화
    env.cfg.datagen_config.use_skillgen = True
```

**CuroboPlanner 내부 구조**:
- **URDF 로딩**: Nucleus에서 Franka URDF 다운로드
- **Collision spheres**: 로봇을 sphere로 근사화
- **World model**: 환경의 obstacle 정보
- **GPU 가속**: CUDA에서 병렬 motion planning

**중요**:
- `motion_planners[env_id]`는 각 환경의 독립적인 planner
- `num_envs=4`면 4개의 planner가 병렬로 작동

---

### 155-162: 비동기 데이터 생성 설정

```python
# Async 컴포넌트 설정
async_components = setup_async_generation(
    env=env,
    num_envs=args_cli.num_envs,
    input_file=args_cli.input_file,        # Source dataset
    success_term=success_term,
    pause_subtask=args_cli.pause_subtask,
    motion_planners=motion_planners,       # CuRobo planners 전달
)
```

**setup_async_generation() 반환값** (generation.py:195-270):
```python
{
    "tasks": [task0, task1, task2, task3],      # 각 환경의 async task
    "reset_queue": asyncio.Queue(),             # Reset 요청 큐
    "action_queue": asyncio.Queue(),            # Action 전달 큐
    "info_pool": DataGenInfoPool,               # Source demo 정보 공유
    "event_loop": asyncio event loop            # 비동기 이벤트 루프
}
```

**내부 동작**:
1. **Source dataset 로딩**: Input HDF5 파일에서 demos 읽기
2. **DataGenerator 생성**: 각 환경마다 `DataGenerator` 인스턴스
3. **Async tasks 생성**: 각 DataGenerator의 `generate()` 코루틴
4. **Queue 생성**: 환경 ↔ DataGenerator 간 통신 채널

---

### 164-172: 메인 실행 루프

```python
try:
    # 모든 async tasks를 하나의 future로 묶기
    data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))

    # 환경 메인 루프 실행 (블로킹 호출)
    env_loop(
        env,
        async_components["reset_queue"],    # Reset 큐
        async_components["action_queue"],   # Action 큐
        async_components["info_pool"],      # 공유 데이터
        async_components["event_loop"],     # Event loop
    )
```

**실행 구조**:

```
┌─────────────────────────────────────────────────────────────┐
│                      Main Thread                            │
│  ┌────────────────────────────────────────────────────┐    │
│  │ env_loop() - 환경 시뮬레이션                        │    │
│  │   while True:                                       │    │
│  │     actions = get_from_action_queue()              │    │
│  │     env.step(actions)                              │    │
│  │     check_termination()                            │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
           ↑ action_queue              ↓ observations
┌─────────────────────────────────────────────────────────────┐
│                   Asyncio Thread Pool                       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │ DataGen 0    │ │ DataGen 1    │ │ DataGen 2    │       │
│  │ (CuRobo 0)   │ │ (CuRobo 1)   │ │ (CuRobo 2)   │       │
│  │              │ │              │ │              │       │
│  │ generate()   │ │ generate()   │ │ generate()   │       │
│  │   plan()     │ │   plan()     │ │   plan()     │       │
│  │   action →   │ │   action →   │ │   action →   │       │
│  └──────────────┘ └──────────────┘ └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

**env_loop() 동작** (generation.py:65-134):
1. Action queue에서 모든 환경의 action 수집
2. `env.step(actions)` 실행
3. 성공/실패 통계 업데이트
4. Termination 조건 체크 (`num_success >= generation_num_trials`)

---

### 173-192: 정리 및 종료

```python
except asyncio.CancelledError:
    print("Tasks were cancelled.")
finally:
    # Async tasks 취소
    data_gen_tasks.cancel()
    try:
        async_components["event_loop"].run_until_complete(data_gen_tasks)
    except asyncio.CancelledError:
        print("Remaining async tasks cancelled and cleaned up.")

    # Motion planner 정리
    if motion_planners is not None:
        for env_id, planner in motion_planners.items():
            # Visualizer 닫기
            if getattr(planner, "plan_visualizer", None) is not None:
                print(f"Closing plan visualizer for environment {env_id}")
                planner.plan_visualizer.close()
                planner.plan_visualizer = None
        motion_planners.clear()
```

**정리 과정**:
1. 진행 중인 async tasks 취소
2. CuRobo visualizer 닫기 (Rerun viewer)
3. Motion planner 메모리 해제

---

### 195-201: 프로그램 엔트리 포인트

```python
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    # Isaac Sim 앱 종료
    simulation_app.close()
```

---

## Input/Output 데이터 흐름

### Input
1. **Source Dataset (HDF5)**:
   ```
   input_file.hdf5
   └── data/
       ├── demo_0/
       │   ├── actions: (T, 7)
       │   ├── obs/eef_pos: (T, 3)
       │   ├── subtasks: [(start, end, subtask_id), ...]
       │   └── initial_state/...
       └── demo_1/...
   ```

2. **Environment Config**:
   - Task name
   - Datagen config (generation_num_trials, use_skillgen, etc.)
   - Robot URDF, scene objects

### Processing
1. **DataGenerator.generate()** (각 환경마다):
   - Source demo에서 subtask 선택
   - Target pose 추출
   - **CuRobo로 collision-free path 계획** (skillgen 모드)
   - Waypoint 실행
   - Success 체크

2. **env_loop()**:
   - Actions 수집 및 `env.step()` 실행
   - Episode 데이터 recording
   - Reset 시 episode export (HDF5 저장)

### Output
1. **Generated Dataset (HDF5)**:
   ```
   output_file.hdf5
   └── data/
       ├── @total: 20
       ├── demo_0/
       │   ├── @success: True
       │   ├── actions: (T', 7)
       │   ├── obs/eef_pos: (T', 3)
       │   └── initial_state/...
       └── demo_1/...

   output_file_failed.hdf5  # 실패한 demos
   └── data/
       └── demo_0/ (@success: False)
   ```

---

## CuRobo 사용 위치 및 방법

### 1. CuRobo 초기화 (generate_dataset.py:127-152)
```python
motion_planners = {}
for env_id in range(num_envs):
    planner_config = CuroboPlannerCfg.from_task_name(env_name)
    motion_planners[env_id] = CuroboPlanner(...)
```

### 2. CuRobo 전달 (generate_dataset.py:155-162)
```python
async_components = setup_async_generation(
    ...
    motion_planners=motion_planners,
)
```

### 3. CuRobo 실제 사용 (data_generator.py:650-780)

**DataGenerator.generate() 내부**:

```python
# Subtask 간 전환이 필요할 때
if self.env_cfg.datagen_config.use_skillgen and motion_planner:
    # Target pose 설정
    target_eef_pose = ...  # Source demo에서 추출

    # CuRobo로 motion planning
    planning_success = motion_planner.update_world_and_plan_motion(
        target_pose=target_eef_pose,
        expected_attached_object=attached_object,
        env_id=env_id,
        step_size=0.01,          # 1cm steps
        enable_retiming=True,    # Time-optimal retiming
    )

    if planning_success:
        # Planned trajectory를 waypoints로 변환
        planned_poses = motion_planner.get_planned_poses()
        waypoints = self._convert_planned_trajectory_to_waypoints(
            motion_planner, target_gripper_action
        )

        # Waypoints 실행
        for waypoint in waypoints:
            action = self.env.target_eef_pose_to_action(
                target_eef_pose=waypoint.eef_pose,
                target_gripper_action=waypoint.gripper_action,
            )
            await self._apply_action(action, ...)
    else:
        print(f"Motion planning failed!")
        return {"success": False}
```

**CuRobo의 역할**:
- **Collision detection**: 환경의 obstacle과 충돌 체크
- **Path planning**: Start pose → Target pose 경로 생성
- **Trajectory optimization**: Smooth, time-optimal trajectory
- **Retiming**: Velocity constraints 적용

---

## 전체 실행 흐름 요약

```
1. Command Line Args 파싱
   ├─ --input_file: Source dataset
   ├─ --output_file: Generated dataset
   ├─ --use_skillgen: CuRobo 사용 여부
   └─ --num_envs: 병렬 환경 개수

2. Isaac Sim 시뮬레이터 초기화
   └─ AppLauncher(args_cli)

3. 환경 설정
   ├─ setup_env_config()
   │   └─ Recorder 설정 (HDF5 export)
   └─ gym.make(env_name, cfg=env_cfg)

4. CuRobo 초기화 (--use_skillgen=True인 경우)
   └─ motion_planners[env_id] = CuroboPlanner(...)

5. Async 데이터 생성 설정
   ├─ Source dataset 로딩
   ├─ DataGenerator 생성 (각 환경마다)
   └─ Async tasks 생성

6. 메인 실행 루프
   ├─ Asyncio: DataGenerator.generate()
   │   ├─ Source demo 선택
   │   ├─ Subtask 실행
   │   ├─ **CuRobo motion planning** (subtask 전환 시)
   │   └─ Action 생성
   └─ env_loop(): env.step(actions)
       ├─ Episode recording
       └─ Success 체크

7. Termination
   ├─ num_success >= generation_num_trials
   ├─ Episode export (HDF5 저장)
   └─ CuRobo cleanup

8. Output
   ├─ output_file.hdf5 (성공 demos)
   └─ output_file_failed.hdf5 (실패 demos)
```

---

## 핵심 요약

1. **입력**: Source demonstrations (HDF5) + Environment
2. **처리**:
   - Source demo에서 subtask pattern 추출
   - 새로운 초기 조건에서 실행
   - **CuRobo로 subtask 간 collision-free transition 생성**
   - Episode recording 및 HDF5 저장
3. **출력**: Generated demonstrations (성공/실패 분리)

**CuRobo의 역할**:
- Subtask A → Subtask B로 전환할 때 collision-free path 생성
- **실제 manipulation (grasp, place)은 source demo 재현**
- Skillgen 모드에서만 활성화
