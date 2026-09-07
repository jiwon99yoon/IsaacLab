# Motion Planning 실패 문제 해결 가이드

## 현재 문제
```
Env 0: Motion planning failed for franka
```

모든 target pose에 대해 motion planning이 실패하고 있습니다.

## 가능한 원인 및 해결 방법

### 1. Dataset 문제

**확인:**
```bash
python3 inspect_hdf5.py datasets/annotated_dataset_skillgen.hdf5 --stats
```

**예상 문제:**
- Dataset이 비어있음 (total: 0)
- Subtask term signals가 없음

**해결:**
```bash
# Annotation 다시 실행 (auto mode)
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0 \
    --input_file ./datasets/dataset.hdf5 \
    --output_file ./datasets/annotated_dataset_skillgen.hdf5 \
    --auto \
    --headless
```

### 2. CuRobo 설정이 너무 엄격

**현재 설정 (curobo_planner_cfg.py:433):**
```python
collision_activation_distance = 0.01  # 너무 작음
position_threshold = 0.005  # 너무 엄격
rotation_threshold = 0.05
max_planning_attempts = 1  # 재시도 없음
```

**해결 방법 A: 설정 완화**

`source/isaaclab_mimic/isaaclab_mimic/motion_planners/curobo/curobo_planner_cfg.py` 수정:

```python
@classmethod
def franka_stack_cube_config(cls) -> "CuroboPlannerCfg":
    config = cls.franka_config()
    config.static_objects = ["table"]
    config.visualize_plan = False
    config.debug_planner = True  # 디버그 활성화
    config.motion_noise_scale = 0.02

    # 완화된 설정
    config.collision_activation_distance = 0.05  # 0.01 -> 0.05
    config.position_threshold = 0.01  # 0.005 -> 0.01
    config.rotation_threshold = 0.1  # 0.05 -> 0.1
    config.max_planning_attempts = 3  # 1 -> 3 (재시도)

    config.approach_distance = 0.05
    config.retreat_distance = 0.05
    config.surface_sphere_radius = 0.01
    config.get_world_config = lambda: config._get_world_config_with_table_adjustment()
    return config
```

**해결 방법 B: 환경 변수로 디버그 모드 활성화**

Generate dataset 실행 시 수정:

```python
# scripts/imitation_learning/isaaclab_mimic/generate_dataset.py
# Line 136 근처 수정

planner_config = CuroboPlannerCfg.from_task_name(env_name)

# 디버그 옵션 추가
planner_config.debug_planner = True
planner_config.collision_activation_distance = 0.05
planner_config.max_planning_attempts = 3

if env_id != 0:
    planner_config.visualize_spheres = False
    planner_config.visualize_plan = False
```

### 3. Target Pose가 도달 불가능

**확인 방법:**

`check_target_poses.py` 생성:
```python
#!/usr/bin/env python3
import h5py
import numpy as np

dataset_path = "./datasets/annotated_dataset_skillgen.hdf5"

with h5py.File(dataset_path, 'r') as f:
    demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]

    if len(demo_keys) == 0:
        print("ERROR: No demos in dataset!")
        exit(1)

    demo = f['data'][demo_keys[0]]

    if 'obs/datagen_info/eef_pose' in demo:
        eef_poses = demo['obs/datagen_info/eef_pose'][:]
        print(f"EEF poses: {eef_poses.shape}")

        # Extract positions from 4x4 matrices
        if len(eef_poses.shape) == 3:  # (T, 4, 4)
            positions = eef_poses[:, :3, 3]
        else:
            print("Unexpected shape!")
            exit(1)

        print(f"\nPosition ranges:")
        print(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
        print(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
        print(f"  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")

        # Franka 도달 범위 확인
        unreachable = (
            (positions[:, 0] < 0.2) | (positions[:, 0] > 0.8) |
            (positions[:, 1] < -0.4) | (positions[:, 1] > 0.4) |
            (positions[:, 2] < 0.05) | (positions[:, 2] > 0.6)
        )

        if unreachable.any():
            print(f"\nWARNING: {unreachable.sum()} poses may be unreachable!")
    else:
        print("No datagen_info found - dataset not annotated properly")
```

실행:
```bash
python3 check_target_poses.py
```

### 4. Franka 모델 불일치

**확인:**

```bash
# CuRobo가 사용하는 URDF 확인
./isaaclab.sh -p check_curobo_franka_files.py --headless

# IsaacLab 환경의 Franka 확인
find source/isaaclab_assets -name "*franka*" -o -name "*panda*"
```

**만약 당신이 Franka 모델을 수정했다면:**

Option A: Nucleus URDF 업데이트 (복잡)
Option B: 커스텀 URDF 경로 지정

`curobo_planner_cfg.py` 수정:
```python
@classmethod
def franka_stack_cube_config(cls) -> "CuroboPlannerCfg":
    # 커스텀 URDF 경로
    custom_urdf = "/path/to/your/modified/franka.urdf"

    robot_cfg_file = cls._create_temp_robot_yaml("franka.yml", custom_urdf)

    return cls(
        robot_config_file=robot_cfg_file,
        ...
    )
```

### 5. Collision World 문제

**디버그 출력 활성화:**

`curobo_planner.py`에 디버그 추가:
```python
def update_world_and_plan_motion(self, ...):
    # World 업데이트
    self.update_world()

    # 디버그: collision objects 확인
    if self.config.debug_planner:
        print(f"[DEBUG] Current collision objects:")
        # Print world state

    # Motion planning
    success = self.plan_motion(...)

    if not success and self.config.debug_planner:
        print(f"[DEBUG] Planning failed for target: {target_pose}")
        # Print detailed failure reason

    return success
```

### 6. CPU vs GPU 문제

당신의 명령어에 `--device cpu`가 있습니다:
```bash
--device cpu  # CuRobo는 GPU 최적화되어 있음
```

**시도:**
```bash
# GPU 사용 (권장)
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --device cuda:0 \
    --num_envs 1 \
    --generation_num_trials 10 \
    --input_file ./datasets/annotated_dataset_skillgen.hdf5 \
    --output_file ./datasets/generated_dataset_small_skillgen_cube_stack.hdf5 \
    --task Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0 \
    --use_skillgen
```

## 빠른 진단 체크리스트

```bash
# 1. Dataset 확인
python3 inspect_hdf5.py datasets/annotated_dataset_skillgen.hdf5 --stats

# 2. Annotation 확인
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0 \
    --input_file ./datasets/dataset.hdf5 \
    --output_file ./datasets/annotated_dataset_skillgen.hdf5 \
    --auto

# 3. Visualization 모드로 확인
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --device cuda:0 \
    --num_envs 1 \
    --generation_num_trials 1 \
    --input_file ./datasets/annotated_dataset_skillgen.hdf5 \
    --output_file ./datasets/test.hdf5 \
    --task Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0 \
    --use_skillgen
    # --headless 제거 -> 시각적으로 확인

# 4. Debug 모드 활성화
# curobo_planner_cfg.py에서 debug_planner = True로 설정 후 재실행
```

## 추천 해결 순서

1. **Dataset 재생성** (가장 가능성 높음)
   ```bash
   ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
       --task Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0 \
       --input_file ./datasets/dataset.hdf5 \
       --output_file ./datasets/annotated_dataset_skillgen_new.hdf5 \
       --auto \
       --headless
   ```

2. **CuRobo 설정 완화**
   - `curobo_planner_cfg.py` 수정
   - `collision_activation_distance = 0.05`
   - `max_planning_attempts = 3`

3. **GPU 사용**
   - `--device cuda:0`

4. **Visualization으로 확인**
   - `--headless` 제거
   - 실제로 무엇이 문제인지 눈으로 확인

5. **Debug 출력 활성화**
   - `debug_planner = True`
   - 실패 원인 로그 확인

## 참고

- CuRobo는 GPU에서 가장 잘 작동합니다
- Motion planning은 collision-free path를 찾는 것이므로, 도달 불가능한 pose나 collision이 있으면 실패
- Dataset의 품질이 중요 - Nvidia 제공 dataset도 썩 좋지 않다고 했으니 직접 재생성 권장
