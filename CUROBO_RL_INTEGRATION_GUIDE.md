# CuRobo와 IsaacLab Reinforcement Learning 통합 가이드

## 목표
IsaacLab의 reinforcement learning 환경에 CuRobo motion planning을 통합하여:
- 특정 위치까지는 CuRobo의 motion planning으로 이동
- 도착 후 강화학습 알고리즘 실행

## 1. HDF5 데이터셋 읽기

### 기본 사용법

```python
import h5py

# HDF5 파일 열기
with h5py.File('dataset.hdf5', 'r') as f:
    # 구조 확인
    print("Top-level keys:", list(f.keys()))

    # Demo 목록
    demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]

    # 특정 demo 읽기
    demo = f['data/demo_0']

    # Actions 읽기
    actions = demo['actions'][:]  # shape: (T, action_dim)

    # Observations 읽기
    eef_pos = demo['obs/eef_pos'][:]  # End-effector position
    joint_pos = demo['obs/joint_pos'][:]  # Joint positions

    # Initial state
    initial_joint_pos = demo['initial_state/articulation/robot/joint_position'][:]
```

### HDF5 구조를 텍스트로 변환

```bash
# HDF5 파일을 텍스트로 변환
python3 inspect_hdf5.py --input datasets/dataset.hdf5 --output datasets/dataset_structure.txt

# 다른 파일들
python3 inspect_hdf5.py --input datasets/annotated_dataset.hdf5 --output datasets/annotated_structure.txt
python3 inspect_hdf5.py --input datasets/generated_dataset.hdf5 --output datasets/generated_structure.txt

# 도움말
python3 inspect_hdf5.py --help
```


## 2. IsaacLab 데이터셋 구조

```
dataset.hdf5
└── data/
    ├── @env_args: 환경 정보
    ├── @total: 총 샘플 수
    ├── demo_0/
    │   ├── @num_samples: 236
    │   ├── @success: True
    │   ├── actions: (236, 7) - 로봇 액션 (position + gripper)
    │   ├── initial_state/
    │   │   ├── articulation/robot/ - 로봇 초기 상태
    │   │   └── rigid_object/cube_X/ - 객체 초기 상태
    │   ├── obs/
    │   │   ├── eef_pos: (236, 3) - End-effector 위치
    │   │   ├── eef_quat: (236, 4) - End-effector 방향
    │   │   ├── joint_pos: (236, 9) - Joint 위치
    │   │   ├── gripper_pos: (236, 2) - Gripper 상태
    │   │   └── object: (236, 39) - 객체 상태
    │   └── states/
    │       ├── articulation/robot/ - 로봇 전체 상태
    │       └── rigid_object/cube_X/ - 객체 전체 상태
    ├── demo_1/
    └── ...
```

## 3. IsaacLab에서 CuRobo 사용 방법

### 현재 구조: Imitation Learning (Skillgen)

IsaacLab에서 CuRobo는 **imitation learning**의 skillgen 모드에서 사용됩니다:

```python
# source/isaaclab_mimic/isaaclab_mimic/datagen/data_generator.py
if self.env_cfg.datagen_config.use_skillgen and motion_planner:
    # CuRobo로 collision-free trajectory 계획
    planning_success = motion_planner.update_world_and_plan_motion(
        target_pose=target_eef_pose,
        expected_attached_object=expected_attached_object,
        env_id=env_id,
        step_size=motion_planner.step_size,
        enable_retiming=True,
    )

    if planning_success:
        # 계획된 trajectory를 waypoints로 변환하여 실행
        waypoints = self._convert_planned_trajectory_to_waypoints(
            motion_planner, target_gripper_action
        )
```

### CuroboPlanner 초기화

```python
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

# 환경별로 motion planner 생성
motion_planners = {}
for env_id in range(num_envs):
    # Task name으로부터 config 생성
    planner_config = CuroboPlannerCfg.from_task_name(env_name)

    # Visualization 설정 (첫 번째 env만)
    if env_id != 0:
        planner_config.visualize_spheres = False
        planner_config.visualize_plan = False

    motion_planners[env_id] = CuroboPlanner(
        env=env,
        robot=env.scene["robot"],
        config=planner_config,
        env_id=env_id,
    )
```

### CuroboPlanner 주요 API

```python
class CuroboPlanner:
    def update_world(self) -> None:
        """현재 stage에서 collision world 업데이트"""

    def plan_motion(
        self,
        target_pose: torch.Tensor,
        current_joint_positions: torch.Tensor | None = None,
        ...
    ) -> bool:
        """Target pose로 motion plan 생성"""

    def update_world_and_plan_motion(
        self,
        target_pose: torch.Tensor,
        expected_attached_object: str | None = None,
        env_id: int = 0,
        step_size: float | None = None,
        enable_retiming: bool = False,
    ) -> bool:
        """World 업데이트 후 motion plan 생성 (통합 API)"""

    def get_planned_poses(self) -> list[torch.Tensor]:
        """계획된 trajectory의 poses 반환"""

    def attach_object(self, object_name: str, parent_link: str) -> bool:
        """객체를 robot link에 attach"""

    def detach_all_objects(self) -> None:
        """모든 attached 객체 detach"""
```

## 4. RL 환경에 CuRobo 통합 방법

### 아이디어: Hybrid RL-Planning 시스템

```python
class HybridRLPlanningEnv(ManagerBasedRLEnv):
    """
    CuRobo motion planning과 RL을 결합한 환경
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # CuRobo planner 초기화
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

        self.motion_planners = {}
        for env_id in range(self.num_envs):
            planner_config = CuroboPlannerCfg.from_task_name(cfg.env_name)
            self.motion_planners[env_id] = CuroboPlanner(
                env=self,
                robot=self.scene["robot"],
                config=planner_config,
                env_id=env_id,
            )

        # Phase tracking: "planning" or "rl"
        self.phase = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        # 0: planning phase, 1: RL phase

        # Planned trajectories
        self.planned_trajectories = {}
        self.trajectory_indices = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        # Reset to planning phase
        self.phase[env_ids] = 0
        self.trajectory_indices[env_ids] = 0

        # Plan initial trajectory to approach pose
        for env_id in env_ids:
            target_pose = self._get_approach_pose(env_id)

            success = self.motion_planners[env_id].update_world_and_plan_motion(
                target_pose=target_pose,
                env_id=env_id,
                step_size=0.01,  # 1cm steps
                enable_retiming=True,
            )

            if success:
                planned_poses = self.motion_planners[env_id].get_planned_poses()
                self.planned_trajectories[env_id] = planned_poses
            else:
                print(f"Motion planning failed for env {env_id}")
                # Fallback: direct RL control
                self.phase[env_id] = 1

    def step(self, action: torch.Tensor):
        """
        Hybrid step:
        - Planning phase: Execute planned trajectory
        - RL phase: Use RL action
        """
        processed_action = torch.zeros_like(action)

        for env_id in range(self.num_envs):
            if self.phase[env_id] == 0:  # Planning phase
                if env_id in self.planned_trajectories:
                    traj = self.planned_trajectories[env_id]
                    idx = self.trajectory_indices[env_id]

                    if idx < len(traj):
                        # Use planned pose as action
                        target_pose = traj[idx]
                        processed_action[env_id] = self._pose_to_action(target_pose)
                        self.trajectory_indices[env_id] += 1
                    else:
                        # Planning phase complete -> switch to RL
                        print(f"Env {env_id}: Switching to RL phase")
                        self.phase[env_id] = 1
                        processed_action[env_id] = action[env_id]
                else:
                    # No planned trajectory, use RL
                    processed_action[env_id] = action[env_id]
            else:  # RL phase
                processed_action[env_id] = action[env_id]

        # Execute action
        return super().step(processed_action)

    def _get_approach_pose(self, env_id: int) -> torch.Tensor:
        """
        Define approach pose for the task
        예: 물체 위 10cm 위치
        """
        object_pos = self.scene["object"].data.root_pos_w[env_id]
        approach_pos = object_pos + torch.tensor([0, 0, 0.1], device=self.device)

        # Gripper pointing down
        approach_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)

        # 4x4 transformation matrix
        from isaaclab.utils.math import matrix_from_quat, make_pose
        rot_mat = matrix_from_quat(approach_quat)
        approach_pose = make_pose(approach_pos, rot_mat)

        return approach_pose

    def _pose_to_action(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Convert 4x4 pose to action format
        """
        # Extract position and orientation
        pos = pose[:3, 3]
        quat = # extract quaternion from rotation matrix

        # Convert to delta action (depending on your action space)
        current_eef_pos = self.scene["robot"].data.eef_pos[env_id]
        delta_pos = pos - current_eef_pos

        return torch.cat([delta_pos, quat, gripper_action])
```

### 사용 예시

```python
# Train script
env = gym.make("Isaac-MyTask-v0", cfg=env_cfg)

# Episode loop
obs, _ = env.reset()
for step in range(max_steps):
    # RL policy
    action = policy(obs)

    # Env will automatically use:
    # - Planned trajectory during planning phase
    # - RL action during RL phase
    obs, reward, done, truncated, info = env.step(action)

    if done or truncated:
        break
```

## 5. 참고 파일

### CuRobo 관련
- `source/isaaclab_mimic/isaaclab_mimic/motion_planners/curobo/curobo_planner.py`
- `source/isaaclab_mimic/isaaclab_mimic/motion_planners/curobo/curobo_planner_cfg.py`

### Imitation Learning (참고용)
- `scripts/imitation_learning/isaaclab_mimic/generate_dataset.py` - CuRobo를 사용한 dataset 생성
- `source/isaaclab_mimic/isaaclab_mimic/datagen/data_generator.py` - CuRobo 통합 예시

### Environment
- `source/isaaclab/isaaclab/envs/manager_based_rl_env.py` - RL 환경 base class
- `source/isaaclab_mimic/isaaclab_mimic/envs/pick_place_mimic_env.py` - Mimic 환경 예시

## 6. 다음 단계

1. **CuRobo API 테스트**
   ```bash
   # 기존 imitation learning 코드로 CuRobo 동작 확인
   python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
       --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --input_file datasets/dataset.hdf5 \
       --output_file datasets/test_curobo.hdf5 \
       --use_skillgen \
       --num_envs 1 \
       --headless
   ```

2. **Hybrid 환경 구현**
   - 위의 `HybridRLPlanningEnv` 클래스를 구현
   - `manager_based` RL 환경을 상속
   - Planning phase와 RL phase 전환 로직 추가

3. **Reward 설계**
   - Planning phase: trajectory following reward
   - RL phase: task-specific reward
   - Phase transition bonus

4. **Training**
   - PPO/SAC 등 RL 알고리즘 적용
   - Planning phase를 warm-up으로 사용
   - RL phase에서 fine-tuning

## 7. 추가 고려사항

### Collision Checking
- CuRobo는 자체적으로 collision checking 수행
- RL phase에서도 필요시 CuRobo의 collision checker 활용 가능

### Sim-to-Real Transfer
- CuRobo는 real robot에도 적용 가능
- Planning phase는 sim과 real 모두 동일하게 작동
- RL phase만 domain randomization 필요

### Computational Cost
- CuRobo는 GPU에서 빠르게 동작
- Multi-env 설정시 각 env마다 planner 필요
- Visualization은 첫 번째 env만 활성화 권장

## 8. 문제 해결

### HDF5 데이터셋 문제
현재 발견된 문제:
- `annotated_dataset.hdf5`: 비어있음 (total: 0)
- `generated_dataset_small.hdf5`: 손상됨

해결 방법:
```bash
# 1. Annotate dataset (auto mode)
python scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
    --input_file datasets/dataset.hdf5 \
    --output_file datasets/annotated_dataset_new.hdf5 \
    --auto \
    --headless

# 2. Generate new dataset
python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
    --input_file datasets/annotated_dataset_new.hdf5 \
    --output_file datasets/generated_dataset_new.hdf5 \
    --generation_num_trials 50 \
    --num_envs 4 \
    --headless
```

## 참고 자료
- IsaacLab Imitation Learning: https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/skillgen.html
- CuRobo Documentation: https://curobo.org/
- 벨로그 튜토리얼: https://velog.io/@yjseo/0707UR5로-Isaac-Lab의-Imitation-Learning-사용하기
