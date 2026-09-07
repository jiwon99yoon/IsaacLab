# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Main data generation script.

=== 전체 개요 ===
이 스크립트는 Imitation Learning을 위한 데이터셋 생성을 담당합니다.

INPUT:
  - Source demonstrations (HDF5): 사람이 기록한 원본 데모
  - Environment config: 환경 설정 (로봇, 객체, 초기 조건 등)

PROCESSING:
  - Source demo의 subtask pattern을 새로운 초기 조건에서 재현
  - CuRobo (--use_skillgen): Subtask 간 collision-free transition 생성
  - 병렬 실행 (--num_envs): 여러 환경에서 동시에 데모 생성

OUTPUT:
  - Generated demonstrations (HDF5): 성공/실패 데모 분리 저장
    * output_file.hdf5: 성공한 데모들
    * output_file_failed.hdf5: 실패한 데모들 (디버깅용)

실행 예시:
  ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \\
    --device cuda \\
    --num_envs 4 \\
    --generation_num_trials 20 \\
    --input_file ./datasets/annotated_dataset.hdf5 \\
    --output_file ./datasets/generated_dataset.hdf5 \\
    --task Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0 \\
    --use_skillgen
"""


# =============================================================================
# SECTION 1: Isaac Sim 시뮬레이터 초기화
# =============================================================================
# Isaac Sim은 NVIDIA Omniverse 기반 시뮬레이터로, 초기화가 다른 라이브러리보다 먼저 이루어져야 합니다.
# AppLauncher를 먼저 실행해야 omniverse 환경이 준비됩니다.

"""Launch Isaac Sim Simulator first."""

import argparse  # Command line argument 파싱

from isaaclab.app import AppLauncher  # Isaac Sim 런처


# -----------------------------------------------------------------------------
# 1.1 Command Line Arguments 정의
# -----------------------------------------------------------------------------
# 사용자가 스크립트 실행 시 전달할 수 있는 옵션들을 정의합니다.

parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab environments.")

# Task 이름 (환경 이름)
# 예: "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0"
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

# 생성할 데모 개수
# generation_guarantee=True인 경우: 성공 데모 N개를 생성할 때까지 시도
# generation_guarantee=False인 경우: N번 시도 후 종료
parser.add_argument("--generation_num_trials", type=int, help="Number of demos to be generated.", default=None)

# 병렬 실행할 환경 개수
# num_envs=4이면 4개 환경이 동시에 다른 데모를 생성
# GPU 메모리가 충분하면 높일수록 빠름
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to instantiate for generating datasets."
)

# Input: Source dataset (필수!)
# 사람이 기록한 원본 demonstrations
parser.add_argument("--input_file", type=str, default=None, required=True, help="File path to the source dataset file.")

# Output: 생성된 dataset 저장 경로
# 성공 데모: output_file.hdf5
# 실패 데모: output_file_failed.hdf5 (자동 생성)
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="File path to export recorded and generated episodes.",
)

# 디버깅 옵션: Subtask마다 일시정지
# --headless가 아닐 때만 유용 (렌더링하면서 관찰)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="pause after every subtask during generation for debugging - only useful with render flag",
)

# Pinocchio 라이브러리 활성화
# Pink IK controller나 GR1T2 retargeter 사용 시 필요
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

# ★★★ 핵심 옵션: CuRobo 사용 여부 ★★★
# True: Subtask 간 전환을 CuRobo motion planning으로 생성 (Skillgen 모드)
# False: 단순 interpolation으로 전환 (기본 모드)
parser.add_argument(
    "--use_skillgen",
    action="store_true",
    default=False,
    help="use skillgen to generate motion trajectories",
)

# AppLauncher arguments 추가
# --headless, --device, --enable_cameras 등
AppLauncher.add_app_launcher_args(parser)

# Command line arguments 파싱
args_cli = parser.parse_args()


# -----------------------------------------------------------------------------
# 1.2 Pinocchio Import (조건부)
# -----------------------------------------------------------------------------
# Pinocchio를 AppLauncher 전에 import해야 버전 충돌 방지
# Isaac Sim에도 pinocchio가 있지만, IsaacLab 버전을 우선 사용

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401


# -----------------------------------------------------------------------------
# 1.3 Isaac Sim 시뮬레이터 실행
# -----------------------------------------------------------------------------
# 이 시점에서 Omniverse가 초기화됩니다.
# 이후 모든 Isaac Lab 코드는 이 시뮬레이터 위에서 동작합니다.

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app  # Omniverse SimulationApp 객체


# =============================================================================
# SECTION 2: 나머지 라이브러리 Import
# =============================================================================
# Isaac Sim이 초기화된 후에야 IsaacLab 관련 라이브러리를 import할 수 있습니다.

"""Rest everything follows."""

import asyncio  # 비동기 병렬 실행 (여러 환경 동시 제어)
import gymnasium as gym  # 환경 인터페이스 (OpenAI Gym API)
import inspect  # API signature 검증용
import numpy as np  # 수치 연산
import random  # 난수 생성
import torch  # PyTorch (state, action은 torch.Tensor)

import omni  # Omniverse API

from isaaclab.envs import ManagerBasedRLMimicEnv  # Mimic 환경 기반 클래스

import isaaclab_mimic.envs  # noqa: F401  # Mimic 환경들을 gym에 등록
# 이 import로 "Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0" 같은 환경이 gym.make()에서 사용 가능

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401  # Pinocchio 환경들 등록

# 데이터 생성 관련 핵심 함수들
from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation, setup_env_config
# - env_loop: 환경 시뮬레이션 메인 루프 (env.step 실행)
# - setup_async_generation: 비동기 데이터 생성 설정
# - setup_env_config: 환경 config 설정 (recorder, termination 등)

from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths
# - get_env_name_from_dataset: Input HDF5에서 환경 이름 추출
# - setup_output_paths: 출력 경로 설정

import isaaclab_tasks  # noqa: F401  # IsaacLab task 환경들 등록


# =============================================================================
# SECTION 3: 메인 함수
# =============================================================================

def main():
    """
    데이터 생성 메인 함수

    전체 흐름:
    1. 환경 설정 (output 경로, env config)
    2. 환경 생성 (gym.make)
    3. CuRobo 초기화 (--use_skillgen인 경우)
    4. 비동기 데이터 생성 설정
    5. 메인 실행 루프
    6. 정리 및 종료
    """

    # -------------------------------------------------------------------------
    # 3.1 환경 개수 설정
    # -------------------------------------------------------------------------
    num_envs = args_cli.num_envs  # 병렬 실행할 환경 개수 (예: 4)


    # -------------------------------------------------------------------------
    # 3.2 출력 경로 설정 및 환경 이름 결정
    # -------------------------------------------------------------------------

    # Setup output paths and get env name
    # 예: "./datasets/output.hdf5" → output_dir="./datasets", output_file_name="output"
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)

    # Task 이름 추출
    task_name = args_cli.task
    if task_name:
        # "Isaac-Lab:Task-v0" 형식인 경우 ":"로 split하여 마지막 부분 사용
        task_name = args_cli.task.split(":")[-1]

    # Task 이름이 없으면 input dataset의 metadata에서 추출
    # HDF5 파일의 @env_args attribute에 환경 이름 저장되어 있음
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)
    # 최종 env_name 예: "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0"


    # -------------------------------------------------------------------------
    # 3.3 환경 설정 구성
    # -------------------------------------------------------------------------

    # Configure environment
    # setup_env_config() 함수는 다음을 수행합니다:
    # 1. 환경 config 파싱 (parse_env_cfg)
    # 2. Success termination term 추출
    # 3. Recorder 설정 (HDF5 저장 담당)
    #    - dataset_export_mode 설정 (성공/실패 분리 여부)
    # 4. Observation concatenation 비활성화
    env_cfg, success_term = setup_env_config(
        env_name=env_name,  # 환경 이름
        output_dir=output_dir,  # 출력 디렉토리
        output_file_name=output_file_name,  # 출력 파일 이름 (확장자 제외)
        num_envs=num_envs,  # 환경 개수
        device=args_cli.device,  # "cuda" 또는 "cpu"
        generation_num_trials=args_cli.generation_num_trials,  # 생성 시도 횟수 override
    )
    # 반환값:
    # - env_cfg: 환경 설정 (recorders 포함)
    # - success_term: Success 판정 함수 (env.cfg.terminations.success)


    # -------------------------------------------------------------------------
    # 3.4 환경 생성
    # -------------------------------------------------------------------------

    # Create environment
    # gym.make(): Gymnasium API로 환경 생성
    # .unwrapped: Wrapper를 제거하고 실제 환경 객체 접근
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    # 환경이 ManagerBasedRLMimicEnv을 상속하는지 확인
    # Mimic 환경은 다음 API를 구현해야 함:
    # - target_eef_pose_to_action(): Target pose → Action 변환
    # - get_initial_conditions(): 초기 조건 설정
    # - get_subtasks(): Subtask annotation
    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")


    # -------------------------------------------------------------------------
    # 3.5 API 버전 체크
    # -------------------------------------------------------------------------

    # Check if the mimic API from this environment contains decprecated signatures
    # target_eef_pose_to_action() API가 새로운 signature를 사용하는지 확인
    # 예전: target_eef_pose_to_action(..., noise=...)
    # 지금: target_eef_pose_to_action(..., action_noise_dict=...)
    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        omni.log.warn(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action", '
            "is deprecated. Please update the API to take action_noise_dict instead."
        )


    # -------------------------------------------------------------------------
    # 3.6 시드 설정 (Reproducibility)
    # -------------------------------------------------------------------------

    # Set seed for generation
    # 동일한 시드로 재현 가능한 데이터 생성
    # env.cfg.datagen_config.seed는 환경 config에서 설정됨 (기본값 보통 42)
    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)


    # -------------------------------------------------------------------------
    # 3.7 환경 초기화
    # -------------------------------------------------------------------------

    # Reset before starting
    # 환경을 초기 상태로 리셋
    # 이 시점에 scene이 생성되고 로봇/객체가 배치됨
    env.reset()


    # =========================================================================
    # ★★★ 3.8 CuRobo Motion Planner 초기화 (핵심!) ★★★
    # =========================================================================

    # CuRobo를 사용하지 않는 경우 None
    motion_planners = None

    if args_cli.use_skillgen:  # --use_skillgen 플래그가 True인 경우에만 실행
        # CuRobo 관련 클래스 import
        # 여기서 import하는 이유: use_skillgen=False일 때는 불필요하므로
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

        # =====================================================================
        # CuRobo Motion Planner 생성
        # =====================================================================
        # 각 환경마다 독립적인 motion planner 필요
        # 이유: 각 환경의 world state가 다르므로 (객체 위치, attached objects 등)

        # Create one motion planner per environment
        motion_planners = {}  # {env_id: CuroboPlanner 인스턴스}

        for env_id in range(num_envs):
            print(f"Initializing motion planner for environment {env_id}")

            # -----------------------------------------------------------------
            # CuRobo Config 생성
            # -----------------------------------------------------------------
            # Create a config instance from the task name
            # CuroboPlannerCfg.from_task_name()은 task에 맞는 설정을 자동 생성:
            # - Robot URDF 경로 (Nucleus에서 다운로드)
            # - Collision spheres 설정
            # - Planning parameters (max iterations, success threshold 등)
            planner_config = CuroboPlannerCfg.from_task_name(env_name)

            # -----------------------------------------------------------------
            # Visualization 설정
            # -----------------------------------------------------------------
            # Ensure visualization is only enabled for the first environment
            # If not, sphere and plan visualization will be too slow in isaac lab
            # It is efficient to visualize the spheres and plan for the first environment in rerun

            # Visualization은 성능 이슈로 첫 번째 환경만 활성화
            # - visualize_spheres: 로봇의 collision sphere 시각화
            # - visualize_plan: 계획된 trajectory 시각화 (Rerun viewer)
            if env_id != 0:
                planner_config.visualize_spheres = False
                planner_config.visualize_plan = False

            # -----------------------------------------------------------------
            # CuroboPlanner 인스턴스 생성
            # -----------------------------------------------------------------
            motion_planners[env_id] = CuroboPlanner(
                env=env,  # 환경 객체 (world obstacles 정보)
                robot=env.scene["robot"],  # 로봇 articulation (joint info)
                config=planner_config,  # CuRobo 설정
                env_id=env_id,  # 환경 ID
            )
            # CuroboPlanner 초기화 시:
            # 1. Nucleus에서 Franka URDF 다운로드 (~/.curobo/robot 에 캐시)
            # 2. Collision spheres 생성 (로봇을 sphere로 근사화)
            # 3. CuRobo MotionGen 객체 생성 (GPU에서 parallel planning)
            # 4. World model 초기화 (환경의 obstacles)

        # Skillgen 모드 활성화 플래그 설정
        # 이 플래그가 True이면 DataGenerator에서 CuRobo를 사용
        env.cfg.datagen_config.use_skillgen = True


    # -------------------------------------------------------------------------
    # 3.9 비동기 데이터 생성 설정
    # -------------------------------------------------------------------------

    # Setup and run async data generation
    # setup_async_generation() 함수는 다음을 수행:
    # 1. Input HDF5에서 source demos 로딩
    # 2. DataGenInfoPool 생성 (source demo 정보 공유)
    # 3. DataGenerator 인스턴스 생성 (각 환경마다)
    # 4. Async tasks 생성 (각 DataGenerator.generate() 코루틴)
    # 5. Reset/Action queue 생성 (환경 ↔ DataGenerator 통신)
    async_components = setup_async_generation(
        env=env,  # 환경 객체
        num_envs=args_cli.num_envs,  # 환경 개수
        input_file=args_cli.input_file,  # Source dataset HDF5
        success_term=success_term,  # Success 판정 함수
        pause_subtask=args_cli.pause_subtask,  # 디버깅용 일시정지 플래그
        motion_planners=motion_planners,  # ★ CuRobo planners 전달 ★
    )
    # 반환값 (dict):
    # - "tasks": [async task0, task1, task2, ...]  # 각 환경의 DataGenerator.generate()
    # - "reset_queue": asyncio.Queue()  # Reset 요청 큐
    # - "action_queue": asyncio.Queue()  # Action 전달 큐
    # - "info_pool": DataGenInfoPool  # Source demo 정보
    # - "event_loop": asyncio event loop  # 비동기 이벤트 루프


    # =========================================================================
    # 3.10 메인 실행 루프
    # =========================================================================
    #
    # 실행 구조:
    #
    # ┌─────────────────────────────────────────┐
    # │         Main Thread                     │
    # │  ┌───────────────────────────────────┐  │
    # │  │ env_loop()                        │  │
    # │  │   while True:                     │  │
    # │  │     actions = get_actions()       │  │  ← action_queue에서 수신
    # │  │     env.step(actions)             │  │  → 시뮬레이션 실행
    # │  │     check_termination()           │  │
    # │  └───────────────────────────────────┘  │
    # └─────────────────────────────────────────┘
    #              ↑ actions           ↓ obs
    # ┌─────────────────────────────────────────┐
    # │      Asyncio Thread Pool                │
    # │  ┌────────────┐  ┌────────────┐        │
    # │  │ DataGen 0  │  │ DataGen 1  │  ...   │
    # │  │            │  │            │        │
    # │  │ generate() │  │ generate() │        │  ← 비동기로 병렬 실행
    # │  │  - Select  │  │  - Select  │        │
    # │  │    subtask │  │    subtask │        │
    # │  │  - CuRobo  │  │  - CuRobo  │        │  ★ 여기서 CuRobo 사용!
    # │  │    plan    │  │    plan    │        │
    # │  │  - Send    │  │  - Send    │        │
    # │  │    action  │  │    action  │        │
    # │  └────────────┘  └────────────┘        │
    # └─────────────────────────────────────────┘

    try:
        # ---------------------------------------------------------------------
        # 모든 async tasks를 하나의 future로 묶기
        # ---------------------------------------------------------------------
        # asyncio.gather(): 여러 코루틴을 동시에 실행
        # asyncio.ensure_future(): Future 객체 생성
        data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        # data_gen_tasks = Future[
        #     [DataGen0.generate(), DataGen1.generate(), DataGen2.generate(), ...]
        # ]

        # ---------------------------------------------------------------------
        # 환경 메인 루프 실행 (블로킹 호출)
        # ---------------------------------------------------------------------
        # env_loop()은 블로킹 함수로, 종료 조건이 만족될 때까지 계속 실행됩니다:
        # - num_success >= generation_num_trials (generation_guarantee=True)
        # - num_attempts >= generation_num_trials (generation_guarantee=False)
        # - Keyboard interrupt (Ctrl+C)
        # - Simulation stopped
        env_loop(
            env,  # 환경 객체
            async_components["reset_queue"],  # Reset 요청 큐
            async_components["action_queue"],  # Action 전달 큐
            async_components["info_pool"],  # Source demo 정보 풀
            async_components["event_loop"],  # Asyncio event loop
        )
        # env_loop() 내부 동작:
        # 1. action_queue에서 모든 환경의 action 수집 (블로킹)
        # 2. env.step(actions) 실행
        # 3. Episode 완료 시 env.recorders.export_episodes() → HDF5 저장
        # 4. 성공/실패 통계 업데이트
        # 5. Termination 조건 체크
        # 6. 반복

    except asyncio.CancelledError:
        # Async task가 취소된 경우 (정상 종료)
        print("Tasks were cancelled.")

    finally:
        # =====================================================================
        # 3.11 정리 및 종료
        # =====================================================================

        # ---------------------------------------------------------------------
        # Async tasks 취소
        # ---------------------------------------------------------------------
        # env_loop()이 종료되면 async tasks도 취소해야 함
        # Cancel all async tasks when env_loop finishes
        data_gen_tasks.cancel()

        try:
            # Wait for tasks to be cancelled
            # 모든 async tasks가 취소될 때까지 대기
            async_components["event_loop"].run_until_complete(data_gen_tasks)
        except asyncio.CancelledError:
            print("Remaining async tasks cancelled and cleaned up.")
        except Exception as e:
            print(f"Error cancelling remaining async tasks: {e}")

        # ---------------------------------------------------------------------
        # CuRobo Motion Planner 정리
        # ---------------------------------------------------------------------
        # Cleanup of motion planners and their visualizers
        if motion_planners is not None:
            for env_id, planner in motion_planners.items():
                # plan_visualizer 닫기 (Rerun viewer)
                if getattr(planner, "plan_visualizer", None) is not None:
                    print(f"Closing plan visualizer for environment {env_id}")
                    planner.plan_visualizer.close()
                    planner.plan_visualizer = None
            # Motion planner 딕셔너리 비우기 (메모리 해제)
            motion_planners.clear()


# =============================================================================
# SECTION 4: 프로그램 엔트리 포인트
# =============================================================================

if __name__ == "__main__":
    try:
        # 메인 함수 실행
        main()
    except KeyboardInterrupt:
        # Ctrl+C로 중단한 경우
        print("\nProgram interrupted by user. Exiting...")
    # Isaac Sim 시뮬레이터 종료
    # 이 호출이 없으면 프로세스가 종료되지 않음
    simulation_app.close()


# =============================================================================
# CuRobo 사용 흐름 요약
# =============================================================================
"""
CuRobo가 실제로 사용되는 위치:

1. generate_dataset.py (여기):
   - Line 127-152: CuRobo 초기화
     * motion_planners[env_id] = CuroboPlanner(...)
   - Line 161: DataGenerator에 전달
     * setup_async_generation(..., motion_planners=motion_planners)

2. setup_async_generation() (generation.py:195-270):
   - DataGenerator 생성 시 motion_planner 전달
     * DataGenerator(..., motion_planner=motion_planners[env_id])

3. DataGenerator.generate() (data_generator.py:650-780):
   - Subtask 전환 시 CuRobo 사용:
     ```python
     if self.env_cfg.datagen_config.use_skillgen and motion_planner:
         # Target pose 계산
         target_pose = ...

         # ★★★ CuRobo Motion Planning ★★★
         planning_success = motion_planner.update_world_and_plan_motion(
             target_pose=target_pose,
             expected_attached_object=attached_object,
             env_id=env_id,
             step_size=0.01,  # 1cm steps
             enable_retiming=True,
         )

         if planning_success:
             # 계획된 경로를 waypoints로 변환
             waypoints = convert_to_waypoints(motion_planner.get_planned_poses())

             # Waypoints 실행
             for waypoint in waypoints:
                 action = self.env.target_eef_pose_to_action(waypoint)
                 await self._apply_action(action)
     ```

CuRobo의 역할:
- Subtask A 끝 → Subtask B 시작 지점으로 collision-free 경로 생성
- 실제 manipulation (grasp, place)은 source demo에서 가져옴
- 즉, "경로 계획자"일 뿐 "skill generator"가 아님!
"""
