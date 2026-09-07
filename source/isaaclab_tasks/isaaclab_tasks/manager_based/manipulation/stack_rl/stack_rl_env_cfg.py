# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the stack scene with a robot and cubes.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target cubes, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """
    20251211 수정: Observation 간소화 (65-dim → 46-dim)

    변경사항:
        - 제거: joint_vel (7-dim) - IK-Rel에서 덜 중요
        - 제거: cube_dimensions (9-dim) - 상수이므로 불필요
        - 제거: cube_positions (9-dim, world frame 절대 좌표)
        - 추가: cube_1_position (3-dim) - Base cube 절대 위치
        - 추가: cube_relative_positions (6-dim) - Cube 간 상대 벡터

    이유:
        IsaacGym 분석 결과 상대 좌표가 절대 좌표보다 효과적:
        - IsaacGym (19-dim): cubeA_to_cubeB_pos 사용 → 성공
        - IsaacLab (33-dim): robot root frame 사용 → 성공
        - 우리 (65-dim): world frame 절대 좌표 → 실패

        상대 벡터의 장점:
        1. Task-specific: Policy가 직접 사용 가능
        2. Sample efficiency: 학습해야 할 관계가 명시적
        3. Dimension 감소: 65 → 46 (29% 감소)

    새 구성 (46-dim):
        joint_pos: 7
        cube_1_position: 3 (base cube)
        cube_relative_positions: 6 (cube_2→cube_1, cube_3→cube_2)
        cube_orientations: 12 (quaternions)
        eef_pos: 3
        eef_quat: 4
        gripper_pos: 2
        actions: 9 (last action, 유지)
        ---
        Total: 46-dim

    참고:
        - DIFFERENCE_LIFT_AND_GYM_STACK_AND_WHATTODO_IN_OUR.md
        - mdp/observations.py (새로 추가된 함수들)
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 20251211: Joint position 유지 (IK-Rel에서 필요)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)

        # 20251211 제거: joint_vel (IK-Rel에서 덜 중요, 7-dim 절약)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # 20251211 수정: Cube observations - 상대 좌표로 변경
        # 기존: cube_positions (9-dim, world frame) → 제거
        # 새로: cube_1_position (3-dim) + cube_relative_positions (6-dim)
        cube_1_position = ObsTerm(func=mdp.cube_1_position)  # Base cube (world frame)
        cube_relative_positions = ObsTerm(func=mdp.cube_relative_positions)  # 상대 벡터 (6-dim)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_compact)  # 12-dim (기존과 동일)

        # End-effector observations (유지)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)

        # Gripper state (유지)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        # Actions (유지 - 일부 환경에서 도움이 됨)
        actions = ObsTerm(func=mdp.last_action)

        # 20251211 제거: cube_dimensions (상수이므로 observation에 불필요)
        # reward 함수에서는 여전히 사용 가능
        # cube_dimensions = ObsTerm(func=mdp.cube_dimensions_in_world_frame)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Randomize cube positions on reset
    randomize_cube_positions = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.15, 0.15), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_1"),
        },
    )


@configclass
class RewardsCfg:
    # ==============================================================================
    # 20251212 (5차 수정): Anca et al. (2023) Style Reward Configuration
    # ==============================================================================
    """
    Reward terms for the MDP - Anca et al. Style with Curriculum Learning

    논문: "Achieving Goals using Reward Shaping and Curriculum Learning"
    arXiv:2206.02462

    배경:
        - 기존 (20251211): IsaacGym mutual exclusion + height penalty
        - 문제: Gripper opening 학습 안 됨, stage 구분 없음
        - 해결: Anca et al. 논문의 검증된 reward shaping + curriculum

    새로운 구조 (Anca et al. 철학):
        1. Dense Shaping: Box-to-goal + EE-to-box (λ=5.0)
        2. Sub-goal Bonus: One-time 큰 보상 (λ=150.0)
        3. Global Success: All cubes stacked (λ=150.0)
        4. Penalties: Action, Table, Orientation (λ=0.01~5.0)
        5. Curriculum: Stage-based gating (Epoch 0-1000-5000)

    Curriculum Timeline:
        - Epoch 0-1000: Stage 1 (cube_2 → cube_1만 active)
        - Epoch 1000-5000: Stage 2 (cube_2 완료 + cube_3 → cube_2)
        - Epoch 5000+: Stage 3 (all cubes active)

    최대 보상 (per step, internal before scaling):
        - Dense shaping: 5.0 (box_to_goal) + 5.0 (ee_to_box) = 10.0
        - Sub-goal bonus: 150.0 (one-time per sub-goal)
        - Global success: 150.0 (one-time, all cubes)
        - Penalties: -0.01 (action) - 5.0 (table) - 0.1 (ori) ≈ -5.11
        - Max practical: ~150 (sub-goal) or ~155 (global success moment)

    After scaling (×0.1):
        - Dense: ~1.0 per step
        - Sub-goal: 15.0 (huge spike!)
        - Global: 15.0 (huge spike!)

    참고:
        - Anca et al. Table 1: λ values
        - 새 함수: mdp/rewards.py line 1250-1745
        - GPT-5.1 제안 + Claude 분석 (toward_more_cube_stack.md)
    """

    # ==============================================================================
    # Anca Dense Shaping Rewards (Continuous Guidance)
    # ==============================================================================

    box_to_goal_distance = RewTerm(
        func=mdp.box_to_goal_distance_anca,
        params={
            "cube_1_goal_height": 0.0203,   # Table height
            "cube_2_goal_offset": 0.0406,   # 2× cube size
            "cube_3_goal_offset": 0.0812,   # 4× cube size (cumulative)
        },
        weight=5.0,  # λ = 5.0 (Anca Table 1)
        # 20251212 주석: r_dense in Anca
        #   - Cube가 goal에 가까워질수록 덜 negative
        #   - Curriculum gating: Stage에 따라 active cube만
        #   - Stage 1: cube_2만, Stage 2: cube_2+3, Stage 3: all
    )

    ee_to_box_distance = RewTerm(
        func=mdp.ee_to_box_distance_anca,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=5.0,  # λ = 5.0 (Anca Table 1)
        # 20251212 주석: r_guide in Anca
        #   - EE가 target cube에 가까워지도록 유도
        #   - Stage 1: EE → cube_2, Stage 2+: EE → cube_3
    )

    # ==============================================================================
    # Anca Sparse Rewards (One-time Bonuses)
    # ==============================================================================

    subgoal_bonus = RewTerm(
        func=mdp.subgoal_sparse_bonus_anca,
        params={
            "goal_eps": 0.03,      # 3cm threshold
            "cube_size": 0.0203,   # 2.03cm cube
        },
        weight=150.0,  # λ = 150.0 (Anca Table 1) - 매우 큼!
        # 20251212 주석: Sub-goal one-time bonus
        #   - Cube_2 on cube_1 달성 시: +150 (한 번만!)
        #   - Cube_3 on cube_2 달성 시: +150 (한 번만!)
        #   - env.extras에 tracking하여 중복 방지
        #   - 이게 "놓는 행위" 학습의 핵심!
    )

    global_success = RewTerm(
        func=mdp.global_success_bonus_anca,
        params={
            "goal_eps": 0.03,
            "cube_size": 0.0203,
        },
        weight=150.0,  # λ = 150.0 (Anca Table 1)
        # 20251212 주석: Global success bonus
        #   - 3개 cube 모두 쌓이면 +150 (한 번만!)
        #   - Episode 종료 trigger
    )

    # ==============================================================================
    # Anca Penalties
    # ==============================================================================

    action_penalty = RewTerm(
        func=mdp.action_penalty_anca,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
        },
        weight=0.01,  # λ = 0.01 (Anca Table 1)
        # 20251212 주석: r_action in Anca
        #   - Action magnitude + joint deviation from home
        #   - 부드러운 움직임 유도
    )

    table_collision_penalty = RewTerm(
        func=mdp.table_collision_penalty_anca,
        params={
            "table_height": 0.0,  # Ground level
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=5.0,  # λ = 5.0 (Anca Table 1)
        # 20251212 주석: r_table in Anca
        #   - EE가 테이블 아래로 가면 -1
        #   - 충돌 방지
    )

    orientation_penalty = RewTerm(
        func=mdp.ee_orientation_penalty_anca,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=0.1,  # λ = 0.1 (Anca Table 1)
        # 20251212 주석: r_orientation in Anca
        #   - 초기 orientation 유지 유도
        #   - Stable grasping에 도움
    )

    # ==============================================================================
    # 기존 Constraints (유지)
    # ==============================================================================

    cube_1_stays_on_table = RewTerm(
        func=mdp.object_stays_on_table,
        params={
            "object_cfg": SceneEntityCfg("cube_1"),
            "initial_height": 0.0203,
            "tolerance": 0.01,
        },
        weight=2.0,
        # 20251212 유지: Cube_1이 base로 유지되는지 체크
    )

    # ==============================================================================
    # Curriculum Success Tracking (weight=0 - tracking only, no reward)
    # ==============================================================================

    curriculum_success_tracker = RewTerm(
        func=mdp.track_curriculum_success,
        params={
            "xy_threshold": 0.05,
            "height_threshold": 0.005,
            "height_diff": 0.0468,
        },
        weight=0.0,
        # 20251212: Success-based curriculum tracking
        #   - Tracks cube2_stacked, cube3_stacked, tower_complete flags
        #   - Stores in env.extras for curriculum manager to use
        #   - Weight=0 means no reward, only tracking
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_1")}
    )

    cube_2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")}
    )

    cube_3_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_3")}
    )

    # Success termination
    success = DoneTerm(func=mdp.cubes_stacked)


@configclass
class CurriculumCfg:
    """
    Curriculum terms for the MDP.

    20251212 (6차 수정): Success-based curriculum (IsaacLab ADR style)

    배경:
        - 기존 (5차): Epoch-based curriculum (고정 epoch으로 stage 전환)
        - 문제:
            1. common_step_counter가 없어서 curriculum_epoch가 0으로 고정됨
            2. Stage 1도 학습이 안 되는데 무조건 epoch으로 넘어가는 문제
            3. 환경마다 학습 속도가 다른데 동일한 timeline 적용

    새로운 방식 (Success-based):
        - IsaacLab의 terrain_levels_vel과 동일한 철학
        - 각 environment마다 독립적으로 stage tracking
        - 성공률 70% 달성 시 자동으로 다음 stage로 전환
        - 성공률이 너무 낮으면 (< 21%) 이전 stage로 regression

    Stage 정의:
        - Stage 1: Cube2를 Cube1 위에 쌓기
        - Stage 2: Cube3를 Cube2 위에 쌓기 (Cube2는 이미 쌓임)
        - Stage 3: 전체 타워 완성 (모든 cube 쌓기)

    성공 조건:
        - track_curriculum_success()가 매 step마다 success flags 업데이트
        - stack_stages_success()가 episode 종료 시 success 기록 및 stage 전환

    참고:
        - IsaacLab terrain curriculum: locomotion/velocity/mdp/curriculums.py
        - 우리 구현: stack_rl/mdp/curriculums.py
        - Success tracking: stack_rl/mdp/rewards.py track_curriculum_success()
    """

    # Success-based curriculum manager
    # This will be called at the end of each episode to update curriculum stages
    stack_curriculum = CurrTerm(
        func=mdp.stack_stages_success,
        params={
            "success_threshold": 0.7,  # 70% success rate to advance
            "window_size": 100,  # Rolling window of 100 episodes
        }
    )


##
# Environment configuration
##


@configclass
class StackRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking RL environment."""

    # Scene settings
    # 20251211 수정: num_envs 4096 → 8192 (2배 증가)
    # 이유: 더 많은 parallel experience 수집으로 학습 속도 향상
    #       GPU 메모리가 충분하다면 더 많은 환경으로 sample efficiency 증가
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=8192, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        # 20251211 수정: 15.0 → 10.0
        # 이유: 3-cube stacking은 2-cube보다 2배 어렵지만, 15초는 너무 길어 학습 속도 저하
        #       IsaacGym/IsaacLab: 5초 (1-2 cube)
        #       우리: 10초 (3-cube, 2배로 설정하여 충분한 시간 확보하면서도 빠른 iteration)
        #       10초 = 500 steps @ 50Hz (decimation=2)
        self.episode_length_s = 10.0  # Longer than lift (5s), shorter than original stack (30s)
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        # 20251211 수정: 32*1024 → 64*1024
        # 이유: num_envs 4096→8192로 2배 증가 시 collision pairs도 증가
        #       PhysX 에러: totalAggregatePairsCapacity ~33913 필요
        #       안전하게 2배로 증가 (32768 → 65536)
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024  # For 8192 envs
        self.sim.physx.friction_correlation_distance = 0.00625
