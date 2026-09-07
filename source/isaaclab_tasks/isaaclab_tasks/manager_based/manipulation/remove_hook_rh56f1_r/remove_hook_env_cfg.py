# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Base configuration for Remove Hook task environments.

This base class provides the common structure for hook removal tasks across different
robot configurations (HDR35_20 + RH56F1_R, HDR35_20 + DG5F_L, etc.).

Task: Remove wire hook from spring on chassis assembly and move to target position.
"""

from dataclasses import MISSING
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp


# ============================================================================
# Scene Configuration
# ============================================================================


@configclass
class RemoveHookSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Remove Hook task.

    Unlike dexsuite (lift/reorient with cubes), this scene is designed for:
    - Chassis assembly with springs and wire hooks (not simple cubes)
    - No table/ground (chassis is fixed in air, robot reaches to it)
    - Hook-specific contact sensors
    - Wide environment spacing for large assembly
    """

    # Robot configuration (to be specified by child class)
    robot: ArticulationCfg = MISSING

    # ============================================================================
    # CHASSIS ASSEMBLY CONFIGURATION (Static Structure)
    # ============================================================================
    # Chassis with springs - fixed structure (static collision geometry)
    # Position: (-0.75, 1.3, 0.1) - Base position in front of robot
    # Using AssetBaseCfg as pure collision geometry (no body tracking needed)
    chassis = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Chassis",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/dyros/IsaacLab/hyundai/1_factory_usd_file/flattended_usd/env_diated_decomposed_chassis_tilted_flattened_no_rigid_instanceable.usd",
            # NO rigid_props → pure collision geometry
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.001,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.75, 1.3, 0.1),  # Base position in front of robot
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ============================================================================
    # WIRE WITH HOOKS CONFIGURATION (Articulation)
    # ============================================================================
    # Wire with RevoluteJoints - the object we want to manipulate
    #
    # USD Structure (wire_revolute_collision_flattened.usd):
    #   wire_model (defaultPrim, Articulation Root)
    #   ├── wire (Xform, RigidBodyAPI) - 본체
    #   ├── right_ring (Xform, RigidBodyAPI)
    #   │   └── right_rjoint (PhysicsRevoluteJoint) - wire와 연결
    #   │       - Joint Limits: -60° ~ +10°
    #   └── left_ring (Xform, RigidBodyAPI)
    #       └── left_rjoint (PhysicsRevoluteJoint) - wire와 연결
    #           - Joint Limits: -10° ~ +60°
    #
    # Position Calculation (robot frame):
    #   - Chassis z: 0.1
    #   - Target wire_model position: 0.585 (robot frame)
    #   - Required spawn position: 0.585 - 0.03745 = 0.54755 (verified from Isaac Sim)
    #   - Wire spawn z: 0.54755 (this value)
    #   - USD internal structure applies offset to reach final position
    #   - Result: Wire correctly positioned to hook onto springs at chassis
    #   - Wire (chassis relative): 0.585 - 0.1 = 0.485 (48.5cm above chassis base)
    #
    # Wire articulation with separate root path specification
    # prim_path: Where USD loads
    # articulation_root_prim_path: Where to find articulation within USD (RELATIVE PATH)
    object = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Wire",  # USD load location
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/dyros/IsaacLab/hyundai/1_factory_usd_file/flattended_usd/wire_revolute_collision_flattened.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,  # TEMP: Disabled to prevent wire falling/explosion
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.001,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                enabled_self_collisions=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.75, 1.3, 0.54755),  # Spawn position: 0.585 - 0.03745 (verified from Isaac Sim)
            rot=(math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)),  # 90° z-rotation
        ),
        actuators={},
        # NOTE: articulation_root_prim_path removed - defaultPrim (wire_model) auto-detected
    )

    # No table - chassis is fixed in air
    table = None

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0)),  # Lower ground to avoid collision
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # Lighting
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ============================================================================
# Command Configuration (ADDED 2026-01-08)
# ============================================================================
# NOTE: Command 시스템 추가 - dexsuite처럼 타겟 위치 기반 학습 가능
# 기존 방식 (hook_pos_b만 사용)으로 되돌리려면:
#   1. 이 CommandsCfg 클래스 전체 주석처리
#   2. RemoveHookEnvCfg에서 commands 라인 주석처리
#   3. RemoveHookObservationsCfg.PolicyCfg에서 target_hook_pose_b 주석처리
#   4. RemoveHookRewardsCfg에서 position_tracking 주석처리


@configclass
class RemoveHookCommandsCfg:
    """Command terms for Remove Hook task.

    Generates random target positions for the hook (left_ring body).
    The robot should move the hook to this target position.

    Unlike dexsuite which tracks object root, we track specific body (left_ring)
    using the modified ObjectUniformPoseCommand with object_body_names parameter.
    """

    hook_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        # CRITICAL: Track left_ring body, not wire root!
        object_body_names=["left_ring"],
        resampling_time_range=(8.0, 8.0),  # Resample once per episode (8s)
        debug_vis=False,
        position_only=True,  # Position tracking only (orientation is complex for wire)
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            # Target position ranges (robot base frame)
            # Based on chassis position: (-0.75, 1.3, 0.1)
            # Hook should move away from spring → outward/upward direction
            pos_x=(-1.0, -0.5),   # Slightly further from robot (away from chassis)
            pos_y=(1.0, 1.6),     # Around chassis y position
            pos_z=(0.6, 1.0),     # Above chassis (hook lift target)
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        # No success_vis_asset (no table in this environment)
        success_vis_asset_name=None,
    )


# ============================================================================
# Observation Configuration
# ============================================================================


@configclass
class RemoveHookObservationsCfg:
    """Observation specifications for Remove Hook task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations - high-level task info."""

        # Hook state (position and orientation in robot frame)
        # TEMP: Strong clipping to prevent network crash from wire explosion
        hook_pos_b = ObsTerm(
            func=mdp.hook_position_b,
            params={"hook_cfg": SceneEntityCfg("object", body_names=["left_ring"])},
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-5.0, 5.0),
        )
        hook_quat_b = ObsTerm(
            func=mdp.hook_orientation_b,
            params={"hook_cfg": SceneEntityCfg("object", body_names=["left_ring"])},
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-2.0, 2.0),
        )

        # ========================================================================
        # TARGET POSE FROM COMMAND (ADDED 2026-01-08)
        # ========================================================================
        # NOTE: Command 시스템에서 생성된 타겟 위치를 observation으로 전달
        # Policy가 "어디로 가야 하는지" 알 수 있음
        # 기존 방식으로 되돌리려면 이 ObsTerm을 주석처리
        target_hook_pose_b = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "hook_pose"},
        )
        # ========================================================================

        # Previous actions (for temporal consistency)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class ProprioObsCfg(ObsGroup):
        """Proprioceptive observations - robot state."""

        # Joint state
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0))

        # Hand/fingertip state (to be configured by child class with specific body_names)
        hand_tips_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-2.0, 2.0),
            params={
                "body_asset_cfg": SceneEntityCfg("robot"),
                "base_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Contact forces (to be configured by child class with specific sensor names)
        contact: ObsTerm = MISSING

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class PerceptionObsCfg(ObsGroup):
        """Perception observations - object point cloud."""

        # Hook point cloud (left_ring만 샘플링 - 타겟에 집중)
        # NOTE (2026-01-07): Wire 전체 → left_ring만으로 변경
        #   - 기존: params={"num_points": 64, ...} → Wire 전체 mesh 샘플링
        #   - 수정: object_cfg에 body_names=["left_ring"] 지정 → left_ring mesh만 샘플링
        #   - 효과: (1) 타겟에 집중된 observation, (2) body_pos_w 사용으로 위치 정확도 향상
        object_point_cloud = ObsTerm(
            func=mdp.object_point_cloud_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-5.0, 5.0),  # Increased from (-2.0, 2.0) to prevent NaN
            params={
                "num_points": 64,
                "flatten": True,
                "object_cfg": SceneEntityCfg("object", body_names=["left_ring"]),
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = True
            self.flatten_history_dim = True
            self.history_length = 5

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()


# ============================================================================
# Event Configuration (Randomization)
# ============================================================================


@configclass
class RemoveHookEventCfg:
    """Randomization configuration for Remove Hook task."""

    # Robot physics randomization
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": [0.9, 1.1],
            "dynamic_friction_range": [0.9, 1.1],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    # Object (chassis assembly) physics randomization
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": [0.5, 1.0],
            "dynamic_friction_range": [0.5, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    # Joint stiffness and damping randomization
    joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": [0.9, 1.1],
            "damping_distribution_params": [0.9, 1.1],
            "operation": "scale",
        },
    )

    # Joint friction randomization
    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": [0.0, 5.0],
            "operation": "scale",
        },
    )

    # ==========================================================================
    # STOP HERE (01071616) - Wire reset 관련 이슈로 주석처리
    # 문제: reset 시 고리가 spring과 충돌하여 계속 움직임 → 학습 불가
    # 나중에 해결되면 주석 해제하여 사용
    # ==========================================================================
    #
    # # Wire (object) root state reset - 정확한 초기 위치로만 reset (randomization 없음)
    # # NOTE: Wire는 spring에 걸려있어야 하므로 정확한 위치가 필수
    # # NOTE: reset_object_joints는 제거 - RevoluteJoint angle을 0으로 reset하면 ring 위치가 틀어짐
    # # 따라서 root state만 reset하고 joint state는 그대로 유지
    # #
    # # IMPORTANT: z offset 0.05075 추가 이유:
    # #   - Spawn 시: wire_model (Articulation Root, no RigidBodyAPI) 기준으로 배치
    # #   - Reset 시: wire (첫 번째 RigidBody = PhysX base link) 기준으로 배치
    # #   - wire는 wire_model 기준 z=+0.05075 offset이 있음
    # #   - 따라서 Reset 시 z에 0.05075를 더해야 Spawn 시와 같은 위치가 됨
    # reset_object = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": [0.0, 0.0],
    #             "y": [0.0, 0.0],
    #             "z": [0.05075, 0.05075],  # wire vs wire_model offset 보정
    #             "roll": [0.0, 0.0],
    #             "pitch": [0.0, 0.0],
    #             "yaw": [0.0, 0.0],
    #         },
    #         "velocity_range": {
    #             "x": [0.0, 0.0],
    #             "y": [0.0, 0.0],
    #             "z": [0.0, 0.0],
    #             "roll": [0.0, 0.0],
    #             "pitch": [0.0, 0.0],
    #             "yaw": [0.0, 0.0],
    #         },
    #         "asset_cfg": SceneEntityCfg("object"),
    #     },
    # )
    #
    # # Wire (object) joint state reset - RevoluteJoint angle을 0으로 초기화
    # # Joint 이름 및 limit (USD에서 설정):
    # #   - left_rjoint (left_ring ↔ wire): -10° ~ +60°, default=0°
    # #   - right_rjoint (right_ring ↔ wire): -60° ~ +10°, default=0°
    # # position_range=[0.0, 0.0] → default joint position (0)으로 reset
    # reset_object_joints = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": [0.0, 0.0],  # default (0)으로 reset
    #         "velocity_range": [0.0, 0.0],  # velocity도 0으로
    #         "asset_cfg": SceneEntityCfg("object"),
    #     },
    # )
    #
    # ==========================================================================

    # Robot base position reset
    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0]},
            "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Robot joint reset to default positions (NO randomization)
    # NOTE: position_range=[0.0, 0.0] → no random offset → exact init_state from robot cfg
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": [0.0, 0.0],  # No randomization - use exact init positions from robot cfg
            "velocity_range": [0.0, 0.0],
        },
    )


# ============================================================================
# Reward Configuration
# ============================================================================


@configclass
class RemoveHookRewardsCfg:
    """Base reward configuration for Remove Hook task.

    Child classes should customize weights and add specific rewards.
    """

    # Action penalties (encourage smooth motion)
    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.005)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.01)

    # ==========================================================================
    # POSITION TRACKING REWARD (ADDED 2026-01-08)
    # ==========================================================================
    # NOTE: Command 시스템의 타겟 위치로 hook을 이동시키는 reward
    # dexsuite의 position_tracking과 유사하지만:
    #   1. body_names 지원 (left_ring body 추적)
    #   2. contact gating 없음 (wire는 grasp 대상이 아니므로)
    # 기존 방식으로 되돌리려면 이 reward를 주석처리
    #
    # IMPORTANT: mdp.position_command_error_tanh_body 사용 (새로 추가된 함수)
    #   - body_names 지원
    #   - contact gating 없음
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh_body,  # body tracking 지원 버전
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.2,
            "command_name": "hook_pose",
            "align_asset_cfg": SceneEntityCfg("object", body_names=["left_ring"]),
        },
    )

    # Reaching reward: Fingertip이 hook에 가까워지도록 유도
    # NOTE: 이미 기존에 mdp/rewards.py에 object_ee_distance가 정의되어 있음
    fingers_to_hook = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.4, "object_cfg": SceneEntityCfg("object", body_names=["left_ring"])},
        weight=1.0,
    )
    # ==========================================================================

    # Early termination penalty
    early_termination = RewTerm(func=mdp.is_terminated_term, weight=-1, params={"term_keys": "abnormal_robot"})


# ============================================================================
# Termination Configuration
# ============================================================================


@configclass
class RemoveHookTerminationsCfg:
    """Termination conditions for Remove Hook task."""

    # Timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Object out of bounds (hook falls too far)
    # Range is relative to chassis position: (-0.75, 1.3, 0.1) ± 1.0m in each axis
    # NOTE (Bug fix 2026-01-07): body_names=["left_ring"]을 지정하여 hook body position 체크
    # 기존: wire root만 체크 → hook 폭발해도 감지 못함
    # 수정: left_ring body position 체크 → hook 폭발 시 termination & reset
    # NOTE (2026-01-07): 범위 축소 2.0m → 1.0m (더 빠른 termination으로 학습 안정화)
    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {
                "x": (-1.75, 0.25),   # chassis_x (-0.75) ± 1.0
                "y": (0.3, 2.3),      # chassis_y (1.3) ± 1.0
                "z": (-0.9, 1.1),     # chassis_z (0.1) ± 1.0
            },
            "asset_cfg": SceneEntityCfg("object", body_names=["left_ring"]),
        },
    )

    # Abnormal robot state (velocity explosion)
    abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)


# ============================================================================
# Base Environment Configuration
# ============================================================================


@configclass
class RemoveHookEnvCfg(ManagerBasedRLEnvCfg):
    """Base environment configuration for Remove Hook task.

    This provides common settings for all remove hook variants.
    Child classes should:
    1. Specify robot configuration
    2. Specify chassis assembly configuration
    3. Add contact sensors
    4. Customize rewards
    5. Add specific observations
    """

    # Viewer settings
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 2.0, 1.5), lookat=(0.0, 0.0, 0.5), origin_type="env")

    # Scene settings
    # MODIFIED (2026-01-12): num_envs 16 → 128, env_spacing 8.0 → 4.0
    # - num_envs=128: 더 많은 병렬 환경으로 학습 효율 향상
    # - env_spacing=4.0: 로봇+샤시 크기에 적합 (8.0은 과도함)
    scene: RemoveHookSceneCfg = RemoveHookSceneCfg(num_envs=128, env_spacing=4.0, replicate_physics=False)

    # MDP components
    observations: RemoveHookObservationsCfg = RemoveHookObservationsCfg()
    actions = MISSING  # To be specified by child class
    rewards: RemoveHookRewardsCfg = RemoveHookRewardsCfg()
    terminations: RemoveHookTerminationsCfg = RemoveHookTerminationsCfg()
    events: RemoveHookEventCfg = RemoveHookEventCfg()

    # ==========================================================================
    # COMMAND SYSTEM (ADDED 2026-01-08)
    # ==========================================================================
    # NOTE: dexsuite처럼 타겟 위치 기반 학습 가능
    # 기존 방식으로 되돌리려면 이 라인을 주석처리하고 commands = None으로 변경
    commands: RemoveHookCommandsCfg = RemoveHookCommandsCfg()
    # 원래 설정 (command 없음):
    # commands = None
    # ==========================================================================

    # No curriculum by default (can be added by child class if needed)
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # Control settings
        self.decimation = 2  # 50 Hz control

        # Episode settings
        self.episode_length_s = 8.0  # 8 seconds per episode (hook removal may take longer than lift)
        self.is_finite_horizon = True

        # Simulation settings
        self.sim.dt = 1 / 120  # 120 Hz simulation
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15


@configclass
class RemoveHookEnvCfg_PLAY(RemoveHookEnvCfg):
    """Base play/evaluation environment configuration for Remove Hook task."""

    def __post_init__(self):
        super().__post_init__()
        # Longer episodes for observation
        self.episode_length_s = 20.0
        # Fewer environments for visualization
        self.scene.num_envs = 1
