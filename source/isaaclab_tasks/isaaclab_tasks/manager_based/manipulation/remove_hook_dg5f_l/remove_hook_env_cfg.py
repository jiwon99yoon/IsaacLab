# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Base configuration for Remove Hook task environments - DG5F_L Variant.

This base class provides the common structure for hook removal tasks with DG5F Left hand.
Key difference from RH56F1_R variant: Target is right_ring (왼손이 오른쪽 고리를 잡음)

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
    """Scene configuration for Remove Hook task (DG5F_L variant).

    Same as RH56F1_R variant - shared chassis and wire configuration.
    """

    # Robot configuration (to be specified by child class)
    robot: ArticulationCfg = MISSING

    # ============================================================================
    # CHASSIS ASSEMBLY CONFIGURATION (Static Structure)
    # ============================================================================
    chassis = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Chassis",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/dyros/IsaacLab/hyundai/1_factory_usd_file/flattended_usd/env_diated_decomposed_chassis_tilted_flattened_no_rigid_instanceable.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.001,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.75, -1.3, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ============================================================================
    # WIRE WITH HOOKS CONFIGURATION (Articulation)
    # ============================================================================
    # USD Structure (wire_revolute_collision_flattened.usd):
    #   wire_model (defaultPrim, Articulation Root)
    #   ├── wire (Xform, RigidBodyAPI) - 본체
    #   │   ├── left_1, left_2, left_3, left_4
    #   │   ├── right_1, right_2, right_3, right_4
    #   │   ├── bar, bar_case
    #   ├── right_ring (Xform, RigidBodyAPI) ← DG5F_L Target!
    #   │   └── right_rjoint (PhysicsRevoluteJoint) - wire와 연결
    #   │       - Joint Limits: -60° ~ +10°, default=0°
    #   └── left_ring (Xform, RigidBodyAPI)
    #       └── left_rjoint (PhysicsRevoluteJoint) - wire와 연결
    #           - Joint Limits: -10° ~ +60°, default=0°
    object = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Wire",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/dyros/IsaacLab/hyundai/1_factory_usd_file/flattended_usd/wire_revolute_collision_flattened.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
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
            pos=(-0.75, -1.3, 0.54755),
            rot=(math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)),
        ),
        actuators={},
    )

    # No table - chassis is fixed in air
    table = None

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0)),
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
# Command Configuration
# ============================================================================
# NOTE: Command 시스템 - dexsuite처럼 타겟 위치 기반 학습 가능
# DG5F_L은 right_ring을 타겟으로 함 (RH56F1_R은 left_ring)


@configclass
class RemoveHookCommandsCfg:
    """Command terms for Remove Hook task (DG5F_L variant).

    Generates random target positions for the hook (right_ring body).
    The robot should move the hook to this target position.

    NOTE: DG5F_L (left hand) targets right_ring (vs RH56F1_R targets left_ring)
    """

    hook_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        # CRITICAL: Track right_ring body, not wire root! (DG5F_L target)
        object_body_names=["right_ring"],
        resampling_time_range=(8.0, 8.0),  # Resample once per episode (8s)
        debug_vis=False,
        position_only=True,  # Position tracking only (orientation is complex for wire)
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            # Target position ranges (robot base frame)
            # Based on chassis position: (-0.75, -1.3, 0.1)  ← y is NEGATIVE for DG5F_L
            # Hook should move away from spring → outward/upward direction
            # NOTE: DG5F_L is on opposite side (y=-1.3), so y values are NEGATIVE
            pos_x=(-0.5, 0.0),    # Slightly toward robot (opposite of RH56F1_R)
            pos_y=(-1.6, -1.0),   # Around chassis y position (NEGATIVE for DG5F_L)
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
    """Observation specifications for Remove Hook task (DG5F_L variant).

    Key difference: Target is right_ring (not left_ring)
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations - high-level task info."""

        # Hook state (position and orientation in robot frame)
        # NOTE: DG5F_L targets right_ring (왼손이 오른쪽 고리를 잡음)
        hook_pos_b = ObsTerm(
            func=mdp.hook_position_b,
            params={"hook_cfg": SceneEntityCfg("object", body_names=["right_ring"])},
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-5.0, 5.0),
        )
        hook_quat_b = ObsTerm(
            func=mdp.hook_orientation_b,
            params={"hook_cfg": SceneEntityCfg("object", body_names=["right_ring"])},
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-2.0, 2.0),
        )

        # ========================================================================
        # TARGET POSE FROM COMMAND
        # ========================================================================
        # NOTE: Command 시스템에서 생성된 타겟 위치를 observation으로 전달
        # Policy가 "어디로 가야 하는지" 알 수 있음
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

        # Hook point cloud (right_ring만 샘플링 - DG5F_L 타겟에 집중)
        # NOTE: DG5F_L targets right_ring (RH56F1_R는 left_ring)
        object_point_cloud = ObsTerm(
            func=mdp.object_point_cloud_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-5.0, 5.0),
            params={
                "num_points": 64,
                "flatten": True,
                "object_cfg": SceneEntityCfg("object", body_names=["right_ring"]),
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

    # Object physics randomization
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

    # Robot joint reset
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": [0.0, 0.0],
            "velocity_range": [0.0, 0.0],
        },
    )


# ============================================================================
# Reward Configuration
# ============================================================================


@configclass
class RemoveHookRewardsCfg:
    """Base reward configuration for Remove Hook task (DG5F_L variant)."""

    # Action penalties
    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.005)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.01)

    # ==========================================================================
    # POSITION TRACKING REWARD
    # ==========================================================================
    # NOTE: Command 시스템의 타겟 위치로 hook을 이동시키는 reward
    # DG5F_L은 right_ring을 타겟으로 함
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh_body,  # body tracking 지원 버전
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.2,
            "command_name": "hook_pose",
            "align_asset_cfg": SceneEntityCfg("object", body_names=["right_ring"]),
        },
    )

    # Reaching reward: Fingertip이 hook에 가까워지도록 유도
    # NOTE: DG5F_L targets right_ring
    fingers_to_hook = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.4, "object_cfg": SceneEntityCfg("object", body_names=["right_ring"])},
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
    """Termination conditions for Remove Hook task (DG5F_L variant)."""

    # Timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Object out of bounds (hook falls too far)
    # NOTE: DG5F_L targets right_ring (not left_ring)
    # NOTE: y values are NEGATIVE because DG5F_L chassis is at y=-1.3
    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {
                "x": (-1.75, 0.25),
                "y": (-2.3, -0.3),  # NEGATIVE for DG5F_L (chassis at y=-1.3)
                "z": (-0.9, 1.1),
            },
            "asset_cfg": SceneEntityCfg("object", body_names=["right_ring"]),
        },
    )

    # Abnormal robot state
    abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)


# ============================================================================
# Base Environment Configuration
# ============================================================================


@configclass
class RemoveHookEnvCfg(ManagerBasedRLEnvCfg):
    """Base environment configuration for Remove Hook task (DG5F_L variant)."""

    # Viewer settings
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 2.0, 1.5), lookat=(0.0, 0.0, 0.5), origin_type="env")

    # Scene settings
    # MODIFIED (2026-01-12): num_envs 16 → 128, env_spacing 8.0 → 4.0
    # - num_envs=128: 더 많은 병렬 환경으로 학습 효율 향상
    # - env_spacing=4.0: 로봇+샤시 크기에 적합 (8.0은 과도함)
    scene: RemoveHookSceneCfg = RemoveHookSceneCfg(num_envs=128, env_spacing=4.0, replicate_physics=False)

    # MDP components
    observations: RemoveHookObservationsCfg = RemoveHookObservationsCfg()
    actions = MISSING
    rewards: RemoveHookRewardsCfg = RemoveHookRewardsCfg()
    terminations: RemoveHookTerminationsCfg = RemoveHookTerminationsCfg()
    events: RemoveHookEventCfg = RemoveHookEventCfg()

    # ==========================================================================
    # COMMAND SYSTEM
    # ==========================================================================
    # NOTE: dexsuite처럼 타겟 위치 기반 학습 가능
    # DG5F_L은 right_ring을 타겟으로 함
    commands: RemoveHookCommandsCfg = RemoveHookCommandsCfg()
    # ==========================================================================

    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 8.0
        self.is_finite_horizon = True
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15


@configclass
class RemoveHookEnvCfg_PLAY(RemoveHookEnvCfg):
    """Play/evaluation environment configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.scene.num_envs = 1
