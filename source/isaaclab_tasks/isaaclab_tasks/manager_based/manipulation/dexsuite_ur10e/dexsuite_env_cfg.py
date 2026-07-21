# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import CapsuleCfg, ConeCfg, CuboidCfg, RigidBodyMaterialCfg, SphereCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .adr_curriculum import CurriculumCfg


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Dexsuite Scene for multi-objects Lifting"""

    # robot
    robot: ArticulationCfg = MISSING

    # object
    # MODIFIED: Object sizes scaled to 0.7x for Inspire Hand (smaller than Allegro Hand)
    # Inspire Hand is approximately 65-70% the size of Allegro Hand
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                # Cuboids (original × 0.7)
                CuboidCfg(size=(0.035, 0.07, 0.07), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),      # was (0.05, 0.1, 0.1)
                CuboidCfg(size=(0.035, 0.035, 0.07), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),    # was (0.05, 0.05, 0.1)
                CuboidCfg(size=(0.0175, 0.07, 0.07), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),    # was (0.025, 0.1, 0.1)
                CuboidCfg(size=(0.0175, 0.035, 0.07), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),   # was (0.025, 0.05, 0.1)
                CuboidCfg(size=(0.0175, 0.0175, 0.07), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # was (0.025, 0.025, 0.1)
                CuboidCfg(size=(0.007, 0.07, 0.07), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),     # was (0.01, 0.1, 0.1)
                # Spheres (original × 0.7)
                SphereCfg(radius=0.035, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),                 # was 0.05
                SphereCfg(radius=0.0175, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),                # was 0.025
                # Capsules (original × 0.7)
                CapsuleCfg(radius=0.028, height=0.0175, physics_material=RigidBodyMaterialCfg(static_friction=0.5)), # was (0.04, 0.025)
                CapsuleCfg(radius=0.028, height=0.007, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # was (0.04, 0.01)
                CapsuleCfg(radius=0.028, height=0.07, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),   # was (0.04, 0.1)
                CapsuleCfg(radius=0.0175, height=0.07, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # was (0.025, 0.1)
                CapsuleCfg(radius=0.0175, height=0.14, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # was (0.025, 0.2)
                CapsuleCfg(radius=0.007, height=0.14, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),   # was (0.01, 0.2)
                # Cones (original × 0.7)
                ConeCfg(radius=0.035, height=0.07, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),      # was (0.05, 0.1)
                ConeCfg(radius=0.0175, height=0.07, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),     # was (0.025, 0.1)
            ],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.0, 0.27)),
        #init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35)),
    )

    # table
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.0, 0.04),
            #size=(0.8, 1.5, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # trick: we let visualizer's color to show the table with success coloring
            visible=False,
        ),
        #init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.0, 0.235), rot=(1.0, 0.0, 0.0, 0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.0, 0.235), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(3.0, 5.0),
        debug_vis=False,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            # 수정: 테이블 주변 (로봇 앞쪽) 범위
            # 수정: 새로운 테이블 위치(x=0.8) 주변
            pos_x=(0.7, 0.9),     # 테이블 x=0.8 주변
            pos_y=(-0.3, 0.3),     # 테이블 y=0.0 주변
            pos_z=(0.3, 0.6),     # 테이블 위 공간 (유지)
            roll=(-3.14, 3.14),
            pitch=(-3.14, 3.14),
            yaw=(0.0, 0.0),
            # pos_x=(-0.7, -0.3),
            # pos_y=(-0.25, 0.25),
            # pos_z=(0.55, 0.95),
            # roll=(-3.14, 3.14),
            # pitch=(-3.14, 3.14),
            # yaw=(0.0, 0.0),
        ),
        success_vis_asset_name="table",
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        target_object_pose_b = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class ProprioObsCfg(ObsGroup):
        """Observations for proprioception group."""

        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        hand_tips_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            # good behaving number for position in m, velocity in m/s, rad/s,
            # and quaternion are unlikely to exceed -2 to 2 range
            clip=(-2.0, 2.0),
            params={
                "body_asset_cfg": SceneEntityCfg("robot"),
                "base_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        contact: ObsTerm = MISSING

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class PerceptionObsCfg(ObsGroup):

        object_point_cloud = ObsTerm(
            func=mdp.object_point_cloud_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-2.0, 2.0),  # clamp between -2 m to 2 m
            params={"num_points": 64, "flatten": True},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = True
            self.flatten_history_dim = True
            self.history_length = 5

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()


@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- pre-startup
    randomize_object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={"scale_range": (0.75, 1.5), "asset_cfg": SceneEntityCfg("object")},
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
#            "static_friction_range": [0.5, 1.0],
#            "dynamic_friction_range": [0.5, 1.0],
            "static_friction_range": [0.9, 1.1],
            "dynamic_friction_range": [0.9, 1.1],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

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

    joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": [0.8, 1.2], #[0.9, 1.1], - hdr
            "damping_distribution_params": [0.8, 1.2], # [0.9, 1.1], - hdr
            "operation": "scale",
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": [0.0, 5.0],
            "operation": "scale",
        },
    )

    # joint_stiffness_and_damping = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": [0.5, 2.0],
    #         "damping_distribution_params": [0.5, 2.0],
    #         "operation": "scale",
    #     },
    # )

    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": [0.2, 2.0],
            "operation": "scale",
        },
    )

    reset_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [0.0, 0.0]},
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("table"),
        },
    )

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                # 수정: 초기 물체 위치(0.6, 0.0, 0.27) 기준 상대 오프셋
                "x": [-0.1, 0.1],     # x=0.5 ~ 0.7 사이
                "y": [-0.2, 0.2],     # y=-0.3 ~ 0.3 사이
                "z": [0.0, 0.2],     # z=0.27 ~ 0.37 사이 (테이블 위)
                "roll": [-3.14, 3.14],
                "pitch": [-3.14, 3.14],
                "yaw": [-3.14, 3.14],
            },
            # "pose_range": {
            #     "x": [-0.2, 0.2],
            #     "y": [-0.2, 0.2],
            #     "z": [0.0, 0.4],
            #     "roll": [-3.14, 3.14],
            #     "pitch": [-3.14, 3.14],
            #     "yaw": [-3.14, 3.14],
            # },
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "yaw": [-0.0, 0.0]},
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # =========================================================================
    # ARM JOINTS: 팔 조인트 랜덤화 [-0.5, 0.5] rad
    # - UR10e 6-DOF arm joints: shoulder_pan, shoulder_lift, elbow, wrist_1/2/3
    # - 탐색 범위 유지하여 다양한 arm configuration 학습
    # =========================================================================
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
            ]),
            "position_range": [-0.50, 0.50],
            "velocity_range": [0.0, 0.0],
        },
    )

    # =========================================================================
    # HAND JOINTS: 손가락 굽힘 조인트만 랜덤화 [-0.15, 0.15] rad
    #
    # [변경 이력]
    # 1. 기존: 모든 조인트 [-0.5, 0.5] → 손가락 충돌
    # 2. 1차 수정: 모든 손가락 [-0.15, 0.15] → 여전히 벌림 조인트에서 충돌
    # 3. 최종: 굽힘 조인트만 랜덤화, 벌림 조인트 제외 → 충돌 방지!
    #
    # [제외한 조인트]
    # - DG5F 벌림: rj_dg_*_1 (손가락 간격 고정 → 충돌 방지)
    # - Inspire 엄지 벌림: left_thumb_1_joint (엄지-검지 간격 고정)
    #
    # [이유]
    # - 벌림 조인트 랜덤화 시 손가락끼리 너무 가까워짐 (특히 엄지-검지)
    # - Episode step 1에서 velocity 폭발 (초기 spawn 충돌)
    # - 예: rj_dg_1_1 (엄지 벌림) 0.6 → 0.45 시 검지와 충돌
    #
    # [참고]
    # - Allegro (4손가락)는 [-0.5, 0.5] 벌림 랜덤화 OK (간격 넓음)
    # - DG5F (5손가락)는 벌림 고정 필요 (간격 좁음)
    # =========================================================================
    reset_hand_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                # ===== DG5F Right Hand (20 DOF) =====
                # 굽힘만: _2, _3, _4 (벌림 _1 제외!)
                # - Finger 1~5: rj_dg_[1-5]_[234]
                r"rj_dg_[1-5]_[234]",

                # ===== Inspire Left Hand (6 DOF) =====
                # - Thumb 굽힘: left_thumb_2_joint (벌림 _1 제외!)
                # - 나머지 손가락 굽힘: left_*_1_joint (Inspire는 _1이 굽힘임)
                r"left_thumb_2_joint",
                r"left_(index|middle|ring|little)_1_joint",
            ]),
            # MODIFIED (11290125): [-0.15, 0.15] → [-0.05, 0.05] (±9° → ±3°)
            # 이유: deg2rad() 정확도 향상 후 초기 joint pos가 limit 근처로 이동
            #       domain randomization 범위 ±0.15가 limit 초과 → velocity 폭발 (10,452 rad/s)
            #       ±0.05로 축소하여 physics 안정성 확보 (grasp configuration 보존)
            "position_range": [-0.05, 0.05],
            "velocity_range": [0.0, 0.0],
        },
    )

    # =========================================================================
    # [DISABLED] reset_robot_wrist_joint
    # =========================================================================
    # 원래 Kuka (7-DOF) 환경에서는 iiwa7_joint_7을 [-3, 3]으로 크게 랜덤화.
    # 이유: 7-DOF 로봇은 null-space가 있어서, 마지막 조인트를 바꿔도
    #       다른 조인트가 보상하여 end-effector pose 유지 가능 → 다양한 arm config 탐색
    #
    # UR10e는 6-DOF 로봇이므로 null-space 없음 → 이 랜덤화 불필요.
    # 이미 reset_robot_joints에서 모든 arm 조인트가 [-0.5, 0.5]로 랜덤화됨.
    # -------------------------------------------------------------------------
    # reset_robot_wrist_joint = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names="elbow_joint"),
    #         "position_range": [-0.1, 0.1],
    #         "velocity_range": [0.0, 0.0],
    #     },
    # )

    # Note (Octi): This is a deliberate trick in Remake to accelerate learning.
    # By scheduling gravity as a curriculum — starting with no gravity (easy)
    # and gradually introducing full gravity (hard) — the agent learns more smoothly.
    # This removes the need for a special "Lift" reward (often required to push the
    # agent to counter gravity), which has bonus effect of simplifying reward composition overall.
    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            "operation": "abs",
        },
    )


@configclass
class ActionsCfg:
    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.005)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)

    fingers_to_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.4}, weight=1.0)

    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.2,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    orientation_tracking = RewTerm(
        func=mdp.orientation_command_error_tanh,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 1.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    success = RewTerm(
        func=mdp.success_reward,
        weight=10,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_std": 0.1,
            "rot_std": 0.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    early_termination = RewTerm(func=mdp.is_terminated_term, weight=-1, params={"term_keys": "abnormal_robot"})

    # ========== ADDITIONAL REWARDS (Not in Kuka-Allegro baseline) ==========
    # HISTORY (2025-11-18):
    # - Added for UH035/UR10e to encourage grasping with smaller Inspire Hand
    # - Issue discovered: High contact rewards caused "table pinning" strategy
    # - Solution: Reverted to Kuka-Allegro baseline (removed grasp_duration)
    # - Kept ground_contact_penalty as safety measure

    # DISABLED: object_lift (UH035 experiment)
    # Reason: Agent found shortcut - pushing with arm instead of grasping with fingers
    # This caused good_finger_contact to remain at 0
    # ORIGINAL (REMOVED):
    # object_lift = RewTerm(
    #     func=mdp.object_lift_height,
    #     weight=5.0,
    #     params={"threshold": 0.05, "object_cfg": SceneEntityCfg("object")},
    # )

    # DISABLED: grasp_duration (2025-11-18)
    # Reason: Combined with high good_finger_contact (2.0), caused "table pinning"
    #         Agent learned to press object on table to maintain contact
    #         Total contact reward (2.0 + 1.0 = 3.0) >> ground penalty (-0.05)
    # Implementation: mdp/rewards.py::grasp_duration() still exists for future use
    # ORIGINAL (REMOVED):
    # grasp_duration = RewTerm(
    #     func=mdp.grasp_duration,
    #     weight=1.0,
    #     params={"threshold": 1.0, "min_duration": 10},
    # )

    # KEPT: Penalty for object touching ground/table
    # Prevents extreme bouncing strategy while allowing initial contact attempts
    # Weight intentionally weak (-0.05) to not discourage contact learning
    # HISTORY: -3.0 (too strong) → -1.5 (still strong) → -0.05 (current)
    ground_contact_penalty = RewTerm(
        func=mdp.object_ground_contact_penalty,
        weight=-0.3,  # -0.05 -> -0.3 change # Weak penalty - prevents extreme bouncing but allows contact
        params={"object_cfg": SceneEntityCfg("object")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            # 수정: 로봇 앞쪽 작업 공간을 포함하도록 범위 변경
            "in_bound_range": {"x": (-0.5, 2.0), "y": (-2.0, 2.0), "z": (0.0, 2.0)},
            "asset_cfg": SceneEntityCfg("object"),
            # "in_bound_range": {"x": (-1.5, 0.5), "y": (-2.0, 2.0), "z": (0.0, 2.0)},
            # "asset_cfg": SceneEntityCfg("object"),
        },
    )

    abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)


@configclass
class DexsuiteReorientEnvCfg(ManagerBasedEnvCfg):
    """Dexsuite reorientation task definition, also the base definition for derivative Lift task and evaluation task"""

    # Scene settings
    viewer: ViewerCfg = ViewerCfg(eye=(2.25, 0.0, 0.75), lookat=(0.0, 0.0, 0.45), origin_type="env")
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg | None = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2  # 50 Hz

        # *single-goal setup
        self.commands.object_pose.resampling_time_range = (10.0, 10.0)
        self.commands.object_pose.position_only = False
        self.commands.object_pose.success_visualizer_cfg.markers["failure"] = self.scene.table.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.15, 0.15), roughness=0.25), visible=True
        )
        self.commands.object_pose.success_visualizer_cfg.markers["success"] = self.scene.table.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.25, 0.15), roughness=0.25), visible=True
        )

        self.episode_length_s = 4.0
        self.is_finite_horizon = True

        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15

        if self.curriculum is not None:
            self.curriculum.adr.params["pos_tol"] = self.rewards.success.params["pos_std"] / 2
            self.curriculum.adr.params["rot_tol"] = self.rewards.success.params["rot_std"] / 2


class DexsuiteLiftEnvCfg(DexsuiteReorientEnvCfg):
    """Dexsuite lift task definition"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.orientation_tracking = None  # no orientation reward
        self.commands.object_pose.position_only = True
        if self.curriculum is not None:
            self.rewards.success.params["rot_std"] = None  # make success reward not consider orientation
            self.curriculum.adr.params["rot_tol"] = None  # make adr not tracking orientation


class DexsuiteReorientEnvCfg_PLAY(DexsuiteReorientEnvCfg):
    """Dexsuite reorientation task evaluation environment definition"""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.resampling_time_range = (2.0, 3.0)
        self.commands.object_pose.debug_vis = True
        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params["max_difficulty"]

        # Disable domain randomization for Play mode
        # Use reset_scene_to_default instead of random joint offsets
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_scene_to_default,
            mode="reset",
        )
        # Disable other robot reset events (they add random offsets)
        self.events.reset_robot_wrist_joint = None
        if hasattr(self.events, "reset_thumb_2_joint"):
            self.events.reset_thumb_2_joint = None


class DexsuiteLiftEnvCfg_PLAY(DexsuiteLiftEnvCfg):
    """Dexsuite lift task evaluation environment definition"""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.resampling_time_range = (2.0, 3.0)
        self.commands.object_pose.debug_vis = True
        self.commands.object_pose.position_only = True
        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params["max_difficulty"]

        # Disable domain randomization for Play mode
        # Use reset_scene_to_default instead of random joint offsets
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_scene_to_default,
            mode="reset",
        )
        # Disable other robot reset events (they add random offsets)
        self.events.reset_robot_wrist_joint = None
        if hasattr(self.events, "reset_thumb_2_joint"):
            self.events.reset_thumb_2_joint = None
