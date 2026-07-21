# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ============================================================================
# USD STRUCTURE (Matching HDR20-17 DG5F)
# ============================================================================
# The USD file now includes tip xforms under RH56F1_L:
#   - palm (replaces base_link as reference point)
#   - thumb_tip, index_tip, middle_tip, ring_tip, little_tip
#
# This matches HDR20-17 structure which has tip xforms under dg5f_right_new:
#   - rl_dg_palm, rl_dg_1_tip, rl_dg_2_tip, etc.
#
# Configuration Structure:
# 1. CONTACT SENSORS: Placed on collision links (left_thumb_4, left_index_2, etc.)
#    - prim_path uses slash: "RH56F1_R/right_thumb_4"
#    - sensor name uses underscore: "RH56F1_L_left_thumb_4_object_s"
#
# 2. OBSERVATIONS/REWARDS: Use tip xforms for accurate fingertip positions
#    - body_names: ["palm", ".*_tip"] (regex pattern matches all tip xforms)
#    - This provides actual fingertip positions, not link origins
# ============================================================================

# Left / Right

#from isaaclab_assets.robots import UR10E_INSPIRE_LEFT_CFG
from isaaclab_assets.robots import UR10E_INSPIRE_RIGHT_CFG

import isaaclab.sim as sim_utils
from isaaclab.sim import CapsuleCfg, ConeCfg, CuboidCfg, RigidBodyMaterialCfg, SphereCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp


# ============================================================================
# INSPIRE HAND SCENE CONFIG (0.5x object size)
# ============================================================================
# Inspire Hand is significantly smaller than Allegro Hand
# Object sizes scaled to 0.5x (vs Kuka-Allegro 1.0x, vs ur10e_dg5f 1.0x)
# This allows for systematic comparison of hand actuation structures
# ============================================================================

@configclass
class InspireSceneCfg(dexsuite.SceneCfg):
    """Scene configuration for Inspire Hand with 0.5x scaled objects.

    Inspire Hand is approximately 50-70% the size of Allegro Hand.
    Objects are scaled to 0.5x to match hand size for fair comparison.

    Comparison:
    - Kuka-Allegro (baseline): 1.0x
    - UR10e-DG5F (this project): 1.0x (same as baseline)
    - UR10e-Inspire (this config): 0.5x (scaled for smaller hand)
    """

    # Override object configuration with 0.5x scaling
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                # Cuboids (Kuka-Allegro × 0.5)
                CuboidCfg(size=(0.025, 0.05, 0.05), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),      # was (0.05, 0.1, 0.1)
                CuboidCfg(size=(0.025, 0.025, 0.05), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),    # was (0.05, 0.05, 0.1)
                CuboidCfg(size=(0.0125, 0.05, 0.05), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),    # was (0.025, 0.1, 0.1)
                CuboidCfg(size=(0.0125, 0.025, 0.05), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),   # was (0.025, 0.05, 0.1)
                CuboidCfg(size=(0.0125, 0.0125, 0.05), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # was (0.025, 0.025, 0.1)
                CuboidCfg(size=(0.005, 0.05, 0.05), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),     # was (0.01, 0.1, 0.1)
                # Spheres (Kuka-Allegro × 0.5)
                SphereCfg(radius=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),                 # was 0.05
                SphereCfg(radius=0.0125, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),                # was 0.025
                # Capsules (Kuka-Allegro × 0.5)
                CapsuleCfg(radius=0.02, height=0.0125, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # was (0.04, 0.025)
                CapsuleCfg(radius=0.02, height=0.005, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),   # was (0.04, 0.01)
                CapsuleCfg(radius=0.02, height=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),    # was (0.04, 0.1)
                CapsuleCfg(radius=0.0125, height=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # was (0.025, 0.1)
                CapsuleCfg(radius=0.0125, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),   # was (0.025, 0.2)
                CapsuleCfg(radius=0.005, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),    # was (0.01, 0.2)
                # Cones (Kuka-Allegro × 0.5)
                ConeCfg(radius=0.025, height=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),      # was (0.05, 0.1)
                ConeCfg(radius=0.0125, height=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),     # was (0.025, 0.1)
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
    )


@configclass
class Ur10eRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)

@configclass
class Ur10eReorientRewardCfg(dexsuite.RewardsCfg):
    """Reorient task rewards (reverted to Kuka-Allegro baseline)

    HISTORY:
    - Original (UH035): weight=2.0 (4x higher than Kuka)
    - Issue: High contact reward + grasp_duration caused "table pinning" strategy
           Agent learned to press object on table instead of lifting
    - Solution (2025-11-18): Reverted to Kuka-Allegro baseline (weight=0.5)
           Removed grasp_duration, kept ground_contact_penalty for safety

    REWARD STRUCTURE (Kuka-Allegro baseline):
    - good_finger_contact: 0.5 (encourages contact but doesn't dominate)
    - Base rewards: fingers_to_object(1.0), position_tracking(2.0), success(10.0)
    - Contact alone insufficient → Agent must learn lifting + positioning
    """
    # NOTE: Inspire Hand sensor names defined inline to avoid RewardManager parsing issues
    # Cannot use class attribute as RewardManager expects only RewardTermCfg instances

    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.5,  # REVERTED to Kuka-Allegro baseline (was 2.0 in UH035)
        params={
            "threshold": 1.0,
            "contact_sensor_names": [
                "ur10e_RH56F1_R_right_thumb_4_object_s",
                "ur10e_RH56F1_R_right_index_2_object_s",
                "ur10e_RH56F1_R_right_middle_2_object_s",
                "ur10e_RH56F1_R_right_ring_2_object_s",
                "ur10e_RH56F1_R_right_little_2_object_s"
            ]
        },
    )

    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.2,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
            "contact_sensor_names": [
                "ur10e_RH56F1_R_right_thumb_4_object_s",
                "ur10e_RH56F1_R_right_index_2_object_s",
                "ur10e_RH56F1_R_right_middle_2_object_s",
                "ur10e_RH56F1_R_right_ring_2_object_s",
                "ur10e_RH56F1_R_right_little_2_object_s"
            ],
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
            "contact_sensor_names": [
                "ur10e_RH56F1_R_right_thumb_4_object_s",
                "ur10e_RH56F1_R_right_index_2_object_s",
                "ur10e_RH56F1_R_right_middle_2_object_s",
                "ur10e_RH56F1_R_right_ring_2_object_s",
                "ur10e_RH56F1_R_right_little_2_object_s"
            ],
        },
    )

@configclass
class Ur10eMixinCfg:
    rewards: Ur10eReorientRewardCfg = Ur10eReorientRewardCfg()
    actions: Ur10eRelJointPosActionCfg = Ur10eRelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()

        # === Inspire Hand Configuration ===
        # Inspire Hand Structure (from USD):
        # - Palm: RH56F1_L_base_link (base), plam_1, plam_2, plam_3, plam_force_sensor
        # - 5 Fingertips: thumb_tip, index_tip, middle_tip, ring_tip, little_tip
        # - 5 Force Sensors: thumb_force_sensor, index_force_sensor, etc.

        # Command body: Use palm xform (added to USD matching HDR20-17 structure)
        # HDR20-17 uses "rl_dg_palm", UR10e uses "palm"
        # NOTE: Isaac Lab flattens body names, so no ur10e/ prefix needed
        self.commands.object_pose.body_name = "palm"

        # Robot configuration
        self.scene.robot = UR10E_INSPIRE_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = UR10E_INSPIRE_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Contact sensor locations (placed on collision links, not tip xforms)
        # These are the last links of each finger where collision actually occurs
        # Matching HDR20-17 pattern: contact sensors on links (rl_dg_1_4, etc.)
        # NOTE: UR10e has "ur10e/" prefix due to extra Xform layer in USD structure
        finger_tip_body_list = [
            "ur10e/RH56F1_R/right_thumb_4",    # Thumb collision link (tip parent)
            "ur10e/RH56F1_R/right_index_2",    # Index collision link
            "ur10e/RH56F1_R/right_middle_2",   # Middle collision link
            "ur10e/RH56F1_R/right_ring_2",     # Ring collision link
            "ur10e/RH56F1_R/right_little_2"    # Little collision link
        ]

        # Create contact sensors for each collision link
        # Sensor names use underscore (e.g., "RH56F1_L_left_thumb_4_object_s")
        # Prim paths use slash (e.g., "{ENV_REGEX_NS}/Robot/RH56F1_R/right_thumb_4")
        for link_name in finger_tip_body_list:
            sensor_name = link_name.replace("/", "_") + "_object_s"
            setattr(
                self.scene,
                sensor_name,
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )

        # Observation: Contact force from collision links
        # Convert "/" to "_" in sensor names (Python attribute naming requirement)
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [link.replace("/", "_") + "_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),  # Contact force in finger tips is under 20N normally (Inspire: effort=1.0)
        )

        # Observation: Hand tips state (palm + all fingertips)
        # Uses tip xforms added to USD (matching HDR20-17 structure)
        # HDR20-17: ["rl_dg_palm", ".*_tip"]
        # UR10e: ["palm", ".*_tip"] - Isaac Lab flattens body names
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = [
            "palm",      # Palm xform (replaces base_link)
            ".*_tip"     # All fingertip xforms (index_tip, little_tip, middle_tip, ring_tip, thumb_tip)
        ]

        # Reward: Distance from fingers to object
        # Uses tip xforms added to USD (matching HDR20-17 structure)
        # HDR20-17: ["rl_dg_palm", ".*_tip"]
        # UR10e: ["palm", ".*_tip"] - Isaac Lab flattens body names
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            body_names=[
                "palm",      # Palm xform (replaces base_link)
                ".*_tip"     # All fingertip xforms (index_tip, little_tip, middle_tip, ring_tip, thumb_tip)
            ]
        )

        # [DISABLED] reset_robot_wrist_joint
        # 6-DOF UR10e에서는 7-DOF Kuka의 null-space 탐색이 불필요하므로
        # dexsuite_env_cfg.py에서 reset_robot_wrist_joint를 주석 처리함
        # -------------------------------------------------------------------------
        # if self.events.reset_robot_wrist_joint is not None:
        #     self.events.reset_robot_wrist_joint.params["asset_cfg"] = SceneEntityCfg(
        #         "robot", joint_names="elbow_joint"
        #     )

        # OVERRIDE: reset_hand_joints (Remove DG5F joints, keep only Inspire Hand)
        # Base config (dexsuite_env_cfg.py) includes both DG5F and Inspire joints
        # This ensures only Inspire Hand joints are randomized for this environment
        # Solution: Override to include only Inspire Hand joints
        if hasattr(self.events, "reset_hand_joints") and self.events.reset_hand_joints is not None:
            self.events.reset_hand_joints.params["asset_cfg"] = SceneEntityCfg(
                "robot",
                joint_names=[
                    # Inspire Left Hand (6 DOF) - bending joints only
                    # Thumb: left_thumb_2_joint (excludes _1 which is spread joint)
                    # Other fingers: left_*_1_joint (for Inspire, _1 is the bending joint)
                    r"right_thumb_2_joint",
                    r"right_(index|middle|ring|little)_1_joint",
                ]
            )


@configclass
class DexsuiteUr10eLiftEnvCfg(Ur10eMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    """UR10e + Inspire Hand Lift task with 0.5x scaled objects.

    Key differences from base config:
    - Object size: 0.5x (vs DG5F 1.0x) - matched to Inspire Hand size
    - Hand: 6-DOF underactuated Inspire Hand
    - Arm init pose: Same as DG5F (for controlled comparison)
    """
    # Override scene to use Inspire-specific object scaling
    scene: InspireSceneCfg = InspireSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)

@configclass
class DexsuiteUr10eLiftEnvCfg_PLAY(Ur10eMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    """UR10e + Inspire Hand Lift task (Play mode) with 0.5x scaled objects."""
    # Override scene to use Inspire-specific object scaling
    scene: InspireSceneCfg = InspireSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)