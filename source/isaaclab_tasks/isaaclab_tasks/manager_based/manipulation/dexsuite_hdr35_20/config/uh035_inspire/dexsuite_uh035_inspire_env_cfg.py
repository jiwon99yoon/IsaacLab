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
#    - prim_path uses slash: "RH56F1_L/left_thumb_4"
#    - sensor name uses underscore: "RH56F1_L_left_thumb_4_object_s"
#
# 2. OBSERVATIONS/REWARDS: Use tip xforms for accurate fingertip positions
#    - body_names: ["palm", ".*_tip"] (regex pattern matches all tip xforms)
#    - This provides actual fingertip positions, not link origins
# ============================================================================

# Left / Right

from isaaclab_assets.robots import HDR35_20_RH56F1_R_CFG

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp


@configclass
class Uh035RelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)

@configclass
class Uh035ReorientRewardCfg(dexsuite.RewardsCfg):
    """Reorient 태스크용 리워드 (손가락 접촉 추가)"""
    # MODIFIED: Increased weight to encourage finger contact learning
    # ORIGINAL (Kuka-Allegro): weight=0.5
    # MODIFIED (HDR-DG5F): weight=3.0 (6x increase to prioritize grasping)
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=2.0,  # 3.0 -> 2.0으로줄임 # Increased from 0.5 to prioritize multi-finger contact
        params={"threshold": 1.0},
    )

@configclass
class Uh035MixinCfg:
    rewards: Uh035ReorientRewardCfg = Uh035ReorientRewardCfg()
    actions: Uh035RelJointPosActionCfg = Uh035RelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()

        # === Inspire Hand Configuration ===
        # Inspire Hand Structure (from USD):
        # - Palm: RH56F1_L_base_link (base), plam_1, plam_2, plam_3, plam_force_sensor
        # - 5 Fingertips: thumb_tip, index_tip, middle_tip, ring_tip, little_tip
        # - 5 Force Sensors: thumb_force_sensor, index_force_sensor, etc.

        # Command body: Use gripper base link as palm reference
        # HDR20-17 uses "rl_dg_palm", HDR35_20_RH56F1_R uses "gripper_base_link"
        # gripper_base_link is the parent of all finger links (right_thumb_1, right_index_1, etc.)
        self.commands.object_pose.body_name = "gripper_base_link"

        # Robot configuration
        self.scene.robot = HDR35_20_RH56F1_R_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = UH035_INSPIRE_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Contact sensor locations (placed on collision links, not tip xforms)
        # These are the last links of each finger where collision actually occurs
        # HDR35_20_RH56F1_R: Flat USD structure (no RH56F1_L scope), right hand
        finger_tip_body_list = [
            "right_thumb_4",    # Thumb collision link (tip parent)
            "right_index_2",    # Index collision link
            "right_middle_2",   # Middle collision link
            "right_ring_2",     # Ring collision link
            "right_little_2"    # Little collision link
        ]

        # Create contact sensors for each collision link
        # Sensor names use underscore (e.g., "right_thumb_4_object_s")
        # Prim paths: "{ENV_REGEX_NS}/Robot/right_thumb_4" (flat structure, no scope)
        for link_name in finger_tip_body_list:
            sensor_name = link_name + "_object_s"
            setattr(
                self.scene,
                sensor_name,
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )

        # Observation: Contact force from collision links
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [link + "_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),  # Contact force in finger tips is under 20N normally (Inspire: effort=1.0)
        )

        # Observation: Hand tips state (palm + all fingertips)
        # Uses tip xforms added to USD (matching HDR20-17 structure)
        # HDR20-17: ["rl_dg_palm", ".*_tip"]
        # HDR35_20_RH56F1_R: ["gripper_base_link", ".*_tip"]
        # gripper_base_link = parent of all fingers, .*_tip matches thumb_tip, index_tip, middle_tip, ring_tip, little_tip
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = [
            "gripper_base_link",  # Gripper base (palm reference)
            ".*_tip"              # All fingertip xforms (thumb_tip, index_tip, middle_tip, ring_tip, little_tip)
        ]

        # Reward: Distance from fingers to object
        # Uses tip xforms added to USD (matching HDR20-17 structure)
        # HDR20-17: ["rl_dg_palm", ".*_tip"]
        # HDR35_20_RH56F1_R: ["gripper_base_link", ".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            body_names=[
                "gripper_base_link",  # Gripper base (palm reference)
                ".*_tip"              # All fingertip xforms (thumb_tip, index_tip, middle_tip, ring_tip, little_tip)
            ]
        )


@configclass
class DexsuiteUh035LiftEnvCfg(Uh035MixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass

@configclass
class DexsuiteUh035LiftEnvCfg_PLAY(Uh035MixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass