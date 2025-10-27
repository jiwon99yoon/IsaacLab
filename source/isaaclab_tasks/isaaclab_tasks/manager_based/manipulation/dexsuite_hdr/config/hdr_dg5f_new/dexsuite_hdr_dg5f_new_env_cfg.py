# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots import HDR_DG5F_CFG_NEW

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp


@configclass
class HdrDg5fNewRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)

@configclass
class HdrDg5fNewReorientRewardCfg(dexsuite.RewardsCfg):
    """Reorient 태스크용 리워드 (손가락 접촉 추가)"""
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.5,
        params={"threshold": 1.0},
    )



@configclass
class HdrDg5fNewMixinCfg:
    rewards: HdrDg5fNewReorientRewardCfg = HdrDg5fNewReorientRewardCfg()
    actions: HdrDg5fNewRelJointPosActionCfg = HdrDg5fNewRelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "rl_dg_palm" #"rl_dg_mount"
        self.scene.robot = HDR_DG5F_CFG_NEW.replace(prim_path="{ENV_REGEX_NS}/Robot")
        finger_tip_body_list = ["rl_dg_1_4", "rl_dg_2_4", "rl_dg_3_4", "rl_dg_4_4", "rl_dg_5_4"]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/dg5f_right_new/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["rl_dg_palm", ".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["rl_dg_palm", ".*_tip"])


@configclass
class DexsuiteHdrDg5fNewLiftEnvCfg(HdrDg5fNewMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass

@configclass
class DexsuiteHdrDg5fNewLiftEnvCfg_PLAY(HdrDg5fNewMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass