# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Remove Hook Task Environment Configuration - HDR35_20 + RH56F1_R

Task Description:
- Robot: Single arm (HDR35_20 + RH56F1_R inspire hand right)
- Object: Front chassis assembly with springs and wire hooks
- Goal: Grasp the right hook and remove it from spring, then move to target position
- Target: Left 0.2m, Up 0.3m from initial hook position

USD Structure (from env_diated_decomposed_chassis_spring_flattened.usd):
    env_decomposed_chassis_spring/
    ├── wire-model/ (Chassis + Springs - Static/Kinematic)
    │   ├── right_struct_spring (오른쪽 spring)
    │   ├── left_struct_spring (왼쪽 spring)
    │   └── SV_GT_PROTO_DPA_FR_* (chassis frames)
    ├── wire-revolute-collision-flattened/wire_model/wire/ (Wire - Dynamic)
    │   ├── right_1, right_2, right_3
    │   ├── left_1, left_2, left_3
    │   └── bar
    └── right-ring/ & left_ring/ (Hooks - Dynamic with RevoluteJoint)
        ├── right_ring_cover (+ RevoluteJoint) ← Target!
        └── left_ring_cover (+ RevoluteJoint)

Environment Spacing:
- Uniform spacing: 8.0m (wide spacing for chassis assembly and future dual-arm extension)
"""

from isaaclab_assets.robots import HDR35_20_RH56F1_R_CFG

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

# Import from our new base class (not dexsuite!)
from ... import remove_hook_env_cfg as remove_hook
from ... import mdp


# ============================================================================
# Action Configuration
# ============================================================================


@configclass
class Hdr35RelJointPosActionCfg:
    """Relative joint position action for HDR35_20 + RH56F1_R (12 DoF)

    MODIFIED (2026-01-12): 조인트별 다른 action scale 적용
    =========================================================

    문제 분석:
    - iiwa + Allegro에서는 마지막 암 조인트(j6/j7)가 회전하는 문제가 없음
    - HDR35 + DG5F/RH56F1에서는 j5/j6이 심하게 회전함

    핵심 원인:
    1. 극단적인 Stiffness 불균형:
       - iiwa j7/Allegro = 25/3 = 8.3:1
       - HDR35 j6/RH56F1 = 1,500/2.5 = 600:1 (72배 더 불균형!)

    2. 핸드에 전달되는 토크 부족:
       - Allegro: 3.0 × 0.1 = 0.3 Nm
       - RH56F1: 2.5 × 0.001 = 0.0025 Nm (1/120)

    3. iiwa j7의 "버퍼" 역할:
       - iiwa j7은 stiffness 25로 낮아서 핸드의 관성을 흡수
       - HDR35 j6은 stiffness 1,500으로 60배 강해서 버퍼 역할 못함

    해결책: 조인트별 다른 scale 적용
    - j1-5: 0.01 (기본 암 조인트)
    - j6: 0.02 (wrist roll - 핸드와의 연결, 더 부드럽게)
    - hand: 0.1 (핸드에 충분한 토크 전달 - Allegro 수준)

    토크 계산:
    - j1-3: 30,000 × 0.01 = 300 Nm
    - j4-5: 3,000 × 0.01 = 30 Nm
    - j6: 1,500 × 0.02 = 30 Nm
    - hand: 2.5 × 0.1 = 0.25 Nm (Allegro의 0.3 Nm과 유사)
    """

    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            r"j[1-5]": 0.01,                                    # 암 j1-j5: 기본 scale
            r"j6": 0.02,                                        # 암 j6 (wrist roll): 버퍼 역할
            r"right_thumb_(1|2)_joint": 0.1,                    # RH56F1 엄지
            r"right_(index|middle|ring|little)_1_joint": 0.1,  # RH56F1 손가락
        },
    )


# ============================================================================
# Reward Configuration
# ============================================================================


@configclass
class RemoveHookHdr35RewardCfg(remove_hook.RemoveHookRewardsCfg):
    """
    Reward configuration for Remove Hook task with HDR35_20 + RH56F1_R.

    Reward Structure:
    1. Reaching phase: Encourage hand to approach hook
    2. Grasping phase: Encourage stable grasp on hook
    3. Removal phase: Encourage separating hook from spring
    4. Transport phase: Encourage moving hook to target position
    """

    # ========== Phase 1: Reaching Hook ==========
    # NOTE: RH56F1_R (오른손)은 left_ring을 타겟으로 함
    #
    # std 값 변경 이력:
    # - 기존 0.1: tanh(d/0.1)이 금방 1에 수렴 → 멀리서 gradient 없음
    # - 1.0으로 변경: 초기 거리 ~1.5m 가정 → 너무 관대함 (1m에서도 0.24 보상)
    # - 0.5로 조정 (2026-01-12): 초기 거리 ~0.5m 이내 세팅에 적합
    #   - dexsuite(iiwa_allegro)는 std=0.4 사용
    #   - 0.5m에서 reward ≈ 0.24, 1.0m에서 reward ≈ 0.04
    fingers_to_hook = RewTerm(
        func=mdp.object_ee_distance,
        weight=2.0,
        params={
            "std": 0.5,  # dexsuite uses 0.4, adjusted for ~0.5m initial distance
            "object_cfg": SceneEntityCfg("object", body_names=["left_ring"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_base_link", ".*_tip"]),
        },
    )

    # ========== Phase 2: Grasping Hook ==========
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=3.0,
        params={"threshold": 0.5},  # Lower threshold for thin wire
    )

    # ========== Phase 3: Transport to Target ==========
    # NOTE: Removed hook_spring_separation - not sim-to-real transferable
    # FIXED (2026-01-12): std 0.02 → 0.2 (target_offset 거리 ~0.36m에서 gradient 확보)
    # 문제: std=0.02이면 tanh(0.36/0.02)=tanh(18)≈1 → reward≈0 (학습 신호 없음)
    # 해결: std=0.2이면 tanh(0.36/0.2)≈0.94 → reward≈0.06 (탐색 가능)
    hook_to_target = RewTerm(
        func=mdp.hook_to_target_distance,
        weight=10.0,
        params={
            "std": 0.2,  # Was 0.02 - too small for target distance ~0.36m
            "target_offset": (-0.2, 0.0, 0.3),  # Left 0.2m, up 0.3m from initial
            "hook_cfg": SceneEntityCfg("object", body_names=["left_ring"]),
        },
    )

    # ========== Penalties ==========
    # NOTE: Removed spring_collision_penalty - requires chassis body tracking


# ============================================================================
# Main Environment Configuration
# ============================================================================


@configclass
class RemoveHookHdr35Env_Cfg(remove_hook.RemoveHookEnvCfg):
    """Remove Hook environment configuration for HDR35_20 + RH56F1_R (Training)"""

    # Override actions and rewards
    actions: Hdr35RelJointPosActionCfg = Hdr35RelJointPosActionCfg()
    rewards: RemoveHookHdr35RewardCfg = RemoveHookHdr35RewardCfg()

    def __post_init__(self):
        # Call parent post_init first
        super().__post_init__()
        self.scene.robot = HDR35_20_RH56F1_R_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # # ============================================================================
        # # ROBOT CONFIGURATION
        # # ============================================================================
        # # Initial joint positions optimized for wire grasping
        # # Values from: (313fullest)fixed_v3(1222revised).usd - hdr035_ati_rh56f1_r_flattened
        # # joint1=109°, joint2=74°, joint3=-46°, joint4=-72°, joint5=76°, joint6=54°
        # import math
        # self.scene.robot = HDR35_20_RH56F1_R_CFG.replace(
        #     prim_path="{ENV_REGEX_NS}/Robot",
        #     init_state=HDR35_20_RH56F1_R_CFG.init_state.replace(
        #         joint_pos={
        #             "j1": math.radians(109),   # 1.902 rad
        #             "j2": math.radians(74),    # 1.292 rad
        #             "j3": math.radians(-46),   # -0.803 rad
        #             "j4": math.radians(-72),   # -1.257 rad
        #             "j5": math.radians(76),    # 1.326 rad
        #             "j6": math.radians(54),    # 0.942 rad
        #             # Finger joints remain at default (closed/open position will be controlled by action)
        #         },
        #     ),
        # )

        # ============================================================================
        # CONTACT SENSORS (Fingertips to Hook/Wire)
        # ============================================================================
        finger_tip_body_list = [
            "right_thumb_4",    # Thumb tip
            "right_index_2",    # Index tip
            "right_middle_2",   # Middle tip
            "right_ring_2",     # Ring tip
            "right_little_2",   # Little tip
        ]

        # Create contact sensors for hook/wire grasping
        # NOTE: RH56F1_R (오른손)은 left_ring을 타겟으로 함
        # 원래: filter_prim_paths_expr=["{ENV_REGEX_NS}/Wire"]  # Wire 전체와의 접촉 감지
        # 변경: filter_prim_paths_expr=["{ENV_REGEX_NS}/Wire/left_ring*"]  # left_ring만 감지
        for link_name in finger_tip_body_list:
            sensor_name = link_name + "_hook_sensor"
            setattr(
                self.scene,
                sensor_name,
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Wire/left_ring*"],  # left_ring만 감지
                ),
            )

        # NOTE: Removed spring_collision_sensor - not needed for simplified reward structure

        # ============================================================================
        # OBSERVATIONS
        # ============================================================================
        # Contact forces from fingertips
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [link + "_hook_sensor" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),
        )

        # Hand tips state (palm + fingertips)
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = [
            "gripper_base_link",  # Palm/gripper base
            ".*_tip",             # All fingertip xforms
        ]


@configclass
class RemoveHookHdr35Env_Cfg_PLAY(remove_hook.RemoveHookEnvCfg_PLAY):
    """Remove Hook environment configuration for HDR35_20 + RH56F1_R (Inference/Play)"""

    # Override actions and rewards (same as training)
    actions: Hdr35RelJointPosActionCfg = Hdr35RelJointPosActionCfg()
    rewards: RemoveHookHdr35RewardCfg = RemoveHookHdr35RewardCfg()

    def __post_init__(self):
        # Call parent post_init
        super().__post_init__()
        self.scene.robot = HDR35_20_RH56F1_R_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Robot configuration (same as training)
        # import math
        # self.scene.robot = HDR35_20_RH56F1_R_CFG.replace(
        #     prim_path="{ENV_REGEX_NS}/Robot",
        #     init_state=HDR35_20_RH56F1_R_CFG.init_state.replace(
        #         joint_pos={
        #             "j1": math.radians(109),   # 1.902 rad
        #             "j2": math.radians(74),    # 1.292 rad
        #             "j3": math.radians(-46),   # -0.803 rad
        #             "j4": math.radians(-72),   # -1.257 rad
        #             "j5": math.radians(76),    # 1.326 rad
        #             "j6": math.radians(54),    # 0.942 rad
        #         },
        #     ),
        # )

        # Contact sensors (fingertips to hook/wire)
        # NOTE: RH56F1_R (오른손)은 left_ring을 타겟으로 함
        # 원래: filter_prim_paths_expr=["{ENV_REGEX_NS}/Wire"]  # Wire 전체와의 접촉 감지
        # 변경: filter_prim_paths_expr=["{ENV_REGEX_NS}/Wire/left_ring*"]  # left_ring만 감지
        finger_tip_body_list = ["right_thumb_4", "right_index_2", "right_middle_2", "right_ring_2", "right_little_2"]
        for link_name in finger_tip_body_list:
            sensor_name = link_name + "_hook_sensor"
            setattr(
                self.scene,
                sensor_name,
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Wire/left_ring*"],  # left_ring만 감지
                ),
            )

        # NOTE: Removed spring_collision_sensor - not needed for simplified reward structure

        # Observations
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [link + "_hook_sensor" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),
        )

        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = [
            "gripper_base_link",
            ".*_tip",
        ]
