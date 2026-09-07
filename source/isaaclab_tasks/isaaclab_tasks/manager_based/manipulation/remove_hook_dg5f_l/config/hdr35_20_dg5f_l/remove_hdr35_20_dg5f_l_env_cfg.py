# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Remove Hook Task Environment Configuration - HDR35_20 + DG5F_L

Task Description:
- Robot: Single arm (HDR35_20 + DG5F_L Tesollo hand left, 20-DOF)
- Object: Front chassis assembly with springs and wire hooks
- Goal: Grasp the right hook and remove it from spring, then move to target position
- Target: right_ring (왼손이므로 오른쪽 고리를 타겟)

Key Differences from RH56F1_R variant:
- Hand DoF: 20 (fully-actuated) vs 6 (under-actuated)
- Contact sensor bodies: ll_dg_X_4 vs right_xxx_X
- Palm body: ll_dg_palm vs gripper_base_link
- Target hook: right_ring vs left_ring

DG5F Left Hand Structure:
- Joint naming: lj_dg_[finger]_[joint]
  - finger: 1=thumb, 2=index, 3=middle, 4=ring, 5=little
  - joint: 1-4 (4 joints per finger)
- Collision links: ll_dg_1_4, ll_dg_2_4, ll_dg_3_4, ll_dg_4_4, ll_dg_5_4
- Fingertip xforms: ll_dg_1_tip, ll_dg_2_tip, etc.
- Palm: ll_dg_palm
"""

from isaaclab_assets.robots import HDR35_20_DG5F_L_CFG

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

# Import from our base class
from ... import remove_hook_env_cfg as remove_hook
from ... import mdp


# ============================================================================
# Action Configuration
# ============================================================================


@configclass
class Hdr35Dg5fRelJointPosActionCfg:
    """Relative joint position action for HDR35_20 + DG5F_L (26 DoF)

    MODIFIED (2026-01-12): 조인트별 다른 action scale 적용
    =========================================================

    문제 분석:
    - iiwa + Allegro에서는 마지막 암 조인트(j6/j7)가 회전하는 문제가 없음
    - HDR35 + DG5F/RH56F1에서는 j5/j6이 심하게 회전함

    핵심 원인:
    1. 극단적인 Stiffness 불균형:
       - iiwa j7/Allegro = 25/3 = 8.3:1
       - HDR35 j6/DG5F = 1,500/4 = 375:1 (45배 더 불균형!)

    2. 핸드에 전달되는 토크 부족:
       - Allegro: 3.0 × 0.1 = 0.3 Nm
       - DG5F: 4.0 × 0.001 = 0.004 Nm (1/75)

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
    - hand: 4.0 × 0.1 = 0.4 Nm (Allegro의 0.3 Nm과 유사)
    """

    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            r"j[1-5]": 0.01,           # 암 j1-j5: 기본 scale
            r"j6": 0.02,               # 암 j6 (wrist roll): 버퍼 역할
            r"lj_dg_[1-5]_[1-4]": 0.1, # DG5F 핸드: 높은 scale로 충분한 토크
        },
    )


# ============================================================================
# Reward Configuration
# ============================================================================


@configclass
class RemoveHookHdr35Dg5fRewardCfg(remove_hook.RemoveHookRewardsCfg):
    """
    Reward configuration for Remove Hook task with HDR35_20 + DG5F_L.

    Reward Structure:
    1. Reaching phase: Encourage hand to approach hook
    2. Grasping phase: Encourage stable grasp on hook
    3. Transport phase: Encourage moving hook to target position
    """

    # ========== Phase 1: Reaching Hook ==========
    # NOTE: DG5F_L targets right_ring (RH56F1_R targets left_ring)
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
            "object_cfg": SceneEntityCfg("object", body_names=["right_ring"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["ll_dg_palm", ".*_tip"]),
        },
    )

    # ========== Phase 2: Grasping Hook ==========
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=3.0,
        params={"threshold": 0.5},
    )

    # ========== Phase 3: Transport to Target ==========
    # NOTE: DG5F_L targets right_ring
    # FIXED (2026-01-12): std 0.02 → 0.2 (target_offset 거리 ~0.36m에서 gradient 확보)
    # 문제: std=0.02이면 tanh(0.36/0.02)=tanh(18)≈1 → reward≈0 (학습 신호 없음)
    # 해결: std=0.2이면 tanh(0.36/0.2)≈0.94 → reward≈0.06 (탐색 가능)
    hook_to_target = RewTerm(
        func=mdp.hook_to_target_distance,
        weight=10.0,
        params={
            "std": 0.2,  # Was 0.02 - too small for target distance ~0.36m
            "target_offset": (0.2, 0.0, 0.3),  # Right 0.2m, up 0.3m (반대 방향)
            "hook_cfg": SceneEntityCfg("object", body_names=["right_ring"]),
        },
    )


# ============================================================================
# Main Environment Configuration
# ============================================================================


@configclass
class RemoveHookHdr35Dg5fEnv_Cfg(remove_hook.RemoveHookEnvCfg):
    """Remove Hook environment configuration for HDR35_20 + DG5F_L (Training)"""

    # Override actions and rewards
    actions: Hdr35Dg5fRelJointPosActionCfg = Hdr35Dg5fRelJointPosActionCfg()
    rewards: RemoveHookHdr35Dg5fRewardCfg = RemoveHookHdr35Dg5fRewardCfg()

    def __post_init__(self):
        # Call parent post_init first
        super().__post_init__()
        self.scene.robot = HDR35_20_DG5F_L_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ============================================================================
        # CONTACT SENSORS (Fingertips to Hook/Wire)
        # ============================================================================
        # DG5F Left uses ll_dg_X_4 for fingertip collision links
        finger_tip_body_list = [
            "ll_dg_1_4",    # Thumb (finger 1) last link
            "ll_dg_2_4",    # Index (finger 2) last link
            "ll_dg_3_4",    # Middle (finger 3) last link
            "ll_dg_4_4",    # Ring (finger 4) last link
            "ll_dg_5_4",    # Little (finger 5) last link
        ]

        # Create contact sensors for hook/wire grasping
        # NOTE: DG5F_L (왼손)은 right_ring을 타겟으로 함
        for link_name in finger_tip_body_list:
            sensor_name = link_name + "_hook_sensor"
            setattr(
                self.scene,
                sensor_name,
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Wire/right_ring*"],  # right_ring만 감지
                ),
            )

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
        # DG5F uses ll_dg_palm and ll_dg_X_tip
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = [
            "ll_dg_palm",     # Palm xform
            ".*_tip",         # All fingertip xforms (ll_dg_1_tip ~ ll_dg_5_tip)
        ]


@configclass
class RemoveHookHdr35Dg5fEnv_Cfg_PLAY(remove_hook.RemoveHookEnvCfg_PLAY):
    """Remove Hook environment configuration for HDR35_20 + DG5F_L (Inference/Play)"""

    # Override actions and rewards (same as training)
    actions: Hdr35Dg5fRelJointPosActionCfg = Hdr35Dg5fRelJointPosActionCfg()
    rewards: RemoveHookHdr35Dg5fRewardCfg = RemoveHookHdr35Dg5fRewardCfg()

    def __post_init__(self):
        # Call parent post_init
        super().__post_init__()
        self.scene.robot = HDR35_20_DG5F_L_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Contact sensors
        finger_tip_body_list = ["ll_dg_1_4", "ll_dg_2_4", "ll_dg_3_4", "ll_dg_4_4", "ll_dg_5_4"]
        for link_name in finger_tip_body_list:
            sensor_name = link_name + "_hook_sensor"
            setattr(
                self.scene,
                sensor_name,
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Wire/right_ring*"],
                ),
            )

        # Observations
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [link + "_hook_sensor" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),
        )

        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = [
            "ll_dg_palm",
            ".*_tip",
        ]
