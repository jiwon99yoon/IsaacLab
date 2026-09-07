# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ============================================================================
# USD STRUCTURE (HDR035 + ATI + DG5F Right Hand)
# ============================================================================
# The USD file includes tip xforms under dg5f_right_new:
#   - ll_dg_palm (palm reference point)
#   - ll_dg_1_tip, ll_dg_2_tip, ll_dg_3_tip, ll_dg_4_tip, ll_dg_5_tip (fingertips)
#
# This matches HDR20-17 DG5F structure which has:
#   - ll_dg_palm, ll_dg_1_tip, ll_dg_2_tip, etc.
#
# Configuration Structure:
# 1. CONTACT SENSORS: Placed on collision links (ll_dg_1_4, ll_dg_2_4, etc.)
#    - prim_path uses slash: "dg5f_right_new/ll_dg_1_4"
#    - sensor name uses underscore: "ur10e_dg5f_right_new_ll_dg_1_4_object_s"
#
# 2. OBSERVATIONS/REWARDS: Use tip xforms for accurate fingertip positions
#    - body_names: ["ll_dg_palm", "ll_dg_.*_tip"] (regex pattern matches all tip xforms)
#    - This provides actual fingertip positions, not link origins
# ============================================================================

from isaaclab_assets.robots import UH035_ATI_DG5F_LEFT_CFG

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp


@configclass
class Hdr035Dg5fLeftRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)

@configclass
class Hdr035Dg5fLeftReorientRewardCfg(dexsuite.RewardsCfg):
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
    # NOTE: DG5F Right sensor names defined inline to avoid RewardManager parsing issues
    # Cannot use class attribute as RewardManager expects only RewardTermCfg instances

    good_finger_contact = RewTerm(
        func = mdp.contacts,
        weight=0.5,
        params={"threshold" : 1.0},
    )

    # good_finger_contact = RewTerm(
    #     func=mdp.contacts,
    #     weight=0.5,  # REVERTED to Kuka-Allegro baseline (was 2.0 in UH035)
    #     params={
    #         "threshold": 1.0,
    #         "contact_sensor_names": [
    #             "ur10e_dg5f_right_new_ll_dg_1_4_object_s",  # Thumb
    #             "ur10e_dg5f_right_new_ll_dg_2_4_object_s",  # Index
    #             "ur10e_dg5f_right_new_ll_dg_3_4_object_s",  # Middle
    #             "ur10e_dg5f_right_new_ll_dg_4_4_object_s",  # Ring
    #             "ur10e_dg5f_right_new_ll_dg_5_4_object_s"   # Pinky
    #         ]
    #     },
    # )

    # position_tracking = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=2.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "std": 0.2,
    #         "command_name": "object_pose",
    #         "align_asset_cfg": SceneEntityCfg("object"),
    #         "contact_sensor_names": [
    #             "ur10e_dg5f_right_new_ll_dg_1_4_object_s",  # Thumb
    #             "ur10e_dg5f_right_new_ll_dg_2_4_object_s",  # Index
    #             "ur10e_dg5f_right_new_ll_dg_3_4_object_s",  # Middle
    #             "ur10e_dg5f_right_new_ll_dg_4_4_object_s",  # Ring
    #             "ur10e_dg5f_right_new_ll_dg_5_4_object_s"   # Pinky
    #         ],
    #     },
    # )

    # success = RewTerm(
    #     func=mdp.success_reward,
    #     weight=10,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "pos_std": 0.1,
    #         "rot_std": 0.5,
    #         "command_name": "object_pose",
    #         "align_asset_cfg": SceneEntityCfg("object"),
    #         "contact_sensor_names": [
    #             "ur10e_dg5f_right_new_ll_dg_1_4_object_s",  # Thumb
    #             "ur10e_dg5f_right_new_ll_dg_2_4_object_s",  # Index
    #             "ur10e_dg5f_right_new_ll_dg_3_4_object_s",  # Middle
    #             "ur10e_dg5f_right_new_ll_dg_4_4_object_s",  # Ring
    #             "ur10e_dg5f_right_new_ll_dg_5_4_object_s"   # Pinky
    #         ],
    #     },
    # )

@configclass

class Hdr035Dg5fLeftMixinCfg:
    rewards: Hdr035Dg5fLeftReorientRewardCfg = Hdr035Dg5fLeftReorientRewardCfg()
    actions: Hdr035Dg5fLeftRelJointPosActionCfg = Hdr035Dg5fLeftRelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "ll_dg_palm" #"ll_dg_mount"
        self.scene.robot = UH035_ATI_DG5F_LEFT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        finger_tip_body_list = ["ll_dg_1_4", "ll_dg_2_4", "ll_dg_3_4", "ll_dg_4_4", "ll_dg_5_4"]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/dg5f_left_flattened/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["ll_dg_palm", ".*_tip"] #"ll_dg_.*_4"] #".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["ll_dg_palm", ".*_tip"]) #"ll_dg_.*_4"]) #".*_tip"])

# #========================================================================================================

#     def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
#         super().__post_init__()

#         # === DG5F Right Hand Configuration ===
#         # DG5F Right Hand Structure (from USD):
#         # - Palm: ll_dg_base, ll_dg_palm (palm xform)
#         # - 5 Fingers (20-DOF): Finger 1-5, each with 4 joints (lj_dg_1_1 ~ lj_dg_5_4)
#         # - 5 Fingertips: ll_dg_1_tip, ll_dg_2_tip, ll_dg_3_tip, ll_dg_4_tip, ll_dg_5_tip
#         # - Collision links: ll_dg_1_4, ll_dg_2_4, ll_dg_3_4, ll_dg_4_4, ll_dg_5_4 (last link of each finger)

#         # Command body: Use ll_dg_palm xform (matching HDR20-17 DG5F structure)
#         # NOTE: Isaac Lab flattens body names, so no ur10e/ prefix needed
#         self.commands.object_pose.body_name = "ll_dg_palm"

#         # Robot configuration
#         self.scene.robot = UH035_ATI_DG5F_LEFT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

#         # Contact sensor locations (placed on collision links, not tip xforms)
#         # These are the last links (4th link) of each finger where collision actually occurs
#         # Matching HDR20-17 pattern: contact sensors on links (ll_dg_1_4, ll_dg_2_4, etc.)
#         # NOTE: UR10e has "ur10e/" prefix due to extra Xform layer in USD structure
#         finger_tip_body_list = [
#             "ur10e/dg5f_right_new/ll_dg_1_4",    # Finger 1 (Thumb) collision link
#             "ur10e/dg5f_right_new/ll_dg_2_4",    # Finger 2 (Index) collision link
#             "ur10e/dg5f_right_new/ll_dg_3_4",    # Finger 3 (Middle) collision link
#             "ur10e/dg5f_right_new/ll_dg_4_4",    # Finger 4 (Ring) collision link
#             "ur10e/dg5f_right_new/ll_dg_5_4"     # Finger 5 (Pinky) collision link
#         ]

#         # Create contact sensors for each collision link
#         # Sensor names use underscore (e.g., "ur10e_dg5f_right_new_ll_dg_1_4_object_s")
#         # Prim paths use slash (e.g., "{ENV_REGEX_NS}/Robot/ur10e/dg5f_right_new/ll_dg_1_4")
#         for link_name in finger_tip_body_list:
#             sensor_name = link_name.replace("/", "_") + "_object_s"
#             setattr(
#                 self.scene,
#                 sensor_name,
#                 ContactSensorCfg(
#                     prim_path="{ENV_REGEX_NS}/Robot/" + link_name,
#                     filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
#                 ),
#             )

#         # Observation: Contact force from collision links
#         # Convert "/" to "_" in sensor names (Python attribute naming requirement)
#         self.observations.proprio.contact = ObsTerm(
#             func=mdp.fingers_contact_force_b,
#             params={"contact_sensor_names": [link.replace("/", "_") + "_object_s" for link in finger_tip_body_list]},
#             clip=(-50.0, 50.0),  # Contact force in finger tips (DG5F: effort=5.0, higher than Inspire)
#         )

#         # Observation: Hand tips state (palm + all fingertips)
#         # Uses tip xforms in USD (matching HDR20-17 DG5F structure)
#         # DG5F: ["ll_dg_palm", "ll_dg_.*_tip"] - Isaac Lab flattens body names
#         self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = [
#             "ll_dg_palm",      # Palm xform
#             "ll_dg_.*_tip"     # All fingertip xforms (ll_dg_1_tip ~ ll_dg_5_tip)
#         ]

#         # Reward: Distance from fingers to object
#         # Uses tip xforms in USD (matching HDR20-17 DG5F structure)
#         # DG5F: ["ll_dg_palm", "ll_dg_.*_tip"] - Isaac Lab flattens body names
#         self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
#             "robot",
#             body_names=[
#                 "ll_dg_palm",      # Palm xform
#                 "ll_dg_.*_tip"     # All fingertip xforms (ll_dg_1_tip ~ ll_dg_5_tip)
#             ]
#         )

#         # [DISABLED] reset_robot_wrist_joint
#         # 6-DOF UR10e에서는 7-DOF Kuka의 null-space 탐색이 불필요하므로
#         # dexsuite_env_cfg.py에서 reset_robot_wrist_joint를 주석 처리함
#         # -------------------------------------------------------------------------
#         # if self.events.reset_robot_wrist_joint is not None:
#         #     self.events.reset_robot_wrist_joint.params["asset_cfg"] = SceneEntityCfg(
#         #         "robot", joint_names="elbow_joint"
#         #     )

#         # DISABLE: reset_thumb_2_joint (Inspire Hand specific)
#         # DG5F Right hand doesn't have left_thumb_2_joint
#         # DG5F uses lj_dg_1_1 ~ lj_dg_1_4 for thumb instead
#         # Domain randomization for all joints (including thumb) is handled by reset_robot_joints
#         # NOTE: Skip if event is already None (disabled in Play mode)
#         try:
#             if hasattr(self.events, "reset_thumb_2_joint") and self.events.reset_thumb_2_joint is not None:
#                 delattr(self.events, "reset_thumb_2_joint")
#         except AttributeError:
#             pass  # Event doesn't exist, which is fine

#         # OVERRIDE: reset_hand_joints (Remove Inspire Hand joints, keep only DG5F)
#         # Base config (dexsuite_env_cfg.py) includes both DG5F and Inspire joints
#         # This causes errors when Inspire joints don't exist in DG5F robot
#         # Solution: Override to include only DG5F hand joints
#         if hasattr(self.events, "reset_hand_joints") and self.events.reset_hand_joints is not None:
#             self.events.reset_hand_joints.params["asset_cfg"] = SceneEntityCfg(
#                 "robot",
#                 joint_names=[
#                     # DG5F Right Hand (20 DOF) - bending joints only (_2, _3, _4)
#                     # Excludes spread joints (_1) to prevent finger collision
#                     r"lj_dg_[1-5]_[234]",
#                 ]
#             )


@configclass
class DexsuiteHdr035Dg5fLeftLiftEnvCfg(Hdr035Dg5fLeftMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass

@configclass
class DexsuiteHdr035Dg5fLeftLiftEnvCfg_PLAY(Hdr035Dg5fLeftMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass


# ============================================================================
# ATI F/T SENSOR VARIANT (Forge approach - IMPROVED)
# ============================================================================
#
# REFERENCE: isaaclab_tasks/direct/forge/forge_env.py
#
# ISSUE HISTORY (2025-11-23):
# - Previous config had: smoothing_factor=0.9, threshold=20N, weight=-0.3
# - Result: abnormal_robot termination 100%, episode_length ~2.4 steps
# - Root causes:
#   1. smoothing_factor 0.9 → noisy readings, slow smoothing
#   2. threshold 20N too low (DG5F hand ~34N static force from gravity)
#   3. weight -0.3 too high → policy couldn't explore
#
# KEY CHANGES:
# - smoothing_factor: 0.9 → 0.25 (Forge default, forge_env_cfg.py:100)
# - force_threshold: 20N → 50N (34N static + 16N margin for motion)
# - weight: -0.3 → -0.05 (auxiliary role, not dominant)
# - Added: ati_force_threshold observation (policy knows acceptable force)
#
# ============================================================================

@configclass
class Hdr035Dg5fLeftMixinCfg_FT(Hdr035Dg5fLeftMixinCfg):
    """Mixin for UR10e + DG5F Right with ATI F/T Sensor.

    REFERENCE: isaaclab_tasks/direct/forge/forge_env.py

    Inherits from Ur10eDg5fRightMixinCfg and adds:
    - ATI F/T force observations (3D) + threshold observation
    - ATI-based collision penalty rewards (ReLU style)

    Uses Forge approach: reads force directly from PhysX via get_link_incoming_joint_force().
    No ForceTorqueSensorCfg needed - works with regular RigidBody in USD.

    OBSERVATION ADDITIONS (vs non-FT environment):
    - ati_force: 3D force vector in ATI sensor local frame [3]
    - ati_force_threshold: Acceptable force threshold [1]

    REWARD ADDITIONS (vs non-FT environment):
    - ati_excessive_force_penalty: ReLU(||force|| - threshold) penalty

    THRESHOLD RATIONALE:
    - UR10e + DG5F hand weight: ~3.5kg (hand + ATI sensor)
    - Static force on ATI from gravity: 3.5kg × 9.8 = ~34N
    - Threshold must be ABOVE static force: 50N = 34N + 16N margin
    - Forge uses [5.0, 10.0] but with lighter Franka gripper (~0.7kg)
    """
    # rewards and actions are inherited from Ur10eDg5fRightMixinCfg

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        # Call parent mixin (sets up robot, sensors, etc.)
        super().__post_init__()

        # ===================================================================
        # ATI F/T SENSOR OBSERVATIONS
        # Reference: forge_env.py:95-101, 129
        # ===================================================================
        # No need to add sensor to scene - PhysX handles it automatically!

        # Observation 1: Force vector (3D) in ATI sensor local frame
        # CHANGED: smoothing_factor 0.9 → 0.25 (Forge default)
        # Reference: forge_env_cfg.py:100 (ft_smoothing_factor=0.25)
        self.observations.proprio.ati_force = ObsTerm(
            func=mdp.ati_force_sensor,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_body_name": "ATI_Axia90_M50",
                "smoothing_factor": 0.25,  # CHANGED: 0.9 → 0.25 (Forge default)
            },
            # CHANGED: clip (-100, 100) → (-200, 200) → (-150, 150)
            # Reason: observations.py에서 magnitude clamping을 150N으로 하므로 일치시킴
            # IMPORTANT: Component clip이므로 150N magnitude = 각 축 최대 ~87N (150/√3)
            # 하지만 실제로는 magnitude가 150N으로 보장되므로 여유있게 (-150, 150) 설정
            clip=(-150.0, 150.0),
        )

        # Observation 2: Force threshold (Forge style - policy knows acceptable force)
        # Reference: forge_env.py:129 - "force_threshold": self.contact_penalty_thresholds[:, None]
        # This is ESSENTIAL when using randomized thresholds (Phase 2)
        #
        # IMPORTANT: 이 값은 ati_excessive_force_penalty의 threshold와 **반드시 일치**해야 함!
        #   - Policy가 배우는 것: "이 threshold 넘으면 penalty"
        #   - Reward가 주는 것: 실제 penalty
        #   - 불일치 시: Policy 혼란 → 학습 불안정
        #
        # MODIFICATION HISTORY:
        # - Original: 50N (손 무게 34N 대비 너무 낮음)
        # - 2nd: 100N (평균 force 80N 기준, reward와 일치)
        # - MODIFIED (11290200): 150N (reward threshold와 일치)
        #
        # RATIONALE (11290200):
        # 문제: Observation-Reward threshold 불일치
        #   - 이전: Obs 100N, Reward 150N
        #   - Policy: "100N 넘으면 penalty" 학습
        #   - 실제: 150N부터 penalty 시작
        #   - 결과: Policy 혼란, 불필요한 회피 행동
        #
        # 해결: Observation threshold = Reward threshold = 150N
        #   - Policy: "150N 넘으면 penalty" 학습 (정확!)
        #   - 실제: 150N부터 penalty 시작 (일치!)
        #   - 평균 force 80N, 정상 grasping 110-130N → penalty 없음
        #
        # MODIFIED (2025-11-30): Phase 2 구현 - Threshold randomization
        #   - Threshold randomization: [150, 200]N (Forge 방식) ✓ 구현 완료
        #   - Episode reset 시 환경마다 다른 threshold ✓ 자동 처리 (ati_force_threshold 함수 내부)
        #   - Policy adaptive learning: 다양한 threshold에 robust
        #
        # MODIFIED (251130): Phase 1 적용
        #   - Observation clamp: 150N → 250N (observations.py)
        #   - Penalty threshold: 150-200N randomization
        #   - 이제 penalty가 제대로 작동 가능 (clamp > threshold)
        #
        # Phase 2 TODO (ADR Curriculum - 만약 Phase 1 학습 결과 불만족 시):
        #   - Threshold를 difficulty에 따라 조정
        #   - Initial (쉬움): threshold = 200N (관대)
        #   - Final (어려움): threshold = 150N (엄격)
        #   - curriculums.py에 initial_final_interpolate_fn 활용
        #   - 학습 초기: 기본기 배우기 (관대한 threshold)
        #   - 학습 후기: 정밀 제어 (엄격한 threshold)
        self.observations.proprio.ati_force_threshold = ObsTerm(
            func=mdp.ati_force_threshold,
            params={
                "threshold_range": (150.0, 200.0),  # MODIFIED (2025-11-30): 고정값 → randomization [150, 200]N
                # Episode마다 다른 threshold로 학습 → robust policy
                # 평균 ~175N → 정상 grasping (110-130N) 허용하면서 충돌 감지
                # MODIFIED (251130): Observation clamp 250N으로 상향 → penalty 작동 가능
            },
        )

        # Option B (commented): Force + Torque (6D) - for future expansion
        # Torque can indicate grasp quality and slip detection
        # TODO Phase 2: Enable 6D observation for grasp quality assessment
        # self.observations.proprio.ati_force_torque = ObsTerm(
        #     func=mdp.ati_force_torque_sensor,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot"),
        #         "sensor_body_name": "ATI_Axia90_M50",
        #         "smoothing_factor": 0.25,
        #     },
        #     clip=(-100.0, 100.0),
        # )

        # ===================================================================
        # ATI F/T SENSOR REWARDS
        # Reference: forge_env.py:237-239 (contact_penalty calculation)
        # ===================================================================

        # Penalty: Excessive force (collision detection & hardware protection)
        # Uses ReLU-style penalty: penalty = ReLU(||force|| - threshold)
        # Only penalizes forces ABOVE threshold, zero penalty below
        #
        # PENALTY FORMULA:
        #   penalty = ReLU(||force|| - threshold) × weight
        #   - force < threshold: penalty = 0
        #   - force > threshold: penalty proportional to excess force
        #
        # MODIFICATION HISTORY:
        # - Original: weight=-0.3, threshold=20N (너무 강함)
        # - 1st: weight=-0.05, threshold=50N (여전히 누적 시 dominant)
        # - 2nd: weight=-0.01, threshold=100N (여전히 contact 회피 학습)
        # - MODIFIED (11290135): weight=-0.005, threshold=150.0N
        #
        # RATIONALE (11290135):
        # 문제 진단 (Tensorboard 분석):
        #   - good_finger_contact reward: 0.05 이하 (거의 접촉 안함)
        #   - episode_length: 150 steps (정상의 60%)
        #   - Agent가 물체를 "잡지 않는" 학습 발생
        #
        # 원인 분석:
        #   - 평균 force: 80N (EXPECTED_MEAN_FORCE, rewards.py:469)
        #   - 이전 threshold: 100N (평균 대비 +25% margin만)
        #   - Grasping 시 force: 110-130N → penalty 발생
        #   - Reward balance: good_finger_contact(+0.5) vs penalty(-0.01×150=-1.5)
        #   - 결과: 접촉 회피가 더 이득 (0.0 > -1.0)
        #
        # 해결책:
        #   1. Threshold 상향: 100N → 150N (평균 80N 대비 +88% margin)
        #      → 정상 grasping (100-130N)에서 penalty 없음
        #      → 극단적 collision (>150N)만 penalty
        #   2. Weight 절감: -0.01 → -0.005 (절반)
        #      → 누적 penalty 감소: 150 steps × 0.005 = 0.75
        #      → good_finger_contact(+0.5)와 balance 개선
        #
        # Phase 2 TODO (학습 안정화 후):
        #   - Threshold randomization: [150, 200]N (Forge 방식)
        #   - EventTerm 추가하여 환경마다 다른 threshold 적용
        #   - Policy adaptive learning 강화
        #
        # Reference: forge_env.py:237-239
        #   contact_penalty = torch.relu(fs_norm - self.contact_penalty_thresholds)
        self.rewards.ati_excessive_force_penalty = RewTerm(
            func=mdp.ati_excessive_force_penalty,
            weight=-0.005,  # MODIFIED (11290135): -0.01 → -0.005
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_body_name": "ATI_Axia90_M50",
                "use_randomized_threshold": True,  # MODIFIED (2025-11-30): env._ati_force_threshold_buf 사용
                # Observation의 randomized threshold [150, 200]N를 자동으로 사용
                #
                # MODIFIED (251130): Phase 1 적용
                #   - Observation clamp: 250N
                #   - Penalty threshold: 150-200N (randomized)
                #   - penalty = ReLU(force - threshold) 제대로 작동
                #   - 정상 grasping (50-130N): penalty 없음
                #   - 충돌 (150-250N): penalty 발생 ✓
                #
                # Phase 2 TODO (ADR - 만약 Phase 1 결과 불만족 시):
                #   - Weight를 difficulty에 따라 조정
                #   - Initial: -0.002 (관대)
                #   - Final: -0.005 (엄격)
                #   - 또는 threshold를 difficulty로 조정 (위 참조)
            },
        )

        # Metric: ATI force magnitude (Tensorboard tracking only)
        # CREATED (2025-11-30): FT sensor 값 실시간 모니터링용
        # weight=0 → reward에 영향 없음, Tensorboard에만 기록
        # 목적: 물체 파지 시 vs 충돌 시 force 값 차이 확인, threshold 적절성 판단
        self.rewards.ati_force_magnitude = RewTerm(
            func=mdp.ati_force_magnitude,
            weight=0.0,  # Tracking only, no reward impact
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_body_name": "ATI_Axia90_M50",
            },
        )

        # Penalty 2 (commented): Downward force (hand pressing table)
        # DISABLED for now - may conflict with grasping motion
        # TODO Phase 2: Consider enabling with proper tuning
        # self.rewards.ati_downward_force_penalty = RewTerm(
        #     func=mdp.ati_downward_force_penalty,
        #     weight=-0.02,  # Very light penalty
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot"),
        #         "sensor_body_name": "ATI_Axia90_M50",
        #         "z_threshold": -30.0,  # Only penalize significant downward force
        #     },
        # )

        # ===================================================================
        # OVERRIDE: reset_hand_joints (inherited from parent, but explicit for clarity)
        # NOTE: This is already handled by super().__post_init__() calling
        # Ur10eDg5fRightMixinCfg.__post_init__(), but we keep it here for clarity
        # and to ensure FT environments also have correct hand joint configuration
        # ===================================================================
        if hasattr(self.events, "reset_hand_joints") and self.events.reset_hand_joints is not None:
            self.events.reset_hand_joints.params["asset_cfg"] = SceneEntityCfg(
                "robot",
                joint_names=[
                    # DG5F Right Hand (20 DOF) - bending joints only (_2, _3, _4)
                    # Excludes spread joints (_1) to prevent finger collision
                    r"lj_dg_[1-5]_[234]",
                ]
            )


# @configclass
# class DexsuiteUr10eDg5fRightLiftEnvCfg_FT(Ur10eDg5fRightMixinCfg_FT, dexsuite.DexsuiteLiftEnvCfg):
#     """UR10e + DG5F Right Lift task with ATI F/T sensor for collision detection.

#     Differences from base config:
#     - Adds ATI force observations (3D)
#     - Adds excessive force penalty reward
#     """
#     pass


# @configclass
# class DexsuiteUr10eDg5fRightLiftEnvCfg_FT_PLAY(Ur10eDg5fRightMixinCfg_FT, dexsuite.DexsuiteLiftEnvCfg_PLAY):
#     """UR10e + DG5F Right Lift task (Play mode) with ATI F/T sensor."""
#     pass
