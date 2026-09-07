# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the UH035 arm with Inspire Hand Right.

The following configurations are available:

* :obj:`UH035_INSPIRE_RIGHT_CFG`: UH035 (35kg payload) + Inspire Hand Right (6-DOF under-actuated) with implicit actuator model.

Reference:
* UH035: 35kg payload industrial robotic arm (6-DOF)
* Inspire Hand RH56F1_R: 6-DOF under-actuated robotic hand with mimic joints

URDF Analysis (from USD export):
* Link masses: base~link4=1kg, link5=0.5kg, link6=0.0001kg (simplified values)
* Joint effort limits: 10000000 N⋅m (no limits, requires manual tuning)
* Mass values are approximations from USD export, not actual robot specs

Key Differences from HDR20-DG5F:
- UH035 vs HDR20-17: Heavier payload (35kg vs 20kg) but similar link mass in URDF
- Inspire vs DG5F: Under-actuated hand (6 actuated DOF vs 20 DOF) → Lower stiffness for compliance
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

# USD 파일 경로 (UH035 + Inspire Hand Right)
USD_PATH = "/home/dyros/IsaacLab/hyundai/1_factory_usd_file/UH035_usd/UH035_gripper_R.usd"

UH035_INSPIRE_RIGHT_CFG = ArticulationCfg(
    # prim_path는 환경 쪽에서 replace()로 주입할 예정
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # === 팔 (Arm) - UH035 (6-DOF) ===
            # HDR20과 동일한 초기 자세 유지 (범위 내에서)
            "joint1": 0.0,          # Base rotation
            "joint2": 2.0933,       # 120° (shoulder pitch)
            "joint3": -0.7854,      # -45° (shoulder roll)
            "joint4": 0.0,          # Elbow
            "joint5": -0.7854,      # -45° (wrist pitch)
            "joint6": 0.0,          # Wrist roll

            # === 손 (Hand) - Inspire Right (6 actuated DOF) ===
            # Pre-grasp 자세: 손가락을 적당히 벌리고 약간 굽힌 상태

            # 엄지 (Thumb - 2 actuated joints)
            "right_thumb_1_joint": 0.8,      # 엄지 벌림: [0, 2.0944] 범위 내에서 적당히 벌림
            "right_thumb_2_joint": 0.2,      # 엄지 굽힘: [0, 0.4746] 범위 내에서 살짝 굽힘

            # 검지 (Index - 1 actuated joint, index_2는 mimic)
            "right_index_1_joint": 0.4,      # [0, 1.5286] 범위 내에서 적당히 굽힘

            # 중지 (Middle - 1 actuated joint, middle_2는 mimic)
            "right_middle_1_joint": 0.3,     # [0, 1.5286] 범위 내에서 약간 굽힘

            # 약지 (Ring - 1 actuated joint, ring_2는 mimic)
            "right_ring_1_joint": 0.3,       # [0, 1.5286] 범위 내에서 약간 굽힘

            # 소지 (Little - 1 actuated joint, little_2는 mimic)
            "right_little_1_joint": 0.3,     # [0, 1.5286] 범위 내에서 약간 굽힘
        },
    ),

    # 암+핸드 전부 implicit actuator로 제어
    actuators={
        "uh035_inspire_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                # Arm joints (6-DOF)
                r"joint(1|2|3|4|5|6)",
                # Hand joints (6 actuated DOF)
                r"right_thumb_(1|2)_joint",
                r"right_(index|middle|ring|little)_1_joint",
            ],

            # === Effort Limits (토크/힘 제한) ===
            # URDF 분석 결과: joint effort limits는 모두 10000000 (제한 없음)
            # 따라서 35kg payload와 실제 로봇 특성을 고려하여 합리적인 값 설정
            effort_limit_sim={
                # UH035 Arm (35kg payload)
                # Link mass (URDF): base~link4=1kg, link5=0.5kg, link6=0.0001kg
                # URDF의 mass 값은 USD export 시 단순화된 값
                # 실제 35kg payload 로봇의 자체 무게는 30-50kg 정도로 추정

                # 35kg payload + arm weight를 고려한 effort limit 설정:
                # - Base/Shoulder joints: 큰 모멘트 암 × 35kg payload
                # - Wrist joints: 작은 모멘트 암, 상대적으로 낮은 토크
                r"joint1": 400.0,       # Base rotation (가장 큰 토크, ~35kg × 1m × 10)
                r"joint2": 400.0,       # Shoulder pitch (무거운 payload 지탱)
                r"joint3": 350.0,       # Shoulder roll
                r"joint4": 100.0,       # Elbow (모멘트 암 작아짐)
                r"joint5": 100.0,       # Wrist pitch
                r"joint6": 50.0,        # Wrist roll (가장 작은 토크)

                # Inspire Hand (6 actuated DOF - URDF에서 effort=1.0으로 지정됨)
                # 매우 가벼운 under-actuated hand이므로 낮은 effort limit 사용
                r"right_thumb_(1|2)_joint": 1.0,
                r"right_(index|middle|ring|little)_1_joint": 1.0,
            },

            # === Stiffness (강성) ===
            # PD 제어의 P 게인과 유사, 위치 오차에 대한 복원력
            # URDF link mass: 1kg (가볍게 설정됨) + 35kg payload
            stiffness={
                # UH035 Arm (35kg payload)
                # URDF link mass는 1kg으로 가볍지만, 35kg payload 고려 필수
                #
                # Stiffness 설정 전략:
                # - Link mass가 가볍더라도 35kg payload를 제어하려면 충분한 stiffness 필요
                # - 하지만 URDF mass가 실제보다 가볍게 설정되어 있으므로
                #   너무 높은 stiffness는 진동/불안정성 유발 가능
                # - HDR20-17 (20kg payload, link mass ~3-5kg) 대비 적절히 조정

                r"joint(1|2|3)": 30000.0,   # joint1-joint3: Base/Shoulder joints
                                        # 35kg payload를 지탱 (HDR20: 20000)
                                        # URDF mass는 가볍지만 payload 중심으로 설정

                r"joint4": 3000.0,          # joint4-joint6: Wrist joints (HDR20: 2000)
                r"joint5": 3000.0,          # 손목 관절, 모멘트 암이 작아 상대적으로 낮음
                r"joint6": 1500.0,          # Wrist roll (HDR20: 1000)

                # Inspire Hand (6-DOF under-actuated)
                # DG5F(20-DOF, stiffness=4.0)보다 낮은 값 사용
                # 이유: Under-actuated 설계로 compliance(유연성)가 중요
                #       물체에 적응적으로 감싸는 것이 목표
                r"right_thumb_(1|2)_joint": 2.5,             # 엄지: 적당한 강성
                r"right_(index|middle|ring|little)_1_joint": 2.5,  # 나머지 손가락: 동일한 강성
                # 참고: Kuka-Allegro는 3.0 사용, DG5F는 4.0 사용
                #       Inspire는 under-actuated이므로 2.5로 더 낮게 설정
            },

            # === Damping (감쇠) ===
            # PD 제어의 D 게인과 유사, 속도 오차에 대한 저항력
            # 일반적으로 stiffness의 1/10 ~ 1/100 사용 (진동 억제)
            damping={
                # UH035 Arm
                # Damping = stiffness / 10 ~ 15 정도로 설정
                r"joint(1|2|3)": 2000.0,    # stiffness(30000) / 15
                                        # URDF link mass가 가볍게 설정되어 있으므로
                                        # 과도한 damping은 응답속도 저하 유발
                                        # (HDR20: 1500, stiffness 20000 / 13.3)

                r"joint4": 300.0,           # stiffness(3000) / 10 (HDR20: 200)
                r"joint5": 300.0,           # 손목 관절 damping
                r"joint6": 150.0,           # stiffness(1500) / 10 (HDR20: 100)

                # Inspire Hand
                # stiffness(2.5)에 비례하여 damping 설정
                r"right_thumb_(1|2)_joint": 0.25,            # stiffness / 10
                r"right_(index|middle|ring|little)_1_joint": 0.25,
                # 참고: Kuka-Allegro는 0.1 사용, DG5F는 0.3 사용
                #       Inspire는 중간 값인 0.25 사용
            },

            # === Friction (마찰) ===
            # 정마찰 계수, 움직임 시작 시 저항
            friction={
                # UH035 Arm - 큰 관절들은 마찰을 더 줌
                r"joint(1|2|3)": 5.0,       # 무거운 base/shoulder 관절
                r"joint(4|5|6)": 0.5,       # 상대적으로 가벼운 wrist 관절

                # Inspire Hand - 손가락은 민감해야 하므로 마찰을 매우 낮게 유지
                r"right_thumb_(1|2)_joint": 0.01,
                r"right_(index|middle|ring|little)_1_joint": 0.01,
            },
        ),
    },

    # 소프트 조인트 한계 (기본값 유지)
    soft_joint_pos_limit_factor=1.0,
)
