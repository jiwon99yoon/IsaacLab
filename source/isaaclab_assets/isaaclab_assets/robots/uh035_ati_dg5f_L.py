# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

import os
from pathlib import Path
import math


"""Configuration for the UH035 arm with DG5F Hand Left.
The following configurations are available:

* :obj:`UH035_ATI_DG5F_LEFT_CFG`: UH035 (35kg payload) + DG5F Hand LEFT (20-DOF fully-actuated) 

Reference:
* UH035: 35kg payload industrial robotic arm (6-DOF)
* ATI Axia80-M50: 6-axis Force/Torque sensor
* Tesollo Hand DG5F: 20-DOF actuated robotic hand

URDF Analysis (from USD export):
* Link masses: base~link4=1kg, link5=0.5kg, link6=0.0001kg (simplified values)
* Joint effort limits: 10000000 N⋅m (no limits, requires manual tuning)
* Mass values are approximations from USD export, not actual robot specs

Key Differences from HDR20-DG5F:
- UH035 vs HDR20-17: Heavier payload (35kg vs 20kg) but similar link mass in URDF
- DG5F (RIGHT vs LEFT)
"""

def deg2rad(degrees: float) -> float:
    """Convert degrees to radians.

    Helper function for setting joint positions in init_state.
    IsaacSim displays joint angles in degrees, but joint_pos requires radians.

    Args:
        degrees: Joint angle in degrees (from IsaacSim UI)

    Returns:
        Joint angle in radians

    Example:
        "shoulder_pan_joint": deg2rad(0.0),      # 0° -> 0.0 rad
        "shoulder_lift_joint": deg2rad(-120.0),  # -120° -> -2.0944 rad
        "elbow_joint": deg2rad(120.0),           # 120° -> 2.0944 rad
    """
    return math.radians(degrees)


# USD 파일 상대 경로
# 현재 실행 중인 .py 파일의 경로
CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))

# IsaacLab 경로
ISAACLAB_ROOT = CURRENT_DIR.parents[3]

# USD 파일의 상대 경로 (IsaacLab 루트부터 시작)
# USD_RELATIVE_PATH = "hyundai/1_factory_usd_file/diated/hdr035_ati_dg5f_L_flattened.usd"
USD_RELATIVE_PATH = "hyundai/1_factory_usd_file/diated/hdr035_ati_dg5f_L_flattened.usd"

# flattened + ft sensor와 mesh가 안겹치도록 재구성 + ft sensor 밑 joint의 값이 hand와 겹치도록! : hand아래 정보를 불러오도록 설정

# 최종 USD_PATH: 절대 경로를 다시 구성
USD_PATH = ISAACLAB_ROOT.joinpath(USD_RELATIVE_PATH).as_posix()

UH035_ATI_DG5F_LEFT_CFG = ArticulationCfg(
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
            "joint1": deg2rad(0.0),          # Base rotation
            "joint2": deg2rad(120.0),       # 120° (shoulder pitch)
            "joint3": deg2rad(-30.0),      # -30으로 해야할 것 같은데 
            "joint4": deg2rad(0.0),      # Elbow
            "joint5": deg2rad(0.0),      # wrist pitch
            "joint6": deg2rad(-30.0),      # Wrist roll

            # === 손 (Hand) - Pre-grasp 자세 (Grasping에 최적화) ===
            # DG5F_Right랑 부호 다른 부분들 있음

            # 1. 엄지 (Finger 1) - Opposition 자세 (물체를 향해 대치)
            "lj_dg_1_1": deg2rad(-10.0),        # 적당히 벌림 
            "lj_dg_1_2": deg2rad(30.0),       # 회전 강화 
            "lj_dg_1_3": deg2rad(-30.0),        # 살짝 굽힘
            "lj_dg_1_4": deg2rad(-30.0),        # 적당히 굽힘

            # 2. 검지, 중지 약지의 1 joint (Finger 2) - Opposition 자세 (물체를 향해 대치) : 손가락 사이 벌림
            "lj_dg_2_1": deg2rad(10.0),       #범위 
            "lj_dg_3_1": deg2rad(0.0),         #범위 
            "lj_dg_4_1": deg2rad(-5.0),        #범위 

            r"lj_dg_[2-4]_2": deg2rad(20.0),     # 첫 관절 적당히 굽힘
            r"lj_dg_[2-4]_3": deg2rad(20.0),     # 두번째 관절 적당히 굽힘
            r"lj_dg_[2-4]_4": deg2rad(30.0),     # 세번째 관절 적당히 굽힘

            # 기존 hdr20_dg5f에서 수정 : 약지가 2,3,4와 다른 configuration
            # 3. 소지 (Finger 5) - Opposition 자세 (물체를 향해 대치)
            "lj_dg_5_1": deg2rad(-10.0),        # 조금만 벌림 
            "lj_dg_5_2": deg2rad(-15.0),       # 회전 강화 
            "lj_dg_5_3": deg2rad(30.0),        # 살짝 굽힘
            "lj_dg_5_4": deg2rad(30.0),        # 적당히 굽힘

        },
    ),

    # 암+핸드 전부 implicit actuator로 제어
    actuators={
        "uh035_dg5f_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                # Arm joints (6-DOF)
                r"joint(1|2|3|4|5|6)",
                # Hand joints (20 actuated DOF)
                r"lj_dg_[1-5]_[1-4]",
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

                # 손가락 effort_limit 낮춤 (부드러운 grasping을 위해)
                # ORIGINAL: r"lj_dg_[1-5]_[1-4]": 7.5,
                r"lj_dg_[1-5]_[1-4]": 5.0,
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

                #       물체에 적응적으로 감싸는 것이 목표
                # 참고: Kuka-Allegro는 3.0 사용 -> 4.0 사용
                r"lj_dg_[1-5]_[1-4]": 4.0,  # Kuka의 1.7배 (부드러움 우선, DG5F가 약간 무거움)

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

                # stiffness(2.5)에 비례하여 damping 설정
                # 참고: Kuka-Allegro는 0.1 사용
                r"lj_dg_[1-5]_[1-4]": 0.3,  # Kuka의 3배 (stiffness 5.0에 비례)

            },

            # === Friction (마찰) ===
            # 정마찰 계수, 움직임 시작 시 저항
            friction={
                # UH035 Arm - 큰 관절들은 마찰을 더 줌
                r"joint(1|2|3)": 5.0,       # 무거운 base/shoulder 관절
                r"joint(4|5|6)": 0.5,       # 상대적으로 가벼운 wrist 관절

                # 손가락은 민감해야 하므로 마찰을 매우 낮게 유지
                r"lj_dg_[1-5]_[1-4]": 0.01,
            },
        ),
    },

    # 소프트 조인트 한계 (기본값 유지)
    soft_joint_pos_limit_factor=1.0,
)
