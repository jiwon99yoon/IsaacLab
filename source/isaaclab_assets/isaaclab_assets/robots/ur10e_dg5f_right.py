# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the UR10e arm with ATI F/T Sensor and Dg5f Hand Right.

The following configurations are available:

* :obj:`UR10E_DG5F_RIGHT_CFG`: UR10e + ATI Sensor + DG5F Hand RIGHT (6-DOF under-actuated) with implicit actuator model.

Reference:
* UR10e: Universal Robots UR10e (12.5kg payload, 1300mm reach)
* ATI Axia80-M50: 6-axis Force/Torque sensor
* Tesollo Hand DG5F: 20-DOF actuated robotic hand

Configuration Notes:
* UR10e joint names: shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint
* DG5F Right actuated joints: 20 DOF (all 5 dof)
* Total DOF: 6 (arm) + 20 (hand) = 26 actuated DOF

Key Tuning:
- UR10e actuator parameters based on official UR10e specs
- DG5F hand parameters based on UH035_DG5F_RIGHT_CFG
- Init pose optimized for dexsuite manipulation tasks (workspace center, pre-grasp position)
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

import os
from pathlib import Path
import math


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
# USD_RELATIVE_PATH = "hyundai/1_factory_usd_file/ur10e_usd/ur10e_ati_dg5f_right.usd"
USD_RELATIVE_PATH = "hyundai/1_factory_usd_file/ur10e_usd/ur10e_ati_dg5f_right_flattened_new.usd"
# flattened + ft sensor와 mesh가 안겹치도록 재구성 + ft sensor 밑 joint의 값이 hand와 겹치도록! : hand아래 정보를 불러오도록 설정



# 최종 USD_PATH: 절대 경로를 다시 구성
USD_PATH = ISAACLAB_ROOT.joinpath(USD_RELATIVE_PATH).as_posix()

# USD 파일 절대 경로
# USD file path (UR10e + ATI Sensor + DG5F Hand right)
# USD_PATH = "/home/dyros/IsaacLab/hyundai/1_factory_usd_file/ur10e_usd/ur10e_ati_dg5f_right.usd"


UR10E_DG5F_RIGHT_CFG = ArticulationCfg(
    # prim_path will be injected by environment using replace()
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
            enabled_self_collisions=False,  # MODIFIED (2025-12-01 10:30): True → False (ATI-hand collision 방지)
            # 이유: USD mesh 간격이 있어도 physics instability 시 여전히 collision 가능
            # → extreme force 발생 (96,743N ~ 205,033N)
            # → velocity explosion (4956 rad/s)
            # 해결: self-collision 완전 비활성화
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # Robot base position (on ground, table is at z=0.235)
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # === Arm (UR10e - 6-DOF) ===
            # Init pose optimized for dexsuite manipulation tasks
            # - Hand positioned ABOVE table (table at z=0.235m + 0.02m = 0.255m surface)
            # - Elbow bent for better manipulability
            # - Wrist oriented downward for grasping
            #
            # Comparison with reference configs:
            # UR10e default: [pi, -pi/2, pi/2, -pi/2, -pi/2, 0] → hand pointing forward
            # UH035 default: [0, 2.0933, -0.7854, 0, -0.7854, 0] → hand pointing upward
            # Kuka default: [0, 0, 0.7854, 1.5708, -1.5708, -1.5708, 0] → hand pointing downward
            #
            # MODIFIED: Raised shoulder to position hand ABOVE table instead of below
            # NOTE: Use deg2rad() to easily convert IsaacSim's degree values to radians
            "shoulder_pan_joint": deg2rad(0.0),          # 0° - Centered (facing forward)
            "shoulder_lift_joint": deg2rad(-120.0),      # -120° - RAISED for better object approach
            "elbow_joint": deg2rad(120.0),               # 120° - More bent for better manipulability
            "wrist_1_joint": deg2rad(-120.0),            # -120° - Pitch down
            # wrist_2&3 changed from inspire_left config of ur10e because of dg5f right hand (for init pos)
            "wrist_2_joint": deg2rad(180.0),             # 180° - Neutral roll (CHANGED for dg5f right)
            "wrist_3_joint": deg2rad(-180.0),            # -180° - Neutral yaw

            # === 손 (Hand) - Pre-grasp 자세 (Grasping에 최적화) ===

            # 1. 엄지 (Finger 1) - Opposition 자세 (물체를 향해 대치)
            "rj_dg_1_1": deg2rad(10.0),        # 적당히 벌림 (범위 [-0.38, 0.89])
            "rj_dg_1_2": deg2rad(-60.0),       # 회전 강화 (범위 [-3.14, 0.0])
            "rj_dg_1_3": deg2rad(15.0),        # 살짝 굽힘
            "rj_dg_1_4": deg2rad(20.0),        # 적당히 굽힘

            # 2. 검지, 중지 약지의 1 joint (Finger 2) - Opposition 자세 (물체를 향해 대치) : 손가락 사이 벌림
            "rj_dg_2_1": deg2rad(-10.0),       #범위 [-35, 24) 
            "rj_dg_3_1": deg2rad(0.0),         #범위 [-35, 24)  
            "rj_dg_4_1": deg2rad(5.0),        #범위 [-35, 24) 

            r"rj_dg_[2-4]_2": deg2rad(30.0),     # 첫 관절 적당히 굽힘 (범위 [0, ~2.0])
            r"rj_dg_[2-4]_3": deg2rad(20.0),     # 두번째 관절 적당히 굽힘
            r"rj_dg_[2-4]_4": deg2rad(30.0),     # 세번째 관절 적당히 굽힘

            # 기존 hdr20_dg5f에서 수정 : 약지가 2,3,4와 다른 configuration
            # 3. 소지 (Finger 5) - Opposition 자세 (물체를 향해 대치)
            "rj_dg_5_1": deg2rad(10.0),        # 조금만 벌림 
            "rj_dg_5_2": deg2rad(20.0),       # 회전 강화 
            "rj_dg_5_3": deg2rad(30.0),        # 살짝 굽힘
            "rj_dg_5_4": deg2rad(30.0),        # 적당히 굽힘

        },
    ),

    # Arm + Hand: all controlled with implicit actuators
    actuators={
        "ur10e_dg5f_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                # Arm joints (6-DOF)
                r"shoulder_pan_joint",
                r"shoulder_lift_joint",
                r"elbow_joint",
                r"wrist_[123]_joint",
                # Hand joints (20 actuated DOF)
                r"rj_dg_[1-5]_[1-4]",
            ],

            # === Effort Limits (Torque/Force limits) ===
            # UR10e official specs: Maximum joint torques vary by joint
            # Shoulder/elbow: higher torque capability due to 12.5kg payload
            # Wrist: lower torque for precision
            effort_limit_sim={
                # UR10e Arm (12.5kg payload)
                # Based on official UR10e specifications:
                # - Shoulder/Base joints: ~330 Nm (max torque)
                # - Elbow joint: ~150 Nm
                # - Wrist joints: ~54-56 Nm
                #
                # Tuned similar to Kuka-Allegro (300.0 for arm) but adjusted for UR10e specs
                r"shoulder_pan_joint": 330.0,       # Base rotation (highest torque)
                r"shoulder_lift_joint": 330.0,      # Shoulder pitch (heavy payload)
                r"elbow_joint": 150.0,              # Elbow (medium torque)
                r"wrist_1_joint": 54.0,             # Wrist pitch (precision)
                r"wrist_2_joint": 54.0,             # Wrist roll (precision)
                r"wrist_3_joint": 54.0,             # Wrist twist (precision)

                # 손가락 effort_limit 낮춤 (부드러운 grasping을 위해)
                # ORIGINAL: r"rj_dg_[1-5]_[1-4]": 7.5,
                r"rj_dg_[1-5]_[1-4]": 5.0,
            },

            # === Stiffness (Position control proportional gain) ===
            # UR10e stiffness based on official config:
            # - Shoulder: 1320.0
            # - Elbow: 600.0
            # - Wrist: 216.0
            #
            # However, with dg5f hand attached (~5kg), wrist stiffness needs adjustment
            # to avoid oscillation. Using slightly lower values than stock UR10e.
            stiffness={
                # UR10e Arm (12.5kg payload + dg5f hand ~5kg)
                # Based on official UR10e_CFG, slightly reduced for hand attachment
                r"shoulder_pan_joint": 1320.0,      # Shoulder joints (same as UR10e_CFG)
                r"shoulder_lift_joint": 1320.0,     # High stiffness for payload support
                r"elbow_joint": 600.0,              # Elbow (same as UR10e_CFG)
                r"wrist_1_joint": 200.0,            # Wrist (reduced from 216 due to hand weight)
                r"wrist_2_joint": 200.0,            # Lower stiffness reduces oscillation
                r"wrist_3_joint": 200.0,            # with under-actuated gripper

                # 손가락 stiffness 조정 (부드러운 grasping을 위해)
                # ORIGINAL (너무 rigid): r"rj_dg_[1-5]_[1-4]": 30.0,
                # 1st TRY: r"rj_dg_[1-5]_[1-4]": 8.0,
                # 2nd TRY (잘못된 방향): r"rj_dg_[1-5]_[1-4]": 12.0,
                # 참고: Kuka-Allegro는 3.0 사용 -> 4.0 사용
                r"rj_dg_[1-5]_[1-4]": 4.0,  # Kuka의 1.7배 (부드러움 우선, DG5F가 약간 무거움)

            },

            # === Damping (Velocity control derivative gain) ===
            # Damping = stiffness / ratio, typically ratio = 10~20
            # UR10e damping based on official config:
            # - Shoulder: 72.66 (stiffness/18.2)
            # - Elbow: 34.64 (stiffness/17.3)
            # - Wrist: 29.39 (stiffness/7.35)
            damping={
                # UR10e Arm
                # Based on official UR10e_CFG ratios
                r"shoulder_pan_joint": 72.6636085,      # stiffness(1320) / 18.2
                r"shoulder_lift_joint": 72.6636085,     # Same as shoulder_pan
                r"elbow_joint": 34.64101615,            # stiffness(600) / 17.3
                r"wrist_1_joint": 27.0,                 # stiffness(200) / 7.4 (slightly higher damping)
                r"wrist_2_joint": 27.0,                 # Increased damping for hand stability
                r"wrist_3_joint": 27.0,                 # Reduces oscillation with gripper

                # 손가락 damping 조정 (stiffness에 비례하여 조정)
                # ORIGINAL (너무 느림): r"rj_dg_[1-5]_[1-4]": 3.0,
                # 1st TRY: r"rj_dg_[1-5]_[1-4]": 0.5,
                # 2nd TRY (잘못된 방향): r"rj_dg_[1-5]_[1-4]": 0.8,
                # 참고: Kuka-Allegro는 0.1 사용
                r"rj_dg_[1-5]_[1-4]": 0.3,  # Kuka의 3배 (stiffness 5.0에 비례)
            },

            # === Friction (Static friction coefficient) ===
            # Friction at joint starts (resistance to initial motion)
            friction={
                # UR10e Arm
                # UR10e official config uses 0.0 friction
                # But for dexsuite manipulation, adding small friction improves stability
                r"shoulder_(pan|lift)_joint": 0.5,      # Light friction on large joints
                r"elbow_joint": 0.5,                    # Same for elbow
                r"wrist_[123]_joint": 0.1,              # Very light friction on wrist (precision)

                # 손가락은 민감해야 하므로 마찰을 매우 낮게 유지
                r"rj_dg_[1-5]_[1-4]": 0.01,
            },
        ),
    },

    # Soft joint limits (default)
    soft_joint_pos_limit_factor=1.0,
)
