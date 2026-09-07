# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the HDR35_20 arm with DG5F Hand Left.

The following configurations are available:

* :obj:`HDR35_20_DG5F_L_CFG`: HDR35_20 (35kg payload) + DG5F Hand Left (20-DOF fully-actuated) with implicit actuator model.

Reference:
* HDR35_20: 35kg payload industrial robotic arm (6-DOF) - Hyundai Robotics
* Tesollo Hand DG5F Left: 20-DOF fully-actuated robotic hand

URDF Analysis (from USD export):
* Arm joint limits same as hdr35_20_rh56f1_r_sensor
* Hand: DG5F Left uses lj_dg_[1-5]_[1-4] joint naming (20 DOF)

Key Differences from HDR35_20 + RH56F1_R:
- RH56F1_R (Inspire): 6-DOF under-actuated → Lower stiffness for compliance
- DG5F_L (Tesollo): 20-DOF fully-actuated → Higher stiffness for precise control

Combined from:
- ARM: hdr35_20_rh56f1_r_sensor.py (j1~j6 settings)
- HAND: uh035_ati_dg5f_L.py (lj_dg_[1-5]_[1-4] settings)
"""

import math
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

# USD 파일 경로 (HDR35_20 + DG5F Hand Left)
USD_PATH = "/home/dyros/IsaacLab/src/nvidia-curobo/src/curobo/content/assets/robot/hdr35_20_dg5f_l_description/hdr35_20_dg5f_l_temp_flattened.usd"


def deg2rad(degrees: float) -> float:
    """Convert degrees to radians."""
    return math.radians(degrees)


HDR35_20_DG5F_L_CFG = ArticulationCfg(
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
            # ============================================================================
            # ARM JOINTS - HDR35_20 (6-DOF)
            # ============================================================================
            # From: hdr35_20_rh56f1_r_sensor.py
            # Joint Limits (URDF):
            #   j1: [-3.141, 3.141] rad (±180°)    | effort: 3078.7 N⋅m | velocity: 3.141 rad/s
            #   j2: [-2.3558, 1.5702] rad (-135° to 90°) | effort: 3169.7 N⋅m | velocity: 3.141 rad/s
            #   j3: [-1.396, 3.141] rad (-80° to 180°)   | effort: 1507.0 N⋅m | velocity: 3.316 rad/s
            #   j4: [-6.283, 6.283] rad (±360°)    | effort: 259.0 N⋅m  | velocity: 5.410 rad/s
            #   j5: [-2.181, 2.181] rad (±125°)    | effort: 240.4 N⋅m  | velocity: 5.410 rad/s
            #   j6: [-6.283, 6.283] rad (±360°)    | effort: 215.6 N⋅m  | velocity: 7.330 rad/s
            # ============================================================================
            # 학습위한 init pos 설정 (사용자 설정값 유지)
            "j1": math.radians(-109),    #rh56f1_r모델과 음수
            "j2": math.radians(16),
            "j3": math.radians(-46),
            "j4": math.radians(-72),    #rh56f1_r모델과 음수
            "j5": math.radians(-10),
            "j6": math.radians(-5),      #rh56f1_r모델과 음수

            # ============================================================================
            # HAND JOINTS - DG5F Left (20 Actuated DOF)
            # ============================================================================
            # From: uh035_ati_dg5f_L.py
            # Joint naming: lj_dg_[finger]_[joint]
            #   - finger: 1=thumb, 2=index, 3=middle, 4=ring, 5=little
            #   - joint: 1-4 (4 joints per finger)
            # Total: 5 fingers × 4 joints = 20 DOF
            # ============================================================================

            # 1. 엄지 (Finger 1) - Opposition 자세 (물체를 향해 대치)
            "lj_dg_1_1": deg2rad(-10.0),    # 적당히 벌림
            "lj_dg_1_2": deg2rad(30.0),     # 회전 강화
            "lj_dg_1_3": deg2rad(-30.0),    # 살짝 굽힘
            "lj_dg_1_4": deg2rad(-30.0),    # 적당히 굽힘

            # 2. 검지/중지/약지 (Finger 2-4) - Pre-grasp 자세
            # Joint 1: 손가락 사이 벌림 (abduction)
            "lj_dg_2_1": deg2rad(10.0),     # 검지 벌림
            "lj_dg_3_1": deg2rad(0.0),      # 중지 (중앙)
            "lj_dg_4_1": deg2rad(-5.0),     # 약지 벌림

            # Joint 2-4: 굽힘 (flexion) - 공통 설정
            r"lj_dg_[2-4]_2": deg2rad(20.0),    # 첫 관절 적당히 굽힘
            r"lj_dg_[2-4]_3": deg2rad(20.0),    # 두번째 관절 적당히 굽힘
            r"lj_dg_[2-4]_4": deg2rad(30.0),    # 세번째 관절 적당히 굽힘

            # 3. 소지 (Finger 5) - 다른 손가락과 다른 configuration
            "lj_dg_5_1": deg2rad(-10.0),    # 조금만 벌림
            "lj_dg_5_2": deg2rad(-15.0),    # 회전 강화
            "lj_dg_5_3": deg2rad(30.0),     # 살짝 굽힘
            "lj_dg_5_4": deg2rad(30.0),     # 적당히 굽힘
        },
    ),

    # 암+핸드 전부 implicit actuator로 제어
    actuators={
        "hdr35_dg5f_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                # Arm joints (6-DOF) - HDR35_20
                r"j(1|2|3|4|5|6)",
                # Hand joints (20 actuated DOF) - DG5F Left
                r"lj_dg_[1-5]_[1-4]",
            ],

            # ========================================================================
            # EFFORT LIMITS (Torque/Force Limits for PhysX Simulation)
            # ========================================================================
            # ARM: From hdr35_20_rh56f1_r_sensor.py
            # HAND: From uh035_ati_dg5f_L.py
            # ========================================================================
            effort_limit_sim={
                # --- ARM JOINTS (HDR35_20) ---
                # From: hdr35_20_rh56f1_r_sensor.py
                r"j1": 400.0,       # Base rotation (URDF: 3078.7 N⋅m)
                r"j2": 400.0,       # Shoulder pitch (URDF: 3169.7 N⋅m)
                r"j3": 350.0,       # Shoulder roll (URDF: 1507.0 N⋅m)
                r"j4": 100.0,       # Elbow (URDF: 259.0 N⋅m)
                r"j5": 100.0,       # Wrist pitch (URDF: 240.4 N⋅m)
                r"j6": 50.0,        # Wrist roll (URDF: 215.6 N⋅m)

                # --- HAND JOINTS (DG5F Left) ---
                # From: uh035_ati_dg5f_L.py
                # 부드러운 grasping을 위해 낮은 값
                r"lj_dg_[1-5]_[1-4]": 5.0,
            },

            # ========================================================================
            # STIFFNESS (Position Control P-Gain)
            # ========================================================================
            # ARM: From hdr35_20_rh56f1_r_sensor.py (35kg payload)
            # HAND: From uh035_ati_dg5f_L.py (DG5F: 4.0, fully-actuated)
            # ========================================================================
            stiffness={
                # --- ARM JOINTS (HDR35_20 - 35kg payload) ---
                r"j(1|2|3)": 30000.0,   # Base/Shoulder joints
                r"j4": 3000.0,          # Elbow
                r"j5": 3000.0,          # Wrist pitch
                r"j6": 1500.0,          # Wrist roll

                # --- HAND JOINTS (DG5F Left - 20-DOF fully-actuated) ---
                # DG5F is fully-actuated (vs Inspire under-actuated)
                # Higher stiffness (4.0) compared to Inspire (2.5) for precise control
                # Reference: Kuka-Allegro: 3.0, DG5F: 4.0, Inspire: 2.5
                r"lj_dg_[1-5]_[1-4]": 4.0,
            },

            # ========================================================================
            # DAMPING (Velocity Control D-Gain)
            # ========================================================================
            # ARM: From hdr35_20_rh56f1_r_sensor.py
            # HAND: From uh035_ati_dg5f_L.py (DG5F: 0.3)
            # ========================================================================
            damping={
                # --- ARM JOINTS (HDR35_20) ---
                r"j(1|2|3)": 2000.0,    # Base/Shoulder damping (stiffness/15)
                r"j4": 300.0,           # Elbow damping (stiffness/10)
                r"j5": 300.0,           # Wrist pitch damping
                r"j6": 150.0,           # Wrist roll damping

                # --- HAND JOINTS (DG5F Left) ---
                # stiffness(4.0) / ~13 = 0.3
                r"lj_dg_[1-5]_[1-4]": 0.3,
            },

            # ========================================================================
            # FRICTION (Static/Coulomb Friction)
            # ========================================================================
            # ARM: From hdr35_20_rh56f1_r_sensor.py
            # HAND: From uh035_ati_dg5f_L.py (매우 낮은 마찰)
            # ========================================================================
            friction={
                # --- ARM JOINTS (HDR35_20) ---
                r"j(1|2|3)": 5.0,       # Base/Shoulder joints (heavy payload)
                r"j(4|5|6)": 0.5,       # Wrist joints (lighter)

                # --- HAND JOINTS (DG5F Left) ---
                # 손가락은 민감해야 하므로 마찰을 매우 낮게 유지
                r"lj_dg_[1-5]_[1-4]": 0.01,
            },
        ),
    },

    # 소프트 조인트 한계 (기본값 유지)
    soft_joint_pos_limit_factor=1.0,
)
