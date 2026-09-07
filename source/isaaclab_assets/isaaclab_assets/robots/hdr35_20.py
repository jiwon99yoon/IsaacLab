# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the HDR35_20 arm (without hand).

The following configurations are available:

* :obj:`HDR35_20_CFG`: HDR35_20 (35kg payload) arm only with implicit actuator model.

Reference:
* HDR35_20: 35kg payload industrial robotic arm (6-DOF) - Hyundai Robotics

URDF Analysis (from USD export):
* Arm joint limits same as hdr35_20_dg5f_l.py / hdr35_20_rh56f1_r_sensor.py

Joint Limits (URDF):
  j1: [-3.141, 3.141] rad (±180°)    | effort: 3078.7 N⋅m | velocity: 3.141 rad/s
  j2: [-2.3558, 1.5702] rad (-135° to 90°) | effort: 3169.7 N⋅m | velocity: 3.141 rad/s
  j3: [-1.396, 3.141] rad (-80° to 180°)   | effort: 1507.0 N⋅m | velocity: 3.316 rad/s
  j4: [-6.283, 6.283] rad (±360°)    | effort: 259.0 N⋅m  | velocity: 5.410 rad/s
  j5: [-2.181, 2.181] rad (±125°)    | effort: 240.4 N⋅m  | velocity: 5.410 rad/s
  j6: [-6.283, 6.283] rad (±360°)    | effort: 215.6 N⋅m  | velocity: 7.330 rad/s
"""

import math
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

# USD 파일 경로 (HDR35_20 arm only)
USD_PATH = "/home/dyros/IsaacLab/src/nvidia-curobo/src/curobo/content/assets/robot/hdr35_20_rh56f1_r_description/hdr35_20_temp.usd"


HDR35_20_CFG = ArticulationCfg(
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
            # Ready position for reaching target (-0.65, -1.0, 0.70)
            # j1: base rotation towards -Y direction (약 -90°)
            # j2, j3: shoulder configuration for reach
            # j4, j5, j6: wrist orientation
            # ============================================================================
            "j1": -1.57,   # -90° to face -Y direction
            "j2": -0.5,    # slight shoulder tilt
            "j3": 1.0,     # shoulder roll for reach
            "j4": 0.0,     # elbow
            "j5": -0.5,    # wrist pitch
            "j6": 0.0,     # wrist roll
        },
    ),

    # Arm implicit actuator
    actuators={
        "hdr35_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                # Arm joints (6-DOF) - HDR35_20
                r"j(1|2|3|4|5|6)",
            ],

            # ========================================================================
            # EFFORT LIMITS (Torque/Force Limits for PhysX Simulation)
            # ========================================================================
            effort_limit_sim={
                r"j1": 400.0,       # Base rotation (URDF: 3078.7 N⋅m)
                r"j2": 400.0,       # Shoulder pitch (URDF: 3169.7 N⋅m)
                r"j3": 350.0,       # Shoulder roll (URDF: 1507.0 N⋅m)
                r"j4": 100.0,       # Elbow (URDF: 259.0 N⋅m)
                r"j5": 100.0,       # Wrist pitch (URDF: 240.4 N⋅m)
                r"j6": 50.0,        # Wrist roll (URDF: 215.6 N⋅m)
            },

            # ========================================================================
            # STIFFNESS (Position Control P-Gain)
            # ========================================================================
            # HDR35_20 - 35kg payload industrial arm
            # ========================================================================
            stiffness={
                r"j(1|2|3)": 30000.0,   # Base/Shoulder joints
                r"j4": 3000.0,          # Elbow
                r"j5": 3000.0,          # Wrist pitch
                r"j6": 1500.0,          # Wrist roll
            },

            # ========================================================================
            # DAMPING (Velocity Control D-Gain)
            # ========================================================================
            damping={
                r"j(1|2|3)": 2000.0,    # Base/Shoulder damping (stiffness/15)
                r"j4": 300.0,           # Elbow damping (stiffness/10)
                r"j5": 300.0,           # Wrist pitch damping
                r"j6": 150.0,           # Wrist roll damping
            },

            # ========================================================================
            # FRICTION (Static/Coulomb Friction)
            # ========================================================================
            friction={
                r"j(1|2|3)": 5.0,       # Base/Shoulder joints (heavy payload)
                r"j(4|5|6)": 0.5,       # Wrist joints (lighter)
            },
        ),
    },

    # 소프트 조인트 한계 (기본값 유지)
    soft_joint_pos_limit_factor=1.0,
)
