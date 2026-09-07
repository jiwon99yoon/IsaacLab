# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the UR10e arm with ATI F/T Sensor and Inspire Hand Left.

The following configurations are available:

* :obj:`UR10E_INSPIRE_LEFT_CFG`: UR10e + ATI Sensor + Inspire Hand Left (6-DOF under-actuated) with implicit actuator model.

Reference:
* UR10e: Universal Robots UR10e (12.5kg payload, 1300mm reach)
* ATI Axia80-M50: 6-axis Force/Torque sensor
* Inspire Hand RH56F1_L: 6-DOF under-actuated robotic hand with mimic joints

Configuration Notes:
* UR10e joint names: shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint
* Inspire Left actuated joints: 6 DOF (thumb: 2, index/middle/ring/little: 1 each)
* Total DOF: 6 (arm) + 6 (hand) = 12 actuated DOF

Key Tuning:
- UR10e actuator parameters based on official UR10e specs
- Inspire hand parameters based on UH035_INSPIRE_LEFT_CFG
- Init pose optimized for dexsuite manipulation tasks (workspace center, pre-grasp position)
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

import os
from pathlib import Path


# USD 파일 상대 경로
# 현재 실행 중인 .py 파일의 경로
CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))

# IsaacLab 경로
ISAACLAB_ROOT = CURRENT_DIR.parents[3]

# USD 파일의 상대 경로 (IsaacLab 루트부터 시작)
USD_RELATIVE_PATH = "hyundai/1_factory_usd_file/ur10e_usd/ur10e_ati_inspire_left.usd"

# 최종 USD_PATH: 절대 경로를 다시 구성
USD_PATH = ISAACLAB_ROOT.joinpath(USD_RELATIVE_PATH).as_posix()

# USD 파일 절대 경로
# USD file path (UR10e + ATI Sensor + Inspire Hand Left)
# USD_PATH = "/home/dyros/IsaacLab/hyundai/1_factory_usd_file/ur10e_usd/ur10e_ati_inspire_left.usd"


UR10E_INSPIRE_LEFT_CFG = ArticulationCfg(
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
            enabled_self_collisions=True,
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
            "shoulder_pan_joint": 0.0,          # Centered (0° - facing forward)
            "shoulder_lift_joint": -2.0944,     # RAISED from -1.2 to -0.8 (lift arm higher) -> changed to 120°
                                                # → Hand higher than UR10e default for better object approach
            "elbow_joint": 2.0944,              # More bent (120° vs 90°)
                                                # → Better manipulability and workspace coverage
            "wrist_1_joint": -2.0944,           # -120° (pitch down)
            "wrist_2_joint": 0.0,               # 0° (CHANGED: neutral roll)
                                                # → Hand palm faces down toward table
            "wrist_3_joint": -0.7854,           # 45° (neutral yaw)

            # === Hand (Inspire Left - 6 actuated DOF) ===
            # Pre-grasp pose: fingers slightly open and curved
            # Same as UH035_INSPIRE_LEFT_CFG for consistency

            # Thumb (2 actuated joints)
            "left_thumb_1_joint": 1.0,      # Thumb abduction: [0, 2.0944] → moderately open
            "left_thumb_2_joint": 0.0,      # Thumb flexion: [0, 0.4746] → slightly bent

            # Index (1 actuated joint, index_2 is mimic)
            "left_index_1_joint": 0.4,      # [0, 1.5286] → moderately bent

            # Middle (1 actuated joint, middle_2 is mimic)
            "left_middle_1_joint": 0.3,     # [0, 1.5286] → slightly bent

            # Ring (1 actuated joint, ring_2 is mimic)
            "left_ring_1_joint": 0.3,       # [0, 1.5286] → slightly bent

            # Little (1 actuated joint, little_2 is mimic)
            "left_little_1_joint": 0.3,     # [0, 1.5286] → slightly bent
        },
    ),

    # Arm + Hand: all controlled with implicit actuators
    actuators={
        "ur10e_inspire_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                # Arm joints (6-DOF)
                r"shoulder_pan_joint",
                r"shoulder_lift_joint",
                r"elbow_joint",
                r"wrist_[123]_joint",
                # Hand joints (6 actuated DOF)
                r"left_thumb_(1|2)_joint",
                r"left_(index|middle|ring|little)_1_joint",
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

                # Inspire Hand (6 actuated DOF)
                # Same as UH035_INSPIRE_LEFT_CFG (effort=1.0 from URDF)
                # Under-actuated design requires low effort for compliance
                r"left_thumb_(1|2)_joint": 1.0,
                r"left_(index|middle|ring|little)_1_joint": 1.0,
            },

            # === Stiffness (Position control proportional gain) ===
            # UR10e stiffness based on official config:
            # - Shoulder: 1320.0
            # - Elbow: 600.0
            # - Wrist: 216.0
            #
            # However, with Inspire hand attached (~1kg), wrist stiffness needs adjustment
            # to avoid oscillation. Using slightly lower values than stock UR10e.
            stiffness={
                # UR10e Arm (12.5kg payload + Inspire hand ~1kg)
                # Based on official UR10e_CFG, slightly reduced for hand attachment
                r"shoulder_pan_joint": 1320.0,      # Shoulder joints (same as UR10e_CFG)
                r"shoulder_lift_joint": 1320.0,     # High stiffness for payload support
                r"elbow_joint": 600.0,              # Elbow (same as UR10e_CFG)
                r"wrist_1_joint": 200.0,            # Wrist (reduced from 216 due to hand weight)
                r"wrist_2_joint": 200.0,            # Lower stiffness reduces oscillation
                r"wrist_3_joint": 200.0,            # with under-actuated gripper

                # Inspire Hand (6-DOF under-actuated)
                # Same as UH035_INSPIRE_LEFT_CFG (stiffness=2.5)
                # Lower than Kuka-Allegro (3.0) and DG5F (4.0)
                # Reason: Under-actuated design requires compliance for adaptive grasping
                r"left_thumb_(1|2)_joint": 2.5,             # Thumb: moderate stiffness
                r"left_(index|middle|ring|little)_1_joint": 2.5,  # Fingers: same stiffness
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

                # Inspire Hand
                # Same as UH035_INSPIRE_LEFT_CFG (damping=0.25)
                # stiffness(2.5) / 10 = 0.25
                r"left_thumb_(1|2)_joint": 0.25,            # stiffness / 10
                r"left_(index|middle|ring|little)_1_joint": 0.25,
                # Reference: Kuka-Allegro uses 0.1, DG5F uses 0.3
                #            Inspire uses middle value (0.25)
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

                # Inspire Hand
                # Same as UH035_INSPIRE_LEFT_CFG (friction=0.01)
                # Very low friction for sensitive finger control
                r"left_thumb_(1|2)_joint": 0.01,
                r"left_(index|middle|ring|little)_1_joint": 0.01,
            },
        ),
    },

    # Soft joint limits (default)
    soft_joint_pos_limit_factor=1.0,
)
