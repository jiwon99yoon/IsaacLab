# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the UH035 arm with Inspire Hand Right.

The following configurations are available:

* :obj:`HDR35_20_RH56F1_R_CFG`: UH035 (35kg payload) + Inspire Hand Right (6-DOF under-actuated) with implicit actuator model.

Reference:
* UH035: 35kg payload industrial robotic arm (6-DOF)
* Inspire Hand RH56F1_L: 6-DOF under-actuated robotic hand with mimic joints

URDF Analysis (from USD export):
* Link masses: base~link4=1kg, link5=0.5kg, link6=0.0001kg (simplified values)
* Joint effort limits: 10000000 N⋅m (no limits, requires manual tuning)
* Mass values are approximations from USD export, not actual robot specs

Key Differences from HDR20-DG5F:
- UH035 vs HDR20-17: Heavier payload (35kg vs 20kg) but similar link mass in URDF
- Inspire vs DG5F: Under-actuated hand (6 actuated DOF vs 20 DOF) → Lower stiffness for compliance
"""

import math
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

# USD 파일 경로 (UH035 + Inspire Hand Right)
USD_PATH = "/home/dyros/IsaacLab/src/nvidia-curobo/src/curobo/content/assets/robot/hdr35_20_rh56f1_r_description/hdr35_20_rh56f1_r_sensor_temp_flattened.usd"

#USD_PATH = "/home/dyros/IsaacLab/hyundai/1_factory_usd_file/UH035_usd/UH035_gripper_L_tip_with_ati.usd"

HDR35_20_RH56F1_R_CFG = ArticulationCfg(
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
            # URDF Reference: hdr35_20_rh56f1_r_sensor.urdf (lines 153-194)
            #
            # Joint Limits (URDF):
            #   j1: [-3.141, 3.141] rad (±180°)    | effort: 3078.7 N⋅m | velocity: 3.141 rad/s
            #   j2: [-2.3558, 1.5702] rad (-135° to 90°) | effort: 3169.7 N⋅m | velocity: 3.141 rad/s
            #   j3: [-1.396, 3.141] rad (-80° to 180°)   | effort: 1507.0 N⋅m | velocity: 3.316 rad/s
            #   j4: [-6.283, 6.283] rad (±360°)    | effort: 259.0 N⋅m  | velocity: 5.410 rad/s
            #   j5: [-2.181, 2.181] rad (±125°)    | effort: 240.4 N⋅m  | velocity: 5.410 rad/s
            #   j6: [-6.283, 6.283] rad (±360°)    | effort: 215.6 N⋅m  | velocity: 7.330 rad/s
            # ============================================================================
            # 초기 위치 - 학습 전 (로봇팔 초기)
            # "j1": 0.0,          # Base rotation (0°) - Range: [-180°, +180°] ✓
            # "j2": 0.7854,       # Shoulder pitch (45°) - Range: [-135°, +90°] ✓ (USER MODIFIED)
            # "j3": -0.7854,      # Shoulder roll (-45°) - Range: [-80°, +180°] ✓
            # "j4": 0.0,          # Elbow (0°) - Range: [-360°, +360°] ✓
            # "j5": -0.7854,      # Wrist pitch (-45°) - Range: [-125°, +125°] ✓
            # "j6": 0.0,          # Wrist roll (0°) - Range: [-360°, +360°] ✓

            # 학습위한 init pos 설정
            "j1": math.radians(109),   # 1.902 rad      #rh56f1_r모델과 음수
            # 수정 후 <- hdr35_20_rh56f1_r_sensor_temp.usd기준으로 수정 
            "j2": math.radians(16),
            "j3": math.radians(-46),
            "j4": math.radians(72),   # 1.257 rad       #rh56f1_r모델과 음수
            "j5": math.radians(-10),   # 1.257 rad
            "j6": math.radians(5),    # 0.942 rad       #rh56f1_r모델과 음수

            # 수정 전
            # "j2": math.radians(74),    # 1.292 rad
            #"j3": math.radians(-46),   # -0.803 rad
            #"j4": math.radians(-72),   # -1.257 rad
            # "j5": math.radians(76),    # 1.326 rad
            # "j6": math.radians(54),    # 0.942 rad

            # ============================================================================
            # HAND JOINTS - RH56F1_R Inspire Hand Right (6 Actuated DOF)
            # ============================================================================
            # URDF Reference: hdr35_20_rh56f1_r_sensor.urdf (lines 1118-1392)
            #
            # Actuated Joints (6 DOF controlled by actuators):
            #   right_thumb_1_joint:  [0, 2.0944] rad (0° to 120°)   | effort: 10 N⋅m | velocity: 2 rad/s
            #   right_thumb_2_joint:  [0, 0.4746] rad (0° to 27.18°) | effort: 10 N⋅m | velocity: 2 rad/s
            #   right_index_1_joint:  [0, 1.5286] rad (0° to 87.56°) | effort: 10 N⋅m | velocity: 2 rad/s
            #   right_middle_1_joint: [0, 1.5286] rad (0° to 87.56°) | effort: 10 N⋅m | velocity: 2 rad/s
            #   right_ring_1_joint:   [0, 1.5286] rad (0° to 87.56°) | effort: 10 N⋅m | velocity: 2 rad/s
            #   right_little_1_joint: [0, 1.5286] rad (0° to 87.56°) | effort: 10 N⋅m | velocity: 2 rad/s
            #
            # Mimic Joints (자동으로 actuated joint를 따라가는 coupled links):
            #   right_thumb_3_joint:  mimic of thumb_2 × 1.1425  (under-actuated coupling)
            #   right_thumb_4_joint:  mimic of thumb_3 × 0.7508  (under-actuated coupling)
            #   right_index_2_joint:  mimic of index_1 × 1.1169  (under-actuated coupling)
            #   right_middle_2_joint: mimic of middle_1 × 1.1169 (under-actuated coupling)
            #   right_ring_2_joint:   mimic of ring_1 × 1.1169   (under-actuated coupling)
            #   right_little_2_joint: mimic of little_1 × 1.1169 (under-actuated coupling)
            #
            # Initial Pose Strategy: Pre-grasp configuration
            #   - Fingers slightly curved and spread
            #   - Ready to adapt to object shape (under-actuated design)
            # ============================================================================

            # THUMB (엄지) - 2 Actuated Joints
            "right_thumb_1_joint": 0.8,      # Thumb abduction (0.8 rad ≈ 45.8°)
                                             # Range: [0, 2.0944] rad (0° to 120°) ✓
                                             # Purpose: Spread thumb away from palm for opposition grasp

            "right_thumb_2_joint": 0.2,      # Thumb flexion (0.2 rad ≈ 11.5°)
                                             # Range: [0, 0.4746] rad (0° to 27.18°) ✓
                                             # Purpose: Slightly bend thumb tip
                                             # Note: thumb_3 and thumb_4 follow automatically via mimic

            # FOUR FINGERS (검지/중지/약지/소지) - 4 Actuated Joints (1 per finger)
            "right_index_1_joint": 0.4,      # Index flexion (0.4 rad ≈ 22.9°)
                                             # Range: [0, 1.5286] rad (0° to 87.56°) ✓
                                             # Note: index_2 follows via mimic (× 1.1169)

            "right_middle_1_joint": 0.3,     # Middle flexion (0.3 rad ≈ 17.2°)
                                             # Range: [0, 1.5286] rad (0° to 87.56°) ✓
                                             # Note: middle_2 follows via mimic (× 1.1169)

            "right_ring_1_joint": 0.3,       # Ring flexion (0.3 rad ≈ 17.2°)
                                             # Range: [0, 1.5286] rad (0° to 87.56°) ✓
                                             # Note: ring_2 follows via mimic (× 1.1169)

            "right_little_1_joint": 0.3,     # Little flexion (0.3 rad ≈ 17.2°)
                                             # Range: [0, 1.5286] rad (0° to 87.56°) ✓
                                             # Note: little_2 follows via mimic (× 1.1169)
            # ============================================================================
        },
    ),

    # 암+핸드 전부 implicit actuator로 제어
    actuators={
        "uh035_inspire_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                # Arm joints (6-DOF)
                r"j(1|2|3|4|5|6)",
                # Hand joints (6 actuated DOF)
                r"right_thumb_(1|2)_joint",
                r"right_(index|middle|ring|little)_1_joint",
            ],

            # ========================================================================
            # EFFORT LIMITS (Torque/Force Limits for PhysX Simulation)
            # ========================================================================
            # These values override URDF effort limits for simulation stability
            # URDF effort limits (from hdr35_20_rh56f1_r_sensor.urdf):
            #   ARM:  j1=3078.7, j2=3169.7, j3=1507.0, j4=259.0, j5=240.4, j6=215.6 N⋅m
            #   HAND: all=10.0 N⋅m
            #
            # Current settings use LOWER values for safety and simulation stability
            # ========================================================================
            effort_limit_sim={
                # --- ARM JOINTS (HDR35_20) ---
                # Using conservative values (lower than URDF) for RL safety
                # URDF values are manufacturer specs for max continuous torque
                r"j1": 400.0,       # Base rotation (URDF: 3078.7 N⋅m)
                                    # Conservative limit for base joint with large moment arm

                r"j2": 400.0,       # Shoulder pitch (URDF: 3169.7 N⋅m)
                                    # Supports heavy payload, conservative for RL

                r"j3": 350.0,       # Shoulder roll (URDF: 1507.0 N⋅m)
                                    # Lower than j1/j2, still conservative

                r"j4": 100.0,       # Elbow (URDF: 259.0 N⋅m)
                                    # Smaller moment arm, lower torque needed

                r"j5": 100.0,       # Wrist pitch (URDF: 240.4 N⋅m)
                                    # Wrist joint, conservative limit

                r"j6": 50.0,        # Wrist roll (URDF: 215.6 N⋅m)
                                    # Smallest torque requirement, most conservative

                # --- HAND JOINTS (RH56F1_R Inspire Hand) ---
                # Using very low values (10x lower than URDF) for under-actuated hand
                # URDF specifies 10.0 N⋅m for all hand joints
                # Lower values prevent damage to compliant structures
                r"right_thumb_(1|2)_joint": 1.0,                        # URDF: 10.0 N⋅m
                r"right_(index|middle|ring|little)_1_joint": 1.0,       # URDF: 10.0 N⋅m
                # Rationale: Under-actuated design requires gentle actuation
                #            to allow adaptive grasping without breaking objects
            },

            # ========================================================================
            # STIFFNESS (Position Control P-Gain)
            # ========================================================================
            # Stiffness = Proportional gain in implicit PD control
            # Higher values → Stronger position tracking, faster response
            # Lower values → More compliant, adaptive behavior
            #
            # Tuning strategy:
            # - Arm: High stiffness needed to support 35kg payload
            # - Hand: Low stiffness for compliant, adaptive grasping (under-actuated)
            # ========================================================================
            stiffness={
                # --- ARM JOINTS (HDR35_20 - 35kg payload) ---
                # Reference: HDR20-17 uses 20000 for j1-j3 (20kg payload)
                # Increased to 30000 for heavier 35kg payload
                r"j(1|2|3)": 30000.0,   # Base/Shoulder joints (j1, j2, j3)
                                        # Need high stiffness to support payload
                                        # HDR20: 20000 (20kg) → HDR35_20: 30000 (35kg)

                r"j4": 3000.0,          # Elbow (j4)
                                        # Smaller moment arm, lower stiffness
                                        # HDR20: 2000 → HDR35_20: 3000 (scaled up)

                r"j5": 3000.0,          # Wrist pitch (j5)
                                        # Similar to elbow
                                        # HDR20: 2000 → HDR35_20: 3000

                r"j6": 1500.0,          # Wrist roll (j6)
                                        # Lightest joint, lowest stiffness
                                        # HDR20: 1000 → HDR35_20: 1500 (scaled up)

                # --- HAND JOINTS (RH56F1_R Inspire Hand - under-actuated) ---
                # Much lower than arm for compliant grasping
                # Reference comparison:
                #   - Kuka-Allegro (20-DOF): 3.0
                #   - DG5F (20-DOF): 4.0
                #   - Inspire (6-DOF under-actuated): 2.5 (lower for compliance)
                r"right_thumb_(1|2)_joint": 2.5,                        # Thumb joints
                r"right_(index|middle|ring|little)_1_joint": 2.5,       # Four fingers
                # Lower stiffness allows fingers to adapt to object shape
            },

            # ========================================================================
            # DAMPING (Velocity Control D-Gain)
            # ========================================================================
            # Damping = Derivative gain in implicit PD control
            # Purpose: Suppress oscillations and vibrations
            # Typical ratio: damping = stiffness / 10 ~ stiffness / 15
            # ========================================================================
            damping={
                # --- ARM JOINTS (HDR35_20) ---
                r"j(1|2|3)": 2000.0,    # Base/Shoulder damping
                                        # stiffness(30000) / 15 = 2000
                                        # HDR20: 1500 (stiffness 20000 / 13.3)

                r"j4": 300.0,           # Elbow damping
                                        # stiffness(3000) / 10 = 300
                                        # HDR20: 200 (stiffness 2000 / 10)

                r"j5": 300.0,           # Wrist pitch damping
                                        # stiffness(3000) / 10 = 300
                                        # HDR20: 200

                r"j6": 150.0,           # Wrist roll damping
                                        # stiffness(1500) / 10 = 150
                                        # HDR20: 100 (stiffness 1000 / 10)

                # --- HAND JOINTS (RH56F1_R Inspire Hand) ---
                r"right_thumb_(1|2)_joint": 0.25,                       # Thumb damping
                                        # stiffness(2.5) / 10 = 0.25
                                        # Kuka-Allegro: 0.1, DG5F: 0.3

                r"right_(index|middle|ring|little)_1_joint": 0.25,      # Finger damping
                                        # stiffness(2.5) / 10 = 0.25
                                        # Medium value for stable yet compliant grasping
            },

            # ========================================================================
            # FRICTION (Static/Coulomb Friction)
            # ========================================================================
            # Friction coefficient for joint resistance at rest
            # Higher values → More realistic joint behavior, but slower response
            # Lower values → Faster motion, less realistic
            # ========================================================================
            friction={
                # --- ARM JOINTS (HDR35_20) ---
                # Large joints have higher friction due to weight and gearboxes
                r"j(1|2|3)": 5.0,       # Base/Shoulder joints
                                        # Heavy payload creates significant friction

                r"j(4|5|6)": 0.5,       # Wrist joints
                                        # Lighter joints, lower friction

                # --- HAND JOINTS (RH56F1_R Inspire Hand) ---
                # Very low friction for sensitive, adaptive grasping
                r"right_thumb_(1|2)_joint": 0.01,                       # Thumb friction
                r"right_(index|middle|ring|little)_1_joint": 0.01,      # Finger friction
                # Minimal friction allows smooth, compliant finger motion
            },
        ),
    },

    # 소프트 조인트 한계 (기본값 유지)
    soft_joint_pos_limit_factor=1.0,
)
