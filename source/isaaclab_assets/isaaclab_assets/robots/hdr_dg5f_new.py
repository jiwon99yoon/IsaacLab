# /home/dyros/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/hdr_dg5f.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Kuka-lbr-iiwa arm robots and Allegro Hand.

The following configurations are available:

* :obj:`HDR_DG5F_CFG`: Kuka Allegro with implicit actuator model.

Reference:

# * https://www.kuka.com/en-us/products/robotics-systems/industrial-robots/lbr-iiwa
# * https://www.wonikrobotics.com/robot-hand

same as kuka / wonikrobotics-robot-hand

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
 # from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

# 로컬 USD 경로 (Nucleus가 아니라 로컬 파일을 사용)
USD_PATH = "/home/dyros/ros2_ws/src/hdr20_dg5f_new.usd"

HDR_DG5F_CFG_NEW = ArticulationCfg(
    # prim_path는 환경 쪽에서 replace()로 주입할 예정
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        #activate_contact_sensors=False,
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
            # === 팔 (Arm) ===
            # Kuka 자세를 유지 (모두 범위 내)
            "j1": 0.0,
            "j2": 2.0933,     # 120°
            "j3": -0.7854,    # 45°       1.5708,     # 90°
            "j4": 0.0,
            "j5": -0.7854,    # -90°
            "j6": 0.0,
            # === 손 (Hand) ===
            # 1. 검지, 중지, 약지, 새끼 (Fingers 2-5) - Allegro pre-grasp 유지 (모두 범위 내)
            r"rj_dg_[2-5]_1": 0.0,    # 벌림 0
            r"rj_dg_[2-5]_2": 0.3,    # 살짝 굽힘 1
            r"rj_dg_[2-5]_3": 0.3,    # 살짝 굽힘 2
            r"rj_dg_[2-5]_4": 0.3,    # 살짝 굽힘 
            # 2. 엄지 (Finger 1) - Allegro opposition 자세를 DG5F 범위 내에서 좀 더 적극적으로 표현
            "rj_dg_1_1": 0.85,       # 최대 벌림에 가깝게 (Allegro 1.5 의도) / 범위 [-0.38, 0.89] 내
            "rj_dg_1_2": -1.0,       # 회전을 조금 더 줌 (Allegro 0.6 의도) / 범위 [-3.14, 0.0] 내
            "rj_dg_1_3": 0.3,        # Allegro 0.3과 유사하게 / 범위 [-1.57, 1.57] 내
            "rj_dg_1_4": 0.6         # Allegro 0.6과 유사하게 / 범위 [-1.57, 1.57] 내
        },
    ),

    # 암+핸드 전부 implicit actuator로 제어
    actuators={
        "hdr20_dg5f_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                r"j(1|2|3|4|5|6)",
                r"rj_dg_[1-5]_[1-4]",
            ],
            # 토크/힘 제한 (URDF의 <limit effort> 값을 따름)
            effort_limit_sim={
                r"j1": 962.9,
                r"j2": 1010.9,
                r"j3": 934.1,
                r"j4": 44.9,
                r"j5": 46.9,
                r"j6": 25.6,
                # 손가락은 적절한 값 (기존 7.5 유지 또는 약간 상향)
                r"rj_dg_[1-5]_[1-4]": 7.5,  
            },
            
            # 스티프니스 (매우 중요! 로봇 무게를 지탱하도록 대폭 상향)
            stiffness={
                # j1-j3: 매우 무거운 링크들, 높은 강성 필요
                r"j(1|2|3)": 20000.0, #15000.0, 
                # j4-j6: 손목 관절, 상대적으로 낮은 강성
                r"j4": 2000.0, #500.0,
                r"j5": 2000.0, #500.0,
                r"j6": 1000.0, #250.0,
                # 손가락 (기존 3.0은 너무 낮음, 30.0 정도로 상향)
                r"rj_dg_[1-5]_[1-4]": 30.0, #3.0,
            },
            
            # 댐핑 (일반적으로 stiffness의 1/10 ~ 1/100, 진동을 잡는 역할)
            damping={
                r"j(1|2|3)": 1500.0,  # (stiffness / 10)
                r"j4": 200.0, #50.0,
                r"j5": 200.0, #50.0,
                r"j6": 100.0, #25.0,
                r"rj_dg_[1-5]_[1-4]": 3.0, #0.1, 
            },
            
            # 마찰 (정마찰 계수, 움직임 시작 시 저항)
            friction={
                # 큰 관절들은 마찰을 좀 더 줌
                r"j(1|2|3)": 5.0,
                r"j(4|5|6)": 0.5,
                # 손가락은 민감해야 하므로 마찰을 매우 낮게 유지
                r"rj_dg_[1-5]_[1-4]": 0.01,
            },
        ),
    },

    # 소프트 조인트 한계 (기본값 유지)
    soft_joint_pos_limit_factor=1.0, #0.95,            #1.0,
)