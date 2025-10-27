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
USD_PATH = "/home/dyros/ros2_ws/src/hdr20_dg5f.usd"

HDR_DG5F_CFG = ArticulationCfg(
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

    # 초기 자세 (Kuka+Allegro 기본자세를 6축/20DoF에 맞춰 유사하게 조정)
    # init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(0.0, 0.0, 0.0),
    #     rot=(1.0, 0.0, 0.0, 0.0),
    #     joint_pos={
    #         # HDR20 (6 DoF)
    #         "j(1|2)": 0.0,
    #         "j3": 0.785398,     # ~45deg
    #         "j4": 1.570796,     # ~90deg
    #         "j(5|6)": -1.570796,

    #         # DG5F (20 DoF)  ➜ 모든 관절을 '허용범위 안의 음수'로 시작
    #         # DG5F (20 DoF) : 각 손가락 첫 관절은 0, 나머지 0.3 라디안 기본 굽힘 (x)
            
    #          # DG5F
    #         "rj_dg_1_1": -0.05, "rj_dg_2_1": -0.05, "rj_dg_3_1": -0.05, "rj_dg_4_1": -0.05, "rj_dg_5_1": 0.00,
    #         "rj_dg_1_2": -0.20, "rj_dg_2_2": 0.00,  "rj_dg_3_2": 0.00,  "rj_dg_4_2": 0.00,  "rj_dg_5_2": 0.00,
    #         "rj_dg_1_3": -0.20, "rj_dg_2_3": -0.20, "rj_dg_3_3": -0.20, "rj_dg_4_3": -0.20, "rj_dg_5_3": -0.20,
    #         "rj_dg_1_4": -0.20, "rj_dg_2_4": -0.20, "rj_dg_3_4": -0.20, "rj_dg_4_4": -0.20, "rj_dg_5_4": -0.20,
    #         # r"rj_dg_[1-4]_1": -0.05,
    #         # r"rj_dg_[1-5]_[2-4]": -0.20,
    #         # "rj_dg_5_1" : 0.00,
    #         # r"rj_dg_2_2" : 0.00,
    #         # r"rj_dg_3_2" : 0.00,
    #         # r"rj_dg_4_2" : 0.00,
            
    #     },
    # ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # HDR20 - 더 자연스러운 자세
            "j1": 0.0,          #-1.57,        # 0.0
            "j2": 1.047,         #0.5,        # 어깨 약간 올림
            "j3": 0.0,          #0.785398,      #0.5,        # 45° → 28°로 덜 굽힘
            "j4": 0.0,          #1.570796,      #1.2,        # 90° → 68°로 덜 굽힘
            "j5": 0.0,          #-1.570796,     #-1.2,       # -90° → -68°
            "j6": 0.0,
            
            # DG5F - 손가락 덜 오므림
            "rj_dg_1_1": 0.2,  "rj_dg_2_1": 0.0,  "rj_dg_3_1": 0.0,  "rj_dg_4_1": 0.0,  "rj_dg_5_1": 0.2,
            "rj_dg_1_2": -0.3, "rj_dg_2_2": 0.2,  "rj_dg_3_2": 0.2,  "rj_dg_4_2": 0.2,  "rj_dg_5_2": 0.2,
            "rj_dg_1_3": -0.2, "rj_dg_2_3": -0.2, "rj_dg_3_3": -0.2, "rj_dg_4_3": -0.2, "rj_dg_5_3": -0.2,
            "rj_dg_1_4": -0.2, "rj_dg_2_4": -0.2, "rj_dg_3_4": -0.2, "rj_dg_4_4": -0.2, "rj_dg_5_4": -0.2,
        },
    ),
    # init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(0.0, 0.0, 0.0),
    #     rot=(1.0, 0.0, 0.0, 0.0),
    #     joint_pos={
    #         # HDR20 arm
    #         "j1": 0.0,
    #         "j2": 0.0,
    #         "j3": 0.785,      # 45도
    #         "j4": 1.571,      # 90도
    #         "j5": -1.571,     # -90도
    #         "j6": 0.0,
            
    #         # DG5F hand - Allegro처럼 더 오므림 (0.3 → 음수 방향)
    #         # 손가락 1 (특수: _2가 음수만 가능)
    #         "rj_dg_1_1": 0.3,     # 약간 벌림 (물체 감싸기 좋게)
    #         "rj_dg_1_2": -0.5,    # 많이 굽힘 (limit: -3.14 ~ 0.0)
    #         "rj_dg_1_3": -0.4,    # 더 굽힘
    #         "rj_dg_1_4": -0.4,
            
    #         # 손가락 2,3,4 (검지, 중지, 약지)
    #         "rj_dg_2_1": 0.0,     # 중앙
    #         "rj_dg_2_2": 0.5,     # 양수로 굽힘 (limit: 0.0 ~ 2.0)
    #         "rj_dg_2_3": -0.4,    # 음수로 굽힘
    #         "rj_dg_2_4": -0.4,
            
    #         "rj_dg_3_1": 0.0,
    #         "rj_dg_3_2": 0.5,
    #         "rj_dg_3_3": -0.4,
    #         "rj_dg_3_4": -0.4,
            
    #         "rj_dg_4_1": 0.0,
    #         "rj_dg_4_2": 0.5,
    #         "rj_dg_4_3": -0.4,
    #         "rj_dg_4_4": -0.4,
            
    #         # 손가락 5 (새끼)
    #         "rj_dg_5_1": 0.3,     # 약간 벌림
    #         "rj_dg_5_2": 0.3,     # 살짝 굽힘
    #         "rj_dg_5_3": -0.4,
    #         "rj_dg_5_4": -0.4,
    #     },
    # ),
    # 암+핸드 전부 implicit actuator로 제어 (Kuka+Allegro 세팅을 6축/20DoF에 맞게 스케일)
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
                r"rj_dg_[1-5]_[1-4]": 10.0, 
            },
            
            # 스티프니스 (매우 중요! 로봇 무게를 지탱하도록 대폭 상향)
            stiffness={
                # j1-j3: 매우 무거운 링크들, 높은 강성 필요
                r"j(1|2|3)": 15000.0, 
                # j4-j6: 손목 관절, 상대적으로 낮은 강성
                r"j4": 500.0,
                r"j5": 500.0,
                r"j6": 250.0,
                # 손가락 (기존 3.0은 너무 낮음, 30.0 정도로 상향)
                r"rj_dg_[1-5]_[1-4]": 30.0,
            },
            
            # 댐핑 (일반적으로 stiffness의 1/10 ~ 1/100, 진동을 잡는 역할)
            damping={
                r"j(1|2|3)": 1500.0,  # (stiffness / 10)
                r"j4": 50.0,
                r"j5": 50.0,
                r"j6": 25.0,
                r"rj_dg_[1-5]_[1-4]": 3.0,
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
    # actuators={
    #     "hdr20_dg5f_actuators": ImplicitActuatorCfg(
    #         joint_names_expr=[
    #             r"j(1|2|3|4|5|6)",
    #             r"rj_dg_[1-5]_[1-4]",
    #         ],
    #         # 토크/힘 제한
    #         effort_limit_sim={
    #             r"j(1|2|3)": 900.0,    # Base/Shoulder/Elbow: 높게
    #             r"j(4|5|6)": 50.0,     # Wrist: 실제 값에 가깝게
    #             r"rj_dg_[1-5]_[1-4]": 7.5,
    #         },
    #         # effort_limit_sim={
    #         #     r"j(1|2|3|4|5|6)": 300.0,
    #         #     r"rj_dg_[1-5]_[1-4]": 6.0,
    #         # },
            
    #         # 스티프니스 (상위 4축 강, 말단 2축 약 — iiwa 규칙을 6축에 맞춰 적용)
    #         stiffness={
    #             r"j(1|2|3|4)": 300.0,
    #             r"j5": 100.0,
    #             r"j6": 50.0,
    #             r"rj_dg_[1-5]_[1-4]": 3.0,
    #         },
    #         # 댐핑
    #         damping={
    #             r"j(1|2|3|4)": 60.0,
    #             r"j5": 30.0,
    #             r"j6": 20.0,
    #             r"rj_dg_[1-5]_[1-4]": 0.2,
    #         },
    #         # 마찰(정마찰 계수)
    #         friction={
    #             r"j(1|2|3|4|5|6)": 1.0,
    #             r"rj_dg_[1-5]_[1-4]": 0.01,
    #         },
    #     ),
    # },

    # 소프트 조인트 한계 (기본값 유지)
    soft_joint_pos_limit_factor=0.95,            #1.0,
)