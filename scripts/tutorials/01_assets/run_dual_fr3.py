# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn the dual FR3 (fr3_husky) robot and command it.

동작 데모:
- 양팔 joint1/joint3/joint5에 사인파 스윙 (ready pose 기준 오프셋)
- 그리퍼는 주기적으로 open/close 토글

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_dual_fr3.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with the dual FR3 robot.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from isaaclab_assets import DUAL_FR3_CFG  # isort:skip


def design_scene() -> dict:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Articulation
    robot_cfg = DUAL_FR3_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    robot = Articulation(cfg=robot_cfg)

    return {"robot": robot}


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    """Runs the simulation loop."""
    robot = entities["robot"]

    # 조인트 인덱스 조회 (양팔 스윙용 / 그리퍼용)
    swing_joint_ids, swing_joint_names = robot.find_joints([".*_fr3_joint1", ".*_fr3_joint3", ".*_fr3_joint5"])
    finger_joint_ids, finger_joint_names = robot.find_joints([".*_fr3_finger_joint.*"])
    print(f"[INFO] swing joints : {swing_joint_names}")
    print(f"[INFO] finger joints: {finger_joint_names}")

    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset (매 1000 스텝)
        if count % 1000 == 0:
            count = 0
            # root: 기본 스폰 포즈로 (fix_root_link=True라 base는 어차피 고정)
            root_state = robot.data.default_root_state.clone()
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # joint: init_state의 ready pose로
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            print("[INFO]: Resetting robot state...")

        # --- 명령 생성 ---
        # 기본은 ready pose 유지
        joint_pos_target = robot.data.default_joint_pos.clone()
        # 양팔 스윙: joint1/3/5에 사인파 오프셋 (±0.4 rad)
        phase = 2.0 * math.pi * count / 500.0
        joint_pos_target[:, swing_joint_ids] += 0.4 * math.sin(phase)
        # 그리퍼: 250 스텝마다 open(0.04) / close(0.0) 토글
        gripper_open = (count // 250) % 2 == 0
        joint_pos_target[:, finger_joint_ids] = 0.04 if gripper_open else 0.0

        # position target 인가 (implicit PD가 추종)
        robot.set_joint_position_target(joint_pos_target)
        robot.write_data_to_sim()

        # Perform step
        sim.step()
        count += 1
        robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 1.8], [0.0, 0.0, 0.6])
    # Design scene
    scene_entities = design_scene()
    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
