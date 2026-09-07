# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
OSC with FT Sensor logging for HDR35_20 robot (arm only, no hand).

Key differences from ftsensor_to_csv3.py (Franka):
- Robot: HDR35_20_CFG (6-DOF arm) instead of FRANKA_PANDA_HIGH_PD_CFG (7-DOF)
- End-effector: flange_link instead of panda_leftfinger
- Arm joints: j1~j6 (6-DOF)
- F/T sensor link: flange_link
- nullspace_control: "none" (6-DOF는 redundant하지 않음)

Similar to Franka:
- Arm joints controlled by OSC (stiffness/damping = 0)
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FT Sensor logging for HDR35_20.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import pandas as pd

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

# HDR35_20 robot config (6-DOF arm only, no hand)
from isaaclab_assets import HDR35_20_CFG


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a simple scene with a tilted wall."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Contact wall for testing F/T sensor
    # ftsensor_to_csv3.py와 동일한 방식: wall을 target 위치에 배치
    # Z축이 force control이므로 tilted wall에 Z방향으로 밀면 접촉
    tilted_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TiltedWall",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 1.5, 0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            activate_contact_sensors=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            # target (1.776, -0.024, 1.267) 위치에 wall 배치
            pos=(1.776 + 0.085, -0.024, 1.267),
            # 45도 tilt around Y-axis
            rot=(0.9238795325, 0.0, -0.3826834324, 0.0)
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/TiltedWall",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
    )

    # HDR35_20 robot (6-DOF arm only, no hand)
    robot = HDR35_20_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Init state: 모든 arm joints = 0 (cuRobo retract_config과 동일)
    robot.init_state.joint_pos = {
        "j1": 0.0,
        "j2": 0.0,
        "j3": 0.0,
        "j4": 0.0,
        "j5": 0.0,
        "j6": 0.0,
    }

    # Solver 설정을 Franka와 동일하게 (떨림 방지)
    robot.spawn.articulation_props.solver_position_iteration_count = 8
    robot.spawn.articulation_props.solver_velocity_iteration_count = 0

    # OSC를 위해 stiffness/damping을 0으로 설정
    robot.actuators["hdr35_actuators"].stiffness = {
        r"j(1|2|3)": 0.0,
        r"j4": 0.0,
        r"j5": 0.0,
        r"j6": 0.0,
    }
    robot.actuators["hdr35_actuators"].damping = {
        r"j(1|2|3)": 0.0,
        r"j4": 0.0,
        r"j5": 0.0,
        r"j6": 0.0,
    }
    robot.spawn.rigid_props.disable_gravity = True


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop with FT sensor logging."""

    # Extract scene entities
    robot = scene["robot"]
    contact_forces = scene["contact_forces"]

    # HDR35_20 specific settings (arm only, no hand)
    # End-effector: flange_link
    ee_frame_name = "flange_link"
    # Joint names: j1~j6 (6-DOF)
    arm_joint_names = ["j.*"]

    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]

    print(f"\n{'='*60}")
    print(f"[INFO] HDR35_20 Robot Configuration (arm only)")
    print(f"{'='*60}")
    print(f"  End-effector body: {ee_frame_name} (idx: {ee_frame_idx})")
    print(f"  Arm joint IDs: {arm_joint_ids}")
    print(f"  Number of arm joints: {len(arm_joint_ids)}")
    print(f"  All body names: {robot.body_names}")
    print(f"  All joint names: {robot.joint_names}")
    print(f"{'='*60}\n")

    # ===== FT Sensor Configuration =====
    # F/T 센서는 flange_link에서 측정 (end-effector)
    ft_sensor_link_idx = robot.body_names.index("flange_link")

    # 다른 link들도 비교를 위해 추가
    link_indices = {}
    for link_name in ["base_body_link", "lower_frame_link", "upper_frame_link",
                      "arm_link", "wrist_body_link", "wrist_holder_link", "flange_link"]:
        if link_name in robot.body_names:
            link_indices[link_name] = robot.body_names.index(link_name)

    print(f"[INFO] Link indices: {link_indices}")

    # Outlier thresholds
    FORCE_MAX = 500.0   # N (HDR35는 큰 로봇이라 더 큰 값)
    TORQUE_MAX = 50.0   # Nm

    # Create the OSC (6-DOF arm용 설정)
    # NOTE: HDR35_20은 6-DOF라서 redundant하지 않음 → nullspace_control=None
    # NOTE: 무거운 로봇은 더 높은 damping ratio 필요 (1.0 → 2.0)
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs", "wrench_abs"],
        impedance_mode="variable_kp",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_damping_ratio_task=2.0,  # 무거운 로봇용 높은 damping (1.0 → 2.0)
        contact_wrench_stiffness_task=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        motion_control_axes_task=[1, 1, 0, 1, 1, 1],
        contact_wrench_control_axes_task=[0, 0, 1, 0, 0, 0],
        nullspace_control="none",  # 6-DOF arm은 redundant하지 않아서 "none" (문자열!)
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define targets for the arm
    # ee_link: flange_link (HDR35_20 arm end-effector)
    # Init pos (joints=0): (1.348, -0.024, 1.605)
    # Target: 시뮬레이션에서 확인한 도달 가능한 위치
    # Orientation: Y축 기준 10도 회전 → quat(w,x,y,z) = (cos(5°), 0, sin(5°), 0) ≈ (0.9962, 0, 0.0872, 0)
    # Isaac Lab pose format: [x, y, z, qw, qx, qy, qz]
    ee_goal_pose_set_tilted_b = torch.tensor(
        [
            [1.776, -0.024, 1.267, 0.9962, 0.0, 0.0872, 0.0],  # 도달 가능 위치, Y축 10도 회전
            [1.776, 0.0, 1.267, 0.9962, 0.0, 0.0872, 0.0],     # 약간 다른 위치
            [1.7, -0.024, 1.3, 0.9962, 0.0, 0.0872, 0.0],      # 약간 다른 위치
        ],
        device=sim.device,
    )
    ee_goal_wrench_set_tilted_task = torch.tensor(
        [
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],  # Franka와 동일 (10N)
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        ],
        device=sim.device,
    )
    kp_set_task = torch.tensor(
        [
            [360.0, 360.0, 360.0, 360.0, 360.0, 360.0],  # Franka와 동일한 값
            [420.0, 420.0, 420.0, 420.0, 420.0, 420.0],
            [300.0, 300.0, 300.0, 300.0, 300.0, 300.0],
        ],
        device=sim.device,
    )
    ee_target_set = torch.cat([ee_goal_pose_set_tilted_b, ee_goal_wrench_set_tilted_task, kp_set_task], dim=-1)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    robot.update(dt=sim_dt)

    # Get the updated states
    (jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b,
     root_pose_w, ee_pose_w, ee_force_b, joint_pos, joint_vel) = \
        update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids, contact_forces)

    # Track the given target command
    current_goal_idx = 0
    command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    ee_target_pose_w = torch.zeros(scene.num_envs, 7, device=sim.device)

    # Set joint efforts to zero
    zero_joint_efforts = torch.zeros(scene.num_envs, robot.num_joints, device=sim.device)
    joint_efforts = torch.zeros(scene.num_envs, len(arm_joint_ids), device=sim.device)

    acc_estimator = AccelerationEstimator(robot, dt=sim_dt)

    count = 0
    log_data = []

    def clip_outlier(value, max_threshold):
        """Remove outliers beyond threshold"""
        if abs(value) > max_threshold:
            return 0.0
        return value

    try:
        # Simulation loop
        while simulation_app.is_running():
            # Reset every 250 steps
            if count % 250 == 0:
                # Reset joint state to default
                default_joint_pos = robot.data.default_joint_pos.clone()
                default_joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
                robot.set_joint_effort_target(zero_joint_efforts)
                robot.write_data_to_sim()
                robot.reset()

                acc_estimator.reset()

                # Reset contact sensor
                contact_forces.reset()

                # Reset target pose
                robot.update(sim_dt)
                _, _, _, ee_pose_b, _, _, _, _, _, _ = update_states(
                    sim, scene, robot, ee_frame_idx, arm_joint_ids, contact_forces
                )

                command, ee_target_pose_b, ee_target_pose_w, current_goal_idx = update_target(
                    sim, scene, osc, root_pose_w, ee_target_set, current_goal_idx
                )

                # Set the osc command
                osc.reset()
                command, task_frame_pose_b = convert_to_task_frame(
                    osc, command=command, ee_target_pose_b=ee_target_pose_b
                )
                osc.set_command(
                    command=command,
                    current_ee_pose_b=ee_pose_b,
                    current_task_frame_pose_b=task_frame_pose_b
                )

            else:
                # Get updated states
                (jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b,
                 root_pose_w, ee_pose_w, ee_force_b, joint_pos, joint_vel) = \
                    update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids, contact_forces)

                # Compute joint commands
                # NOTE: nullspace_joint_pos_target 제거 (6-DOF는 nullspace control 없음)
                joint_efforts = osc.compute(
                    jacobian_b=jacobian_b,
                    current_ee_pose_b=ee_pose_b,
                    current_ee_vel_b=ee_vel_b,
                    current_ee_force_b=ee_force_b,
                    mass_matrix=mass_matrix,
                    gravity=gravity,
                    current_joint_pos=joint_pos,
                    current_joint_vel=joint_vel,
                )
                joint_acc = acc_estimator.estimate()
                ext_torque, ext_torque_g, damping, armature, friction_coeff = estimate_external_torque(robot, joint_acc=joint_acc)

                # Log data every 5 steps
                if count % 5 == 0:
                    # ===== 1. Contact Force (X, Y, Z) =====
                    if contact_forces.data.net_forces_w.shape[1] > 0:
                        contact_force_raw = contact_forces.data.net_forces_w[0, 0]
                    else:
                        contact_force_raw = torch.zeros(3, device=sim.device)

                    contact_force_x = contact_force_raw[0].item()
                    contact_force_y = contact_force_raw[1].item()
                    contact_force_z = contact_force_raw[2].item()
                    contact_force_magnitude = torch.norm(contact_force_raw).item()

                    # ===== 2. FT Sensor 데이터 at ll_dg_mount =====
                    ft_sensor_forces = robot.root_physx_view.get_link_incoming_joint_force()
                    ft_sensor_data = ft_sensor_forces[0][ft_sensor_link_idx]  # [6] 벡터

                    # Force (X, Y, Z) - Raw at ll_dg_mount
                    fs_force_x_raw = ft_sensor_data[0].item()
                    fs_force_y_raw = ft_sensor_data[1].item()
                    fs_force_z_raw = ft_sensor_data[2].item()

                    # Torque (X, Y, Z) - Raw at ll_dg_mount
                    fs_torque_x_raw = ft_sensor_data[3].item()
                    fs_torque_y_raw = ft_sensor_data[4].item()
                    fs_torque_z_raw = ft_sensor_data[5].item()

                    # External torques for joints 2, 4, 6
                    joint2_ext_torque = ext_torque[0, 1].item() if ext_torque.shape[1] > 1 else 0.0
                    joint4_ext_torque = ext_torque[0, 3].item() if ext_torque.shape[1] > 3 else 0.0
                    joint6_ext_torque = ext_torque[0, 5].item() if ext_torque.shape[1] > 5 else 0.0

                    joint2_ext_torque_g = ext_torque_g[0, 1].item() if ext_torque_g.shape[1] > 1 else 0.0
                    joint4_ext_torque_g = ext_torque_g[0, 3].item() if ext_torque_g.shape[1] > 3 else 0.0
                    joint6_ext_torque_g = ext_torque_g[0, 5].item() if ext_torque_g.shape[1] > 5 else 0.0

                    # ===== 3. All link F/T data for comparison =====
                    all_link_forces = robot.root_physx_view.get_link_incoming_joint_force()
                    link_data = {}

                    for link_name, link_idx in link_indices.items():
                        sensor_data = all_link_forces[0][link_idx]
                        link_data[f'{link_name}_force_x'] = sensor_data[0].item()
                        link_data[f'{link_name}_force_y'] = sensor_data[1].item()
                        link_data[f'{link_name}_force_z'] = sensor_data[2].item()
                        link_data[f'{link_name}_torque_x'] = sensor_data[3].item()
                        link_data[f'{link_name}_torque_y'] = sensor_data[4].item()
                        link_data[f'{link_name}_torque_z'] = sensor_data[5].item()

                    # ===== 4. OSC Joint Torques =====
                    osc_joint_torques = joint_efforts[0]

                    # ===== 5. Data logging =====
                    current_log = {
                        "step": count,

                        # Joint external torques (j1~j6)
                        "joint2_commanded": robot.data.applied_torque[0, 1].item() if robot.data.applied_torque.shape[1] > 1 else 0.0,
                        "joint2_external": joint2_ext_torque,
                        "joint2_external_g": joint2_ext_torque_g,
                        "joint2_acc": robot.data.joint_acc[0, 1].item() if robot.data.joint_acc.shape[1] > 1 else 0.0,

                        "joint4_commanded": robot.data.applied_torque[0, 3].item() if robot.data.applied_torque.shape[1] > 3 else 0.0,
                        "joint4_external": joint4_ext_torque,
                        "joint4_external_g": joint4_ext_torque_g,
                        "joint4_acc": robot.data.joint_acc[0, 3].item() if robot.data.joint_acc.shape[1] > 3 else 0.0,

                        "joint6_commanded": robot.data.applied_torque[0, 5].item() if robot.data.applied_torque.shape[1] > 5 else 0.0,
                        "joint6_external": joint6_ext_torque,
                        "joint6_external_g": joint6_ext_torque_g,
                        "joint6_acc": robot.data.joint_acc[0, 5].item() if robot.data.joint_acc.shape[1] > 5 else 0.0,

                        # Contact Force
                        "contact_force_x": contact_force_x,
                        "contact_force_y": contact_force_y,
                        "contact_force_z": contact_force_z,
                        "contact_force_magnitude": contact_force_magnitude,

                        # FT Sensor at flange_link (primary measurement)
                        "flange_force_x": fs_force_x_raw,
                        "flange_force_y": fs_force_y_raw,
                        "flange_force_z": fs_force_z_raw,
                        "flange_torque_x": fs_torque_x_raw,
                        "flange_torque_y": fs_torque_y_raw,
                        "flange_torque_z": fs_torque_z_raw,
                    }

                    # Add all link F/T data
                    current_log.update(link_data)

                    # OSC torques (6 joints for HDR35_20)
                    for i in range(min(6, len(osc_joint_torques))):
                        current_log[f'osc_torque_j{i+1}'] = osc_joint_torques[i].item()

                    # Joint positions and velocities (6 joints)
                    for i in range(min(6, joint_pos.shape[1])):
                        current_log[f'joint_pos_j{i+1}'] = joint_pos[0][i].item()
                        current_log[f'joint_vel_j{i+1}'] = joint_vel[0][i].item()

                    # Filter outliers
                    if contact_force_magnitude < FORCE_MAX:
                        log_data.append(current_log)

                # Apply actions
                robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
                robot.write_data_to_sim()

            # Update marker positions
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(ee_target_pose_w[:, 0:3], ee_target_pose_w[:, 3:7])

            # Perform step
            sim.step(render=True)
            robot.update(sim_dt)
            scene.update(sim_dt)
            count += 1

    finally:
        print("\nSimulation finished. Saving log data...")
        if log_data:
            df = pd.DataFrame(log_data)
            df.to_csv("simulation_log_hdr35_ft.csv", index=False)
            print(f"Log data saved to simulation_log_hdr35_ft.csv ({len(log_data)} records)")

            # Print summary statistics
            print_summary(log_data)
        else:
            print("No log data to save.")


def update_states(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    ee_frame_idx: int,
    arm_joint_ids: list[int],
    contact_forces,
):
    """Update the robot states."""

    # Obtain dynamics related quantities from simulation
    ee_jacobi_idx = ee_frame_idx - 1
    jacobian_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
    gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]

    # Convert the Jacobian from world to root frame
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    # Compute current pose of the end-effector
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_w = robot.data.body_pos_w[:, ee_frame_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_frame_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=-1)
    ee_pose_w = torch.cat([ee_pos_w, ee_quat_w], dim=-1)
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    # Compute the current velocity of the end-effector
    ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]
    root_vel_w = robot.data.root_vel_w
    relative_vel_w = ee_vel_w - root_vel_w
    ee_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])
    ee_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    # Calculate the contact force
    ee_force_w = torch.zeros(scene.num_envs, 3, device=sim.device)
    sim_dt = sim.get_physics_dt()
    contact_forces.update(sim_dt)
    ee_force_w, _ = torch.max(torch.mean(contact_forces.data.net_forces_w_history, dim=1), dim=1)
    ee_force_b = ee_force_w

    # Get joint positions and velocities
    joint_pos = robot.data.joint_pos[:, arm_joint_ids]
    joint_vel = robot.data.joint_vel[:, arm_joint_ids]

    return (
        jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b,
        root_pose_w, ee_pose_w, ee_force_b, joint_pos, joint_vel,
    )


def update_target(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    osc: OperationalSpaceController,
    root_pose_w: torch.tensor,
    ee_target_set: torch.tensor,
    current_goal_idx: int,
):
    """Update the targets for the operational space controller."""

    command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    command[:] = ee_target_set[current_goal_idx]

    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            ee_target_pose_b[:] = command[:, :7]
        elif target_type == "wrench_abs":
            pass
        else:
            raise ValueError("Undefined target_type within update_target().")

    ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7],
        ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
    )
    ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)

    next_goal_idx = (current_goal_idx + 1) % len(ee_target_set)

    return command, ee_target_pose_b, ee_target_pose_w, next_goal_idx


def convert_to_task_frame(
    osc: OperationalSpaceController,
    command: torch.tensor,
    ee_target_pose_b: torch.tensor
):
    """Converts the target commands to the task frame."""

    command = command.clone()
    task_frame_pose_b = ee_target_pose_b.clone()

    cmd_idx = 0
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            command[:, :3], command[:, 3:7] = subtract_frame_transforms(
                task_frame_pose_b[:, :3], task_frame_pose_b[:, 3:],
                command[:, :3], command[:, 3:7]
            )
            cmd_idx += 7
        elif target_type == "wrench_abs":
            cmd_idx += 6
        else:
            raise ValueError("Undefined target_type within _convert_to_task_frame().")

    return command, task_frame_pose_b


def print_summary(log_data):
    """Print summary of logged data"""

    df = pd.DataFrame(log_data)

    print("\n" + "="*60)
    print("HDR35_20 F/T Sensor Data Summary")
    print("="*60)

    # Flange F/T summary
    print("\nFlange Link F/T Sensor:")
    print(f"  Force X - Mean: {df['flange_force_x'].mean():.3f} N, Std: {df['flange_force_x'].std():.3f} N")
    print(f"  Force Y - Mean: {df['flange_force_y'].mean():.3f} N, Std: {df['flange_force_y'].std():.3f} N")
    print(f"  Force Z - Mean: {df['flange_force_z'].mean():.3f} N, Std: {df['flange_force_z'].std():.3f} N")
    print(f"  Torque X - Mean: {df['flange_torque_x'].mean():.3f} Nm, Std: {df['flange_torque_x'].std():.3f} Nm")
    print(f"  Torque Y - Mean: {df['flange_torque_y'].mean():.3f} Nm, Std: {df['flange_torque_y'].std():.3f} Nm")
    print(f"  Torque Z - Mean: {df['flange_torque_z'].mean():.3f} Nm, Std: {df['flange_torque_z'].std():.3f} Nm")

    # Contact force summary
    print("\nContact Force:")
    print(f"  Magnitude - Mean: {df['contact_force_magnitude'].mean():.3f} N, Max: {df['contact_force_magnitude'].max():.3f} N")

    print("="*60 + "\n")


def estimate_external_torque(robot: Articulation, joint_acc=None):
    """
    외부에서 가해진 토크 추정 (Joint 레벨)

    Returns:
        external_torque: [N, num_joints] - command_torque
    """

    # 1. Controller 명령 토크
    commanded_torque = robot.data.applied_torque  # [N, J]

    # 2. 실제 dynamics (외부력 포함)
    M = robot.root_physx_view.get_generalized_mass_matrices()  # [N, J, J]
    C = robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()  # [N, J]
    G = robot.root_physx_view.get_gravity_compensation_forces()  # [N, J]
    q_ddot = joint_acc if joint_acc is not None else robot.data.joint_acc  # [N, J]

    # M * q̈
    M_times_acc = torch.bmm(M, q_ddot.unsqueeze(-1)).squeeze(-1)  # [N, J]

    # 실제 필요한 토크
    actual_torque = M_times_acc + C
    actual_torque_g = M_times_acc + C + G

    damping_coeff = robot.data.default_joint_damping
    armature = robot.data.default_joint_armature
    friction_coeff = robot.data.default_joint_friction_coeff

    # 3. 외부 토크 = 실제 - 명령
    external_torque = actual_torque - commanded_torque
    external_torque_g = actual_torque_g - commanded_torque

    return external_torque, external_torque_g, damping_coeff, armature, friction_coeff


class AccelerationEstimator:
    def __init__(self, robot: Articulation, dt: float):
        self.robot = robot
        self.dt = dt
        self.prev_joint_vel = None

    def reset(self):
        """Reset 시 호출"""
        self.prev_joint_vel = None

    def estimate(self):
        """가속도 추정"""
        current_vel = self.robot.data.joint_vel

        if self.prev_joint_vel is None:
            joint_acc = torch.zeros_like(current_vel)
        else:
            joint_acc = (current_vel - self.prev_joint_vel) / self.dt

        self.prev_joint_vel = current_vel.clone()
        return joint_acc


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # cuRobo target (-0.65, -1.0, 0.70) 근처를 보도록 카메라 조정
    sim.set_camera_view([1.5, 1.5, 2.0], [-0.5, -0.8, 0.5])

    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)  # 더 넓은 간격
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: HDR35_20 F/T Sensor Test Setup complete...")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
