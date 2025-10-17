
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
OSC with FT Sensor logging and Contact Force comparison.
"""
#from home.dyros.isaacsim.exts.isaacsim.core.prims.isaacsim.core.prims.impl import SingleArticulation
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on using the operational space controller.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
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

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG


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
            pos=(0.6 + 0.085, 0.0, 0.3), 
            rot=(0.9238795325, 0.0, -0.3826834324, 0.0)
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/TiltedWall",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
    )

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.actuators["panda_shoulder"].stiffness = 0.0
    robot.actuators["panda_shoulder"].damping = 0.0
    robot.actuators["panda_forearm"].stiffness = 0.0
    robot.actuators["panda_forearm"].damping = 0.0
    robot.spawn.rigid_props.disable_gravity = True


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop with FT sensor logging."""
    
    # Extract scene entities
    robot = scene["robot"]
    contact_forces = scene["contact_forces"]
    tilted_wall = scene["tilted_wall"]

    # Obtain indices for the end-effector and arm joints
    ee_frame_name = "panda_leftfinger"
    arm_joint_names = ["panda_joint.*"]
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]

    # ===== FT Sensor Configuration =====
    force_sensor_body_idx = robot.body_names.index("force_sensor")
    link1_idx = robot.body_names.index("panda_link1")
    link3_idx = robot.body_names.index("panda_link3")
    link5_idx = robot.body_names.index("panda_link5")
    # link2_idx = robot.body_names.index("panda_link2")
    # link4_idx = robot.body_names.index("panda_link4")
    # link6_idx = robot.body_names.index("panda_link6")
    
    # Outlier thresholds
    FORCE_MAX = 15.0   # N
    TORQUE_MAX = 0.4   # Nm
    # ===================================

    # Create the OSC
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs", "wrench_abs"],
        impedance_mode="variable_kp",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_damping_ratio_task=1.0,
        contact_wrench_stiffness_task=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        motion_control_axes_task=[1, 1, 0, 1, 1, 1],
        contact_wrench_control_axes_task=[0, 0, 1, 0, 0, 0],
        nullspace_control="position",
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define targets for the arm
    ee_goal_pose_set_tilted_b = torch.tensor(
        [
            [0.6, 0.15, 0.3, 0.0, 0.92387953, 0.0, 0.38268343],
            [0.6, -0.3, 0.3, 0.0, 0.92387953, 0.0, 0.38268343],
            [0.8, 0.0, 0.5, 0.0, 0.92387953, 0.0, 0.38268343],
        ],
        device=sim.device,
    )
    # changed from 10 -> 10000
    ee_goal_wrench_set_tilted_task = torch.tensor(
        [
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        ],
        device=sim.device,
    )
    kp_set_task = torch.tensor(
        [
            [360.0, 360.0, 360.0, 360.0, 360.0, 360.0],
            [420.0, 420.0, 420.0, 420.0, 420.0, 420.0],
            [320.0, 320.0, 320.0, 320.0, 320.0, 320.0],
        ],
        device=sim.device,
    )
    ee_target_set = torch.cat([ee_goal_pose_set_tilted_b, ee_goal_wrench_set_tilted_task, kp_set_task], dim=-1)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    robot.update(dt=sim_dt)

    # Get the center of the robot soft joint limits
    joint_centers = torch.mean(robot.data.soft_joint_pos_limits[:, arm_joint_ids, :], dim=-1)

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
                joint_efforts = osc.compute(
                    jacobian_b=jacobian_b,
                    current_ee_pose_b=ee_pose_b,
                    current_ee_vel_b=ee_vel_b,
                    current_ee_force_b=ee_force_b,
                    mass_matrix=mass_matrix,
                    gravity=gravity,
                    current_joint_pos=joint_pos,
                    current_joint_vel=joint_vel,
                    nullspace_joint_pos_target=joint_centers,
                )

                # Log data every 5 steps
                if count % 5 == 0:
                    # ===== 1. Contact Force (X, Y, Z 각각) =====
                    # contact_forces.data.net_forces_w shape: [num_envs, num_contacts, 3]
                    # 첫 번째 환경의 첫 번째 contact의 force
                    if contact_forces.data.net_forces_w.shape[1] > 0:
                        contact_force_raw = contact_forces.data.net_forces_w[0, 0]  # [3] 벡터
                    else:
                        contact_force_raw = torch.zeros(3, device=sim.device)
                    
                    contact_force_x = contact_force_raw[0].item()
                    contact_force_y = contact_force_raw[1].item()
                    contact_force_z = contact_force_raw[2].item()
                    contact_force_magnitude = torch.norm(contact_force_raw).item()
                    
                    # ===== 2. FT Sensor 데이터 (6-DOF) =====
                    ft_sensor_forces = robot.root_physx_view.get_link_incoming_joint_force()
                    force_sensor_data = ft_sensor_forces[0][force_sensor_body_idx]  # [6] 벡터
                
                    # Force (X, Y, Z) - Raw
                    fs_force_x_raw = force_sensor_data[0].item()
                    fs_force_y_raw = force_sensor_data[1].item()
                    fs_force_z_raw = force_sensor_data[2].item()
                    
                    # Torque (X, Y, Z) - Raw
                    fs_torque_x_raw = force_sensor_data[3].item()
                    fs_torque_y_raw = force_sensor_data[4].item()
                    fs_torque_z_raw = force_sensor_data[5].item()
                    
                    
                    actual_torques = robot.data.applied_torque[:, arm_joint_ids]
                    actual_torque_data = actual_torques[0] # 첫번째 환경
                    # Dictionary로 저장
                    # for i in range(7):
                    #     act_tor_j{i+1} = actual_torque_data[i].item()
                    # act_tor = [actual_torque_data[i].item() for i in range(7)]

                    # 첫 번째 환경의 각 joint torque
                    act_tor_j1 = actual_torques[0, 0].item()
                    act_tor_j2 = actual_torques[0, 1].item()
                    act_tor_j3 = actual_torques[0, 2].item()
                    act_tor_j4 = actual_torques[0, 3].item()
                    act_tor_j5 = actual_torques[0, 4].item()
                    act_tor_j6 = actual_torques[0, 5].item()
                    act_tor_j7 = actual_torques[0, 6].item()
                    
                    # ===== 3. Outlier Removal =====
                    fs_force_x = clip_outlier(fs_force_x_raw, FORCE_MAX)
                    fs_force_y = clip_outlier(fs_force_y_raw, FORCE_MAX)
                    fs_force_z = clip_outlier(fs_force_z_raw, FORCE_MAX)
                    
                    fs_torque_x = clip_outlier(fs_torque_x_raw, TORQUE_MAX)
                    fs_torque_y = clip_outlier(fs_torque_y_raw, TORQUE_MAX)
                    fs_torque_z = clip_outlier(fs_torque_z_raw, TORQUE_MAX)
                    
                    # ============== link 2, 4, 6 data ======================= #
                    all_link_forces = robot.root_physx_view.get_link_incoming_joint_force()

                    # link_indices = {
                    #     2: link2_idx,
                    #     4: link4_idx,
                    #     6: link6_idx
                    # }
                    link_indices = {
                        1: link1_idx,
                        3: link3_idx,
                        5: link5_idx
                    }

                    link_data = {}

                    for joint_num, link_idx in link_indices.items():
                        sensor_data = all_link_forces[0][link_idx]
                        
                        link_data[f'link{joint_num}_force_x_raw'] = sensor_data[0].item()
                        link_data[f'link{joint_num}_force_y_raw'] = sensor_data[1].item()
                        link_data[f'link{joint_num}_force_z_raw'] = sensor_data[2].item()
                        link_data[f'link{joint_num}_torque_x_raw'] = sensor_data[3].item()
                        link_data[f'link{joint_num}_torque_y_raw'] = sensor_data[4].item()
                        link_data[f'link{joint_num}_torque_z_raw'] = sensor_data[5].item()
                    
                    # # ===== 실제 Joint Efforts (시뮬레이션이 계산한 실제 토크) ⭐ NEW! =====
                    # actual_joint_efforts_full = robot.root_physx_view.get_joint_efforts()[0]  # 모든 joint
                    # actual_joint_efforts_arm = actual_joint_efforts_full[arm_joint_ids]  # arm joint만
                    
                    # # ===== Measured Joint Efforts (센서 측정 토크) ⭐ NEW! =====
                    # measured_joint_efforts_full = robot.root_physx_view.get_measured_joint_efforts()[0]  # 모든 joint
                    # measured_joint_efforts_arm = measured_joint_efforts_full[arm_joint_ids]  # arm joint만

                    # ===== 4. OSC Joint Torques =====
                    osc_joint_torques = joint_efforts[0]
                    
                    # ===== 5. 데이터 저장 =====
                    current_log = {
                        "step": count,
                        
                        # Contact Force (각 축별)
                        "contact_force_x": contact_force_x,
                        "contact_force_y": contact_force_y,
                        "contact_force_z": contact_force_z,
                        "contact_force_magnitude": contact_force_magnitude,
                    
                        #"actual_torques" : actual_torques
                        
                        # FT Sensor - Force (raw)
                        "fs_force_x_raw": fs_force_x_raw,
                        "fs_force_y_raw": fs_force_y_raw,
                        "fs_force_z_raw": fs_force_z_raw,
                        
                        # FT Sensor - Force (filtered)
                        "fs_force_x": fs_force_x,
                        "fs_force_y": fs_force_y,
                        "fs_force_z": fs_force_z,
                        
                        # FT Sensor - Torque (raw)
                        "fs_torque_x_raw": fs_torque_x_raw,
                        "fs_torque_y_raw": fs_torque_y_raw,
                        "fs_torque_z_raw": fs_torque_z_raw,
                        
                        # FT Sensor - Torque (filtered)
                        "fs_torque_x": fs_torque_x,
                        "fs_torque_y": fs_torque_y,
                        "fs_torque_z": fs_torque_z,

                        # link1 sensor - force (raw)
                        "link_data[joint2_force_x_raw]" : link_data['link1_force_x_raw'],
                        "link_data[joint2_force_y_raw]" : link_data['link1_force_y_raw'],
                        "link_data[joint2_force_z_raw]" : link_data['link1_force_z_raw'],

                        # link2 sensor - torque (raw)
                        "link_data[joint2_torque_x_raw]" : link_data['link1_torque_x_raw'],
                        "link_data[joint2_torque_y_raw]" : link_data['link1_torque_y_raw'],
                        "link_data[joint2_torque_z_raw]" : link_data['link1_torque_z_raw'],

                        # link4 sensor - force (raw)
                        "link_data[joint4_force_x_raw]" : link_data['link3_force_x_raw'],
                        "link_data[joint4_force_y_raw]" : link_data['link3_force_y_raw'],
                        "link_data[joint4_force_z_raw]" : link_data['link3_force_z_raw'],

                        # link4 sensor - torque (raw)
                        "link_data[joint4_torque_x_raw]" : link_data['link3_torque_x_raw'],
                        "link_data[joint4_torque_y_raw]" : link_data['link3_torque_y_raw'],
                        "link_data[joint4_torque_z_raw]" : link_data['link3_torque_z_raw'],

                        # link6 sensor - force (raw)
                        "link_data[joint6_force_x_raw]" : link_data['link5_force_x_raw'],
                        "link_data[joint6_force_y_raw]" : link_data['link5_force_y_raw'],
                        "link_data[joint6_force_z_raw]" : link_data['link5_force_z_raw'],

                        # link6 sensor - torque (raw)
                        "link_data[joint6_torque_x_raw]" : link_data['link5_torque_x_raw'],
                        "link_data[joint6_torque_y_raw]" : link_data['link5_torque_y_raw'],
                        "link_data[joint6_torque_z_raw]" : link_data['link5_torque_z_raw'],
                        # # link2 sensor - force (raw)
                        # "link_data[joint3_force_x_raw]" : link_data['link2_force_x_raw'],
                        # "link_data[joint3_force_y_raw]" : link_data['link2_force_y_raw'],
                        # "link_data[joint3_force_z_raw]" : link_data['link2_force_z_raw'],

                        # # link2 sensor - torque (raw)
                        # "link_data[joint3_torque_x_raw]" : link_data['link2_torque_x_raw'],
                        # "link_data[joint3_torque_y_raw]" : link_data['link2_torque_y_raw'],
                        # "link_data[joint3_torque_z_raw]" : link_data['link2_torque_z_raw'],

                        # # link4 sensor - force (raw)
                        # "link_data[joint5_force_x_raw]" : link_data['link4_force_x_raw'],
                        # "link_data[joint5_force_y_raw]" : link_data['link4_force_y_raw'],
                        # "link_data[joint5_force_z_raw]" : link_data['link4_force_z_raw'],

                        # # link4 sensor - torque (raw)
                        # "link_data[joint5_torque_x_raw]" : link_data['link4_torque_x_raw'],
                        # "link_data[joint5_torque_y_raw]" : link_data['link4_torque_y_raw'],
                        # "link_data[joint5_torque_z_raw]" : link_data['link4_torque_z_raw'],

                        # # link6 sensor - force (raw)
                        # "link_data[joint7_force_x_raw]" : link_data['link6_force_x_raw'],
                        # "link_data[joint7_force_y_raw]" : link_data['link6_force_y_raw'],
                        # "link_data[joint7_force_z_raw]" : link_data['link6_force_z_raw'],

                        # # link6 sensor - torque (raw)
                        # "link_data[joint7_torque_x_raw]" : link_data['link6_torque_x_raw'],
                        # "link_data[joint7_torque_y_raw]" : link_data['link6_torque_y_raw'],
                        # "link_data[joint7_torque_z_raw]" : link_data['link6_torque_z_raw'],

                        "act_tor_j1" : act_tor_j1,
                        "act_tor_j2" : act_tor_j2,
                        "act_tor_j3" : act_tor_j3,
                        "act_tor_j4" : act_tor_j4,
                        "act_tor_j5" : act_tor_j5,
                        "act_tor_j6" : act_tor_j6,
                        "act_tor_j7" : act_tor_j7,

                    }
                    
                    # OSC 토크 추가 (7개 관절)
                    for i in range(7):
                        current_log[f'osc_torque_j{i+1}'] = osc_joint_torques[i]

                    # # 실제 토크 추가 (7개 관절)
                    # for i in range(7):
                    #     current_log[f'act_tor_j{i+1}'] = act_tor[i].item()

                    # # ===== 10. Actual Joint Efforts (실제 시뮬레이션 토크) ⭐ NEW! =====
                    # for i in range(7):
                    #     current_log[f'actual_effort_j{i+1}'] = actual_joint_efforts_arm[i].item()
                    
                    # # ===== 11. Measured Joint Efforts (센서 측정 토크) ⭐ NEW! =====
                    # for i in range(7):
                    #     current_log[f'measured_effort_j{i+1}'] = measured_joint_efforts_arm[i].item()
                    
                    # ===== 12. Joint 위치와 속도 ⭐ NEW! =====
                    for i in range(7):
                        current_log[f'joint_pos_j{i+1}'] = joint_pos[0][i].item()
                        current_log[f'joint_vel_j{i+1}'] = joint_vel[0][i].item()


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
            df.to_csv("simulation_log_enhanced.csv", index=False)
            print(f"Log data saved to simulation_log_enhanced.csv ({len(log_data)} records)")
            
            # Print summary statistics
            print_comparison_summary(log_data)
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


def print_comparison_summary(log_data):
    """Print comparison summary between contact force and FT sensor"""
    
    df = pd.DataFrame(log_data)
    
    print("\n" + "="*60)
    print("Contact Force vs FT Sensor Comparison Summary")
    print("="*60)
    
    # X-axis comparison
    print("\nX-Axis Force:")
    print(f"  Contact Force - Mean: {df['contact_force_x'].mean():.3f} N, Std: {df['contact_force_x'].std():.3f} N")
    print(f"  FT Sensor     - Mean: {df['fs_force_x'].mean():.3f} N, Std: {df['fs_force_x'].std():.3f} N")
    if df['fs_force_x'].std() > 0:
        print(f"  Correlation: {df['contact_force_x'].corr(df['fs_force_x']):.3f}")
    
    # Y-axis comparison
    print("\nY-Axis Force:")
    print(f"  Contact Force - Mean: {df['contact_force_y'].mean():.3f} N, Std: {df['contact_force_y'].std():.3f} N")
    print(f"  FT Sensor     - Mean: {df['fs_force_y'].mean():.3f} N, Std: {df['fs_force_y'].std():.3f} N")
    if df['fs_force_y'].std() > 0:
        print(f"  Correlation: {df['contact_force_y'].corr(df['fs_force_y']):.3f}")
    
    # Z-axis comparison
    print("\nZ-Axis Force:")
    print(f"  Contact Force - Mean: {df['contact_force_z'].mean():.3f} N, Std: {df['contact_force_z'].std():.3f} N")
    print(f"  FT Sensor     - Mean: {df['fs_force_z'].mean():.3f} N, Std: {df['fs_force_z'].std():.3f} N")
    if df['fs_force_z'].std() > 0:
        print(f"  Correlation: {df['contact_force_z'].corr(df['fs_force_z']):.3f}")
    
    # Torque summary
    print("\nFT Sensor Torques:")
    print(f"  X-axis - Mean: {df['fs_torque_x'].mean():.3f} Nm, Std: {df['fs_torque_x'].std():.3f} Nm")
    print(f"  Y-axis - Mean: {df['fs_torque_y'].mean():.3f} Nm, Std: {df['fs_torque_y'].std():.3f} Nm")
    print(f"  Z-axis - Mean: {df['fs_torque_z'].mean():.3f} Nm, Std: {df['fs_torque_z'].std():.3f} Nm")
    
    # Outliers removed
    outlier_force = (df['fs_force_x_raw'].abs() > 15).sum() + \
                    (df['fs_force_y_raw'].abs() > 15).sum() + \
                    (df['fs_force_z_raw'].abs() > 15).sum()
    outlier_torque = (df['fs_torque_x_raw'].abs() > 0.4).sum() + \
                     (df['fs_torque_y_raw'].abs() > 0.4).sum() + \
                     (df['fs_torque_z_raw'].abs() > 0.4).sum()
    
    print(f"\nOutliers Removed:")
    print(f"  Force outliers (>15N): {outlier_force}")
    print(f"  Torque outliers (>0.4Nm): {outlier_torque}")
    print("="*60 + "\n")


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    print("[INFO]: Setup complete...")
    
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()