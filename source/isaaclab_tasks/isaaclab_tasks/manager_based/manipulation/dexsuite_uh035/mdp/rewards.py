# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(-1000, 1000)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-1000, 1000)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reaching the object using a tanh-kernel on end-effector distance.

    The reward is close to 1 when the maximum distance between the object and any end-effector body is small.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    object_pos = object.data.root_pos_w
    object_ee_distance = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    return 1 - torch.tanh(object_ee_distance / std)


def contacts(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold.

    MODIFIED for UH035 + Inspire Hand (6-DOF under-actuated):
    - Changed from DG5F (20-DOF, 5 fingers x 4 links) to Inspire Hand
    - Inspire Hand fingertips: thumb_4, index_2, middle_2, ring_2, little_2
    """

    # Inspire Hand contact sensors (defined in env_cfg.py)
    # NOTE: Sensor names use underscore (converted from USD path with slash)
    thumb_contact_sensor: ContactSensor = env.scene.sensors["RH56F1_L_left_thumb_4_object_s"]
    index_contact_sensor: ContactSensor = env.scene.sensors["RH56F1_L_left_index_2_object_s"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["RH56F1_L_left_middle_2_object_s"]
    ring_contact_sensor: ContactSensor = env.scene.sensors["RH56F1_L_left_ring_2_object_s"]
    last_contact_sensor: ContactSensor = env.scene.sensors["RH56F1_L_left_little_2_object_s"]

    # check if contact force is above threshold
    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_contact = middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    ring_contact = ring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    last_contact = last_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)

    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)
    middle_contact_mag = torch.norm(middle_contact, dim=-1)
    ring_contact_mag = torch.norm(ring_contact, dim=-1)
    last_contact_mag = torch.norm(last_contact, dim=-1)

    # ========== DEBUG: Print contact forces (first env only) ==========
    # if env.episode_length_buf[0] % 100 == 0:  # Print every 100 steps
    #     print(f"\n[DEBUG Contact Forces - Step {env.episode_length_buf[0].item()}]")
    #     print(f"  Thumb: {thumb_contact_mag[0].item():.4f} N")
    #     print(f"  Index: {index_contact_mag[0].item():.4f} N")
    #     print(f"  Middle: {middle_contact_mag[0].item():.4f} N")
    #     print(f"  Ring: {ring_contact_mag[0].item():.4f} N")
    #     print(f"  Little: {last_contact_mag[0].item():.4f} N")
    #     print(f"  Max: {max(thumb_contact_mag[0], index_contact_mag[0], middle_contact_mag[0], ring_contact_mag[0], last_contact_mag[0]).item():.4f} N")
    # ==================================================================

    # === CONTACT CONDITIONS FOR INSPIRE HAND ===
    # Inspire Hand: 6-DOF under-actuated design (vs DG5F 20-DOF)
    # - Under-actuated = More compliant, adaptive grasping
    # - Strategy: Start with relaxed conditions, gradually increase difficulty
    #
    # HISTORY (from DG5F development):
    # ORIGINAL (DG5F - 너무 엄격, 중지 필수):
    #   good_contact_cond1 = (thumb & middle) & (index | ring | little)
    #
    # 1st TRY (Kuka-like - 엄지 + 아무거나): CURRENT for Inspire Hand
    #   good_contact_cond1 = thumb & (index | middle | ring | little)
    #
    # 2nd TRY (엄지 + 약지 필수 + 아무거나):
    #   good_contact_cond1 = (thumb & ring) & (index | middle | little)
    #
    # FINAL (최대한 완화 - 아무 2개 손가락):
    #   contact_count >= 2  (Any 2 fingers)

    # Current Strategy for Inspire Hand (6-DOF under-actuated):
    # - Thumb opposition + any other finger (most natural for under-actuated hand)
    # - Can adjust based on learning progress
    good_contact_cond1 = (thumb_contact_mag > threshold) & (
        (index_contact_mag > threshold) | (middle_contact_mag > threshold) |
        (ring_contact_mag > threshold) | (last_contact_mag > threshold)
    )

    # Alternative: If learning struggles, try relaxed condition (any 2 fingers):
    # contact_count = (
    #     (thumb_contact_mag > threshold).float() +
    #     (index_contact_mag > threshold).float() +
    #     (middle_contact_mag > threshold).float() +
    #     (ring_contact_mag > threshold).float() +
    #     (last_contact_mag > threshold).float()
    # )
    # good_contact_cond1 = contact_count >= 2

    return good_contact_cond1


def success_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    pos_std: float,
    rot_std: float | None = None,
) -> torch.Tensor:
    """Reward success by comparing commanded pose to the object pose using tanh kernels on error.

    MODIFIED FOR UH035+Inspire: Added contact gating to prevent "pushing without grasping" shortcut.
    Agent was achieving success by pushing object with arm instead of grasping with fingers.
    (Originally developed for HDR-DG5F, adapted for UH035+Inspire Hand)
    """

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, des_quat_w = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3], command[:, 3:7]
    )
    pos_err, rot_err = compute_pose_error(des_pos_w, des_quat_w, object.data.root_pos_w, object.data.root_quat_w)
    pos_dist = torch.norm(pos_err, dim=1)

    # ORIGINAL (Kuka-Allegro - no contact requirement):
    # if not rot_std:
    #     return (1 - torch.tanh(pos_dist / pos_std)) ** 2
    # rot_dist = torch.norm(rot_err, dim=1)
    # return (1 - torch.tanh(pos_dist / pos_std)) * (1 - torch.tanh(rot_dist / rot_std))

    # MODIFIED (UH035+Inspire - requires contact to prevent shortcuts):
    # Success only counts if fingers are in contact with object (prevents pushing shortcuts)
    # Adapted from HDR-DG5F development
    if not rot_std:
        base_reward = (1 - torch.tanh(pos_dist / pos_std)) ** 2
    else:
        rot_dist = torch.norm(rot_err, dim=1)
        base_reward = (1 - torch.tanh(pos_dist / pos_std)) * (1 - torch.tanh(rot_dist / rot_std))

    # Gate success with contact (consistent with position_tracking and orientation_tracking)
    # Uses Inspire Hand contact sensors
    return base_reward * contacts(env, 1.0).float()


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of commanded position using tanh kernel, gated by contact presence."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * contacts(env, 1.0).float()


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of commanded orientation using tanh kernel, gated by contact presence."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = math_utils.quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    quat_distance = math_utils.quat_error_magnitude(object.data.root_quat_w, des_quat_w)

    return (1 - torch.tanh(quat_distance / std)) * contacts(env, 1.0).float()


# ========== ADDITIONAL REWARD FUNCTIONS (Originally developed for HDR-DG5F) ==========
# These reward functions were developed for HDR-DG5F to address specific learning challenges:
# - Preventing "pushing without grasping" shortcuts
# - Encouraging sustained contact and lifting
# Adapted for UH035+Inspire Hand (6-DOF under-actuated)


def object_lift_height(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for lifting the object above the table surface.

    This encourages grasping behavior instead of pushing/tapping.

    Args:
        env: The environment.
        threshold: Minimum height (in meters) above table to get full reward. Default 0.05m (5cm).
        object_cfg: Scene entity for the object.

    Returns:
        Tensor of shape (num_envs,): Reward value between 0 and 1.
    """
    object: RigidObject = env.scene[object_cfg.name]
    table_height = 0.255  # Table surface at z=0.255m (from dexsuite_env_cfg.py)

    # Calculate lift distance
    object_height = object.data.root_pos_w[:, 2]
    lift_distance = object_height - table_height

    # Reward scales from 0 to 1 as object lifts from table to threshold height
    return torch.clamp(lift_distance / threshold, 0.0, 1.0)


def object_ground_contact_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty for object contacting the ground plane.

    This prevents the agent from bouncing the object off the ground to reach target positions.
    Returns negative value (penalty) when object is very close to ground.

    Args:
        env: The environment.
        object_cfg: Scene entity for the object.

    Returns:
        Tensor of shape (num_envs,): Penalty value (0 or negative).
    """
    object: RigidObject = env.scene[object_cfg.name]
    table_height = 0.255
    ground_clearance = 0.02  # Consider "ground contact" if within 2cm of table

    object_height = object.data.root_pos_w[:, 2]
    height_above_table = object_height - table_height

    # Penalty when object is very close to or below table surface
    # Returns 1.0 when touching, 0.0 when safely above
    contact_penalty = torch.clamp(1.0 - height_above_table / ground_clearance, 0.0, 1.0)

    return contact_penalty


def grasp_duration(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    min_duration: int = 10,
) -> torch.Tensor:
    """Reward for maintaining grasp over multiple timesteps.

    This encourages sustained grasping rather than momentary tapping.
    Uses a buffer to track contact history.

    Args:
        env: The environment.
        threshold: Contact force threshold.
        min_duration: Minimum number of consecutive steps to get reward.

    Returns:
        Tensor of shape (num_envs,): Reward value (0 or 1).
    """
    # Initialize buffer on first call
    if not hasattr(grasp_duration, 'contact_buffer'):
        grasp_duration.contact_buffer = torch.zeros(
            (env.num_envs, min_duration),
            device=env.device,
            dtype=torch.bool
        )

    # Get current contact state
    current_contact = contacts(env, threshold)

    # Shift buffer and add current contact
    grasp_duration.contact_buffer = torch.roll(grasp_duration.contact_buffer, -1, dims=1)
    grasp_duration.contact_buffer[:, -1] = current_contact

    # Reward if all recent steps have contact
    sustained_grasp = torch.all(grasp_duration.contact_buffer, dim=1).float()

    return sustained_grasp