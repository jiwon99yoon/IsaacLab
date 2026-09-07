# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
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

    NOTE (Bug fix):
    - 기존: object.data.root_pos_w → Wire articulation의 root position (left_ring이 아님)
    - 수정: object.data.body_pos_w[:, object_cfg.body_ids] → left_ring body position
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    # Bug fix: root_pos_w → body_pos_w[:, body_ids] (left_ring body position)
    # 기존: object_pos = object.data.root_pos_w
    object_pos = object.data.body_pos_w[:, object_cfg.body_ids].squeeze(1)  # (num_envs, 3)
    object_ee_distance = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    return 1 - torch.tanh(object_ee_distance / std)


def contacts(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold.

    MODIFIED for Remove Hook Task with HDR35_20 + DG5F_L (Tesollo Hand Left):
    - DG5F Left Hand: 20-DOF fully-actuated (5 fingers x 4 joints)
    - Joint naming: lj_dg_[1-5]_[1-4] (finger_joint)
    - Collision links (fingertips): ll_dg_1_4, ll_dg_2_4, ll_dg_3_4, ll_dg_4_4, ll_dg_5_4
    - Sensor suffix: _hook_sensor (for hook/wire contact)
    - Target: right_ring (왼손이므로 오른쪽 고리를 타겟으로 함)
    """

    # DG5F Left Hand contact sensors (defined in env_cfg.py)
    # HDR35_20_DG5F_L: Left hand, fully-actuated
    # Finger numbering: 1=thumb, 2=index, 3=middle, 4=ring, 5=little
    thumb_contact_sensor: ContactSensor = env.scene.sensors["ll_dg_1_4_hook_sensor"]
    index_contact_sensor: ContactSensor = env.scene.sensors["ll_dg_2_4_hook_sensor"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["ll_dg_3_4_hook_sensor"]
    ring_contact_sensor: ContactSensor = env.scene.sensors["ll_dg_4_4_hook_sensor"]
    last_contact_sensor: ContactSensor = env.scene.sensors["ll_dg_5_4_hook_sensor"]

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

    # === CONTACT CONDITIONS FOR DG5F LEFT HAND ===
    # DG5F Left Hand: 20-DOF fully-actuated design
    # - Fully-actuated = More precise control, but more complex
    # - Strategy: Start with relaxed conditions, gradually increase difficulty
    #
    # HISTORY (from DG5F development):
    # ORIGINAL (DG5F - 너무 엄격, 중지 필수):
    #   good_contact_cond1 = (thumb & middle) & (index | ring | little)
    #
    # 1st TRY (Kuka-like - 엄지 + 아무거나): CURRENT for DG5F
    #   good_contact_cond1 = thumb & (index | middle | ring | little)
    #
    # 2nd TRY (엄지 + 약지 필수 + 아무거나):
    #   good_contact_cond1 = (thumb & ring) & (index | middle | little)
    #
    # FINAL (최대한 완화 - 아무 2개 손가락):
    #   contact_count >= 2  (Any 2 fingers)

    # Current Strategy for DG5F Left Hand (20-DOF fully-actuated):
    # - Thumb opposition + any other finger
    # - DG5F has precise control, can be more strict later if needed
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


# ==========================================================================
# BODY-TRACKING POSITION REWARD (ADDED 2026-01-08)
# ==========================================================================
# NOTE: position_command_error_tanh와 유사하지만:
#   1. body_names 지원 (wire의 right_ring 등 특정 body 추적 가능)
#   2. contact gating 없음 (wire는 grasp 대상이 아니므로)
# 기존 방식으로 되돌리려면 이 함수 사용하는 reward를 주석처리


def position_command_error_tanh_body(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward tracking of commanded position using tanh kernel.

    MODIFIED for Remove Hook Task (DG5F_L):
    - Supports body_names in align_asset_cfg (e.g., right_ring body)
    - No contact gating (wire manipulation doesn't require grasp)
    - NOTE: DG5F_L targets right_ring (vs RH56F1_R targets left_ring)

    Args:
        env: The environment.
        std: Standard deviation for tanh kernel.
        command_name: Name of the command to track.
        asset_cfg: Robot asset config (for frame transformation).
        align_asset_cfg: Object asset config with optional body_names.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset = env.scene[asset_cfg.name]
    object_asset = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Obtain desired position in world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )

    # MODIFIED: Support body_names for tracking specific body
    if align_asset_cfg.body_ids is not None and len(align_asset_cfg.body_ids) > 0:
        # Track specific body (e.g., right_ring for wire)
        body_id = align_asset_cfg.body_ids[0]
        object_pos_w = object_asset.data.body_pos_w[:, body_id, :]
    else:
        # Fallback to root position (original behavior)
        object_pos_w = object_asset.data.root_pos_w

    # Compute distance and reward
    distance = torch.norm(object_pos_w - des_pos_w, dim=1)

    # Pure distance-based reward (no contact gating)
    return 1 - torch.tanh(distance / std)


# ==========================================================================


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
    table_height = 0.255  # Table surface at z=0.255m (from original dexsuite config)

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


# ============================================================================
# HOOK-SPECIFIC REWARD FUNCTIONS (Remove Hook Task)
# ============================================================================


def hook_spring_distance(
    env: ManagerBasedRLEnv,
    separation_threshold: float = 0.05,
    hook_cfg: SceneEntityCfg = SceneEntityCfg("object", body_names=["right-ring/right_ring_inside/right_ring_cover"]),
    spring_cfg: SceneEntityCfg = SceneEntityCfg("object", body_names=["wire-model/right_struct_spring"]),
) -> torch.Tensor:
    """
    Reward for separating hook from spring (Remove Hook task).

    This reward encourages the robot to pull the hook away from the spring
    until a minimum separation distance is achieved, indicating successful removal.

    Args:
        env: The environment.
        separation_threshold: Minimum distance (m) between hook and spring for success.
                             Default 0.05m (5cm).
        hook_cfg: Scene entity for the hook. Defaults to ``SceneEntityCfg("object", body_names=[...])``
                  which references the right_ring_cover child prim.
        spring_cfg: Scene entity for the spring. Defaults to ``SceneEntityCfg("object", body_names=[...])``
                    which references the right_struct_spring child prim.

    Returns:
        Tensor of shape (num_envs,): Reward value between 0 and 1.
                                     1.0 when separation >= threshold (success)
                                     Scaled linearly from 0 to 1 below threshold

    Note:
        Both hook and spring are child prims of the chassis assembly RigidObject.
        We use body_names to access their specific positions within the assembly.
    """
    chassis: RigidObject = env.scene[hook_cfg.name]

    # Get hook position (child prim)
    hook_pos_w = chassis.data.body_pos_w[:, hook_cfg.body_ids[0], :]

    # Get spring position (child prim)
    spring_pos_w = chassis.data.body_pos_w[:, spring_cfg.body_ids[0], :]

    # Calculate 3D distance between hook and spring
    separation_distance = torch.norm(hook_pos_w - spring_pos_w, dim=-1)

    # Reward scales from 0 to 1 as distance increases from 0 to threshold
    # Once separation >= threshold, reward is 1.0 (full success)
    reward = torch.clamp(separation_distance / separation_threshold, 0.0, 1.0)

    return reward


def spring_contact_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
) -> torch.Tensor:
    """
    Penalty for colliding with spring (Remove Hook task).

    The robot should grasp the wire/hook, not the spring. This penalty discourages
    spring collisions and guides the policy toward precise wire grasping.

    Args:
        env: The environment.
        threshold: Contact force threshold (N) above which penalty is applied.
                  Default 1.0N.

    Returns:
        Tensor of shape (num_envs,): Penalty value (0 or 1).
                                     1.0 when spring contact force > threshold
                                     0.0 otherwise

    Note:
        This function requires a contact sensor named "spring_collision_sensor"
        to be configured in the environment scene.
        The sensor should filter for spring collision (e.g., right_struct_spring, left_struct_spring).
    """
    # Get spring collision sensor (configured in env_cfg.py)
    # Sensor is placed on gripper_base_link (palm/wrist)
    # Filters for spring prims: right_struct_spring, left_struct_spring
    spring_contact_sensor: ContactSensor = env.scene.sensors["spring_collision_sensor"]

    # Get contact force magnitude
    spring_contact_force = spring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    spring_contact_mag = torch.norm(spring_contact_force, dim=-1)

    # Penalty when contact force exceeds threshold
    penalty = (spring_contact_mag > threshold).float()

    return penalty


class hook_to_target_distance(ManagerTermBase):
    """
    Class-based reward for moving hook to fixed target position (Remove Hook task).

    Target position is calculated once at environment reset as:
        target = initial_hook_position + offset

    This ensures the target remains fixed in world space throughout the episode.

    Args (from cfg.params):
        std: Standard deviation for tanh kernel (smaller = tighter tolerance). Default 0.02m (2cm).
        target_offset: (dx, dy, dz) offset from initial hook position. Default (-0.2, 0.0, 0.3).
        hook_cfg: Scene entity for the hook. Default ``SceneEntityCfg("object", body_names=[...])``

    Returns (from __call__):
        Tensor of shape (num_envs,): Reward value between 0 and 1.
                                     Close to 1.0 when hook is near target.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # Get parameters
        self.std = cfg.params.get("std", 0.02)
        self.target_offset = torch.tensor(
            cfg.params.get("target_offset", (-0.2, 0.0, 0.3)),
            device=env.device,
            dtype=torch.float32,
        )
        self.hook_cfg: SceneEntityCfg = cfg.params.get(
            "hook_cfg",
            SceneEntityCfg("object", body_names=["right_ring"])
        )

        # Storage for initial hook positions and target positions (set at reset)
        self.initial_hook_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        self.target_pos_w = torch.zeros(env.num_envs, 3, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        """Called when environments are reset. Store initial hook position and calculate target."""
        chassis: RigidObject = self._env.scene[self.hook_cfg.name]

        # Get initial hook position for reset environments
        hook_pos_w = chassis.data.body_pos_w[env_ids, self.hook_cfg.body_ids[0], :]

        # Store initial position
        self.initial_hook_pos_w[env_ids] = hook_pos_w

        # Calculate fixed target position: initial + offset
        self.target_pos_w[env_ids] = hook_pos_w + self.target_offset.unsqueeze(0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float = 0.02,
        target_offset: tuple[float, float, float] = (-0.2, 0.0, 0.3),
        hook_cfg: SceneEntityCfg = SceneEntityCfg("object", body_names=["right-ring/right_ring_inside/right_ring_cover"]),
    ) -> torch.Tensor:
        """Compute reward for hook proximity to target.

        Note: Parameters are kept for API compatibility but actual values are set in __init__.
        """
        chassis: RigidObject = env.scene[self.hook_cfg.name]

        # Get current hook position
        hook_pos_w = chassis.data.body_pos_w[:, self.hook_cfg.body_ids[0], :]

        # Calculate distance to fixed target
        distance = torch.norm(hook_pos_w - self.target_pos_w, dim=-1)

        # Tanh kernel reward (smooth, bounded between 0 and 1)
        reward = 1.0 - torch.tanh(distance / self.std)

        return reward


def hook_grasped(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,
    min_fingers: int = 2,
) -> torch.Tensor:
    """
    Binary reward for successfully grasping the hook (Remove Hook task).

    Success condition: At least min_fingers have contact force > threshold.

    Args:
        env: The environment.
        threshold: Contact force threshold (N). Default 0.5N (lower than cube grasping
                  due to thin wire).
        min_fingers: Minimum number of fingers required for stable grasp. Default 2.

    Returns:
        Tensor of shape (num_envs,): Binary reward (0 or 1).
                                     1.0 when hook is grasped with sufficient fingers.

    Note:
        This function requires contact sensors on fingertips (configured in env_cfg.py).
        DG5F_L sensor names: ll_dg_1_4_hook_sensor, ll_dg_2_4_hook_sensor, etc.
    """
    # Get contact forces from all fingertip sensors
    # DG5F Left hand: ll_dg_[finger]_4 (finger: 1=thumb, 2=index, 3=middle, 4=ring, 5=little)
    finger_sensors = [
        "ll_dg_1_4_hook_sensor",
        "ll_dg_2_4_hook_sensor",
        "ll_dg_3_4_hook_sensor",
        "ll_dg_4_4_hook_sensor",
        "ll_dg_5_4_hook_sensor",
    ]

    # Count fingers with contact force > threshold
    fingers_in_contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    for sensor_name in finger_sensors:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        contact_force = sensor.data.force_matrix_w.view(env.num_envs, 3)
        contact_mag = torch.norm(contact_force, dim=-1)

        # Increment count if this finger has sufficient contact
        fingers_in_contact += (contact_mag > threshold).float()

    # Binary reward: 1.0 if >= min_fingers in contact, 0.0 otherwise
    grasped = (fingers_in_contact >= min_fingers).float()

    return grasped