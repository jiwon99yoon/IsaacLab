# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_apply_inverse

from .utils import reject_force_outliers

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


def contacts(
    env: ManagerBasedRLEnv,
    threshold: float,
    contact_sensor_names: list[str] | None = None
) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold.

    MODIFIED for UR10e + multiple hand types (Inspire/DG5F):
    - Generic version that accepts sensor names as parameter
    - Defaults to Inspire Hand sensors for backward compatibility
    - Supports both 6-DOF (Inspire) and 20-DOF (DG5F) hands

    Args:
        env: The environment.
        threshold: Contact force threshold (N).
        contact_sensor_names: List of 5 sensor names [thumb, index, middle, ring, pinky].
                              If None, defaults to Inspire Hand sensors.

    Returns:
        Boolean tensor indicating good contact (thumb + any other finger).
    """

    # Default to Inspire Hand sensors if not specified (backward compatibility)
    if contact_sensor_names is None:
        contact_sensor_names = [
            "ur10e_RH56F1_L_left_thumb_4_object_s",
            "ur10e_RH56F1_L_left_index_2_object_s",
            "ur10e_RH56F1_L_left_middle_2_object_s",
            "ur10e_RH56F1_L_left_ring_2_object_s",
            "ur10e_RH56F1_L_left_little_2_object_s"
        ]

    # Get contact sensors dynamically
    thumb_contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_names[0]]
    index_contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_names[1]]
    middle_contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_names[2]]
    ring_contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_names[3]]
    last_contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_names[4]]

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
    contact_sensor_names: list[str] | None = None
) -> torch.Tensor:
    """Reward success by comparing commanded pose to the object pose using tanh kernels on error.

    MODIFIED FOR Ur10e+Inspire: Added contact gating to prevent "pushing without grasping" shortcut.
    Agent was achieving success by pushing object with arm instead of grasping with fingers.
    (Originally developed for HDR-DG5F, adapted for Ur10e+Inspire Hand)

    Args:
        env: The environment.
        command_name: Name of the command.
        asset_cfg: Asset configuration.
        align_asset_cfg: Alignment asset configuration.
        pos_std: Position standard deviation.
        rot_std: Rotation standard deviation (optional).
        contact_sensor_names: List of 5 sensor names [thumb, index, middle, ring, pinky].
                              If None, defaults to Inspire Hand sensors.
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

    # MODIFIED (Ur10e+Inspire - requires contact to prevent shortcuts):
    # Success only counts if fingers are in contact with object (prevents pushing shortcuts)
    # Adapted from HDR-DG5F development
    if not rot_std:
        base_reward = (1 - torch.tanh(pos_dist / pos_std)) ** 2
    else:
        rot_dist = torch.norm(rot_err, dim=1)
        base_reward = (1 - torch.tanh(pos_dist / pos_std)) * (1 - torch.tanh(rot_dist / rot_std))

    # Gate success with contact (consistent with position_tracking and orientation_tracking)
    return base_reward * contacts(env, 1.0, contact_sensor_names).float()


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    contact_sensor_names: list[str] | None = None
) -> torch.Tensor:
    """Reward tracking of commanded position using tanh kernel, gated by contact presence.

    Args:
        env: The environment.
        std: Standard deviation for tanh kernel.
        command_name: Name of the command.
        asset_cfg: Asset configuration.
        align_asset_cfg: Alignment asset configuration.
        contact_sensor_names: List of 5 sensor names [thumb, index, middle, ring, pinky].
                              If None, defaults to Inspire Hand sensors.
    """

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * contacts(env, 1.0, contact_sensor_names).float()


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    contact_sensor_names: list[str] | None = None
) -> torch.Tensor:
    """Reward tracking of commanded orientation using tanh kernel, gated by contact presence.

    Args:
        env: The environment.
        std: Standard deviation for tanh kernel.
        command_name: Name of the command.
        asset_cfg: Asset configuration.
        align_asset_cfg: Alignment asset configuration.
        contact_sensor_names: List of 5 sensor names [thumb, index, middle, ring, pinky].
                              If None, defaults to Inspire Hand sensors.
    """

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = math_utils.quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    quat_distance = math_utils.quat_error_magnitude(object.data.root_quat_w, des_quat_w)

    return (1 - torch.tanh(quat_distance / std)) * contacts(env, 1.0, contact_sensor_names).float()


# ========== ADDITIONAL REWARD FUNCTIONS (Originally developed for HDR-DG5F) ==========
# These reward functions were developed for HDR-DG5F to address specific learning challenges:
# - Preventing "pushing without grasping" shortcuts
# - Encouraging sustained contact and lifting
# Adapted for Ur10e+Inspire Hand (6-DOF under-actuated)


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
    contact_sensor_names: list[str] | None = None
) -> torch.Tensor:
    """Reward for maintaining grasp over multiple timesteps.

    This encourages sustained grasping rather than momentary tapping.
    Uses a buffer to track contact history.

    Args:
        env: The environment.
        threshold: Contact force threshold.
        min_duration: Minimum number of consecutive steps to get reward.
        contact_sensor_names: List of 5 sensor names [thumb, index, middle, ring, pinky].
                              If None, defaults to Inspire Hand sensors.

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

    # Get current contact state (pass sensor names to contacts function)
    current_contact = contacts(env, threshold, contact_sensor_names)

    # Shift buffer and add current contact
    grasp_duration.contact_buffer = torch.roll(grasp_duration.contact_buffer, -1, dims=1)
    grasp_duration.contact_buffer[:, -1] = current_contact

    # Reward if all recent steps have contact
    sustained_grasp = torch.all(grasp_duration.contact_buffer, dim=1).float()

    return sustained_grasp


# ============================================================================
# ATI F/T SENSOR REWARDS (Forge approach - Improved)
# ============================================================================
#
# REFERENCE: isaaclab_tasks/direct/forge/forge_env.py:237-239
#
# KEY IMPROVEMENTS over original implementation:
# 1. Reuse force from observation (no duplicate calculation)
# 2. threshold: 20N → 50N (accounts for hand weight ~34N static force)
# 3. penalty_scale: 0.3 → 0.05 (auxiliary role, not dominant)
# 4. Use LOCAL frame force (consistent interpretation)
#
# FORGE PENALTY FORMULA (forge_env.py:238-239):
#   contact_force = torch.norm(force_sensor_smooth[:, 0:3], p=2, dim=-1)
#   contact_penalty = relu(contact_force - threshold)
#
# PURPOSE:
# - Protect real hardware from excessive collision forces
# - Penalize: Hand hitting table, excessive gripping force
# - NOT penalize: Normal grasping contact with object
#
# FUTURE IMPROVEMENTS (TODO):
# - [ ] Phase 2: Randomize threshold per episode (Forge style)
# - [ ] Phase 2: Add torque-based slip penalty
# ============================================================================


def ati_excessive_force_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_body_name: str = "ATI_Axia90_M50",
    use_randomized_threshold: bool = True,  # MODIFIED (2025-11-30): env._ati_force_threshold_buf 사용
) -> torch.Tensor:
    """Penalize excessive force magnitude (collision/hardware protection).

    REFERENCE: forge_env.py:237-239
        contact_force = torch.norm(self.force_sensor_smooth[:, 0:3], p=2, dim=-1)
        contact_penalty = torch.nn.functional.relu(contact_force - self.contact_penalty_thresholds)

    PENALTY FORMULA:
        penalty = ReLU(||force|| - threshold)
        - force < threshold: penalty = 0 (OK)
        - force > threshold: penalty = force - threshold (proportional)

    THRESHOLD RATIONALE (50N):
        - UR10e + DG5F hand weight: ~3kg hand + ~0.5kg ATI = ~3.5kg
        - Static force (gravity): 3.5kg × 9.8 = ~34N
        - Threshold = 50N allows 16N margin above static load
        - Collision forces typically >> 50N, so this catches them
        - Forge uses [5-10N] but for lighter Franka gripper (~0.7kg)

    IMPORTANT: This function reuses the force computed in ati_force_sensor()
    to avoid duplicate PhysX calls and ensure consistency.

    Args:
        env: The environment.
        asset_cfg: Robot asset configuration
        sensor_body_name: ATI sensor body name
        force_threshold: Force magnitude threshold in Newtons (default: 50.0N)

    Returns:
        Penalty [batch]: 0 if force < threshold, positive if excessive
    """
    # =========================================================================
    # REUSE CACHED FORCE FROM OBSERVATION
    # The observation function (ati_force_sensor) computes and caches:
    # - env._ati_force_local: Force in local frame [num_envs, 3]
    # This avoids duplicate PhysX calls and ensures obs/reward consistency
    # =========================================================================
    if hasattr(env, '_ati_force_local') and env._ati_force_local is not None:
        # Use cached local frame force from observation
        force_local = env._ati_force_local
    else:
        # Fallback: compute force if observation hasn't run yet
        # This shouldn't happen in normal operation order (obs → reward)
        robot: Articulation = env.scene[asset_cfg.name]
        cache_key = f"_ati_sensor_body_idx_{sensor_body_name}"

        if not hasattr(env, cache_key):
            body_idx = robot.body_names.index(sensor_body_name)
            setattr(env, cache_key, body_idx)
            env._ati_force_smooth = None

        body_idx = getattr(env, cache_key)

        # Read from PhysX
        all_forces = robot.root_physx_view.get_link_incoming_joint_force()
        force_torque_world = all_forces[:, body_idx, :]

        # NaN/Inf safety check
        nan_mask = torch.isnan(force_torque_world) | torch.isinf(force_torque_world)
        if nan_mask.any():
            force_torque_world = torch.where(nan_mask, torch.zeros_like(force_torque_world), force_torque_world)

        # Buffer initialization & reset
        # CHANGED (2025-11-23): Initialize with mean force (~80N) instead of first reading
        EXPECTED_MEAN_FORCE = 80.0
        if env._ati_force_smooth is None:
            env._ati_force_smooth = torch.zeros_like(force_torque_world)
            env._ati_force_smooth[:, 2] = -EXPECTED_MEAN_FORCE

        reset_mask = env.episode_length_buf == 0
        if reset_mask.any():
            env._ati_force_smooth[reset_mask, :] = 0.0
            env._ati_force_smooth[reset_mask, 2] = -EXPECTED_MEAN_FORCE

        # EMA smoothing (0.25 = Forge default)
        smoothing_factor = 0.25
        env._ati_force_smooth = (
            smoothing_factor * force_torque_world +
            (1 - smoothing_factor) * env._ati_force_smooth
        )

        # Transform to local frame
        sensor_quat_w = robot.data.body_quat_w[:, body_idx, :]
        force_world = env._ati_force_smooth[:, 0:3]
        force_local_raw = quat_apply_inverse(sensor_quat_w, force_world)

        # =====================================================================
        # REJECT OUTLIERS (CONSISTENCY WITH OBSERVATION)
        # MODIFIED (2025-12-01): Clamping → Outlier rejection
        # Observation과 동일한 방식으로 outlier rejection 적용
        # 이유: Observation-Reward consistency 확보
        # Threshold: 5000N (극단적인 outlier만 rejection)
        # =====================================================================
        FORCE_OUTLIER_THRESHOLD = 5000.0  # Newtons - MODIFIED (2025-12-01): 250 → 5000
        force_local = reject_force_outliers(
            env,
            force_local_raw,
            threshold=FORCE_OUTLIER_THRESHOLD,
            buffer_key="_ati_force_valid_reward"  # Separate buffer from observation
        )

    # =========================================================================
    # COMPUTE PENALTY
    # Reference: forge_env.py:238-239
    # ReLU ensures penalty is 0 when below threshold, linear above
    # =========================================================================
    force_mag = torch.norm(force_local, p=2, dim=-1)

    # MODIFIED (2025-11-30): Use randomized threshold from observation
    # If ati_force_threshold observation is used, env._ati_force_threshold_buf exists
    if use_randomized_threshold and hasattr(env, '_ati_force_threshold_buf'):
        force_threshold = env._ati_force_threshold_buf  # [num_envs]
    else:
        # Fallback: Use default 150.0N if randomization not enabled
        force_threshold = 150.0

    penalty = torch.nn.functional.relu(force_mag - force_threshold)

    # NOTE: Forge normalizes by threshold: penalty / threshold
    # We don't normalize here - let the reward weight handle scaling
    # This gives more direct control via weight parameter

    return penalty


def ati_force_magnitude(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_body_name: str = "ATI_Axia90_M50",
) -> torch.Tensor:
    """Return ATI force magnitude for Tensorboard tracking (not a real reward).

    CREATED (2025-11-30): FT sensor 값 tracking용 메트릭

    사용자 요구사항:
    - FT sensor 값이 어떻게 나오는지 실시간 tracking
    - 물체 잡을 때 vs 손-바닥 충돌 시 force 값 차이 확인
    - Tensorboard에서 시각화하여 threshold 적절성 판단

    사용법:
    - env_cfg에서 weight=0으로 설정 → 실제 reward에 영향 없음
    - Tensorboard에만 기록되어 force magnitude 모니터링 가능

    Args:
        env: The environment.
        asset_cfg: Robot asset configuration
        sensor_body_name: ATI sensor body name

    Returns:
        Force magnitude [batch]: ||force|| in Newtons
    """
    # Reuse cached force from observation (same as ati_excessive_force_penalty)
    if hasattr(env, '_ati_force_local') and env._ati_force_local is not None:
        force_local = env._ati_force_local
    else:
        # Fallback: Return zeros if observation hasn't run yet
        return torch.zeros(env.num_envs, device=env.device)

    # Compute and return magnitude
    force_mag = torch.norm(force_local, p=2, dim=-1)
    return force_mag


def ati_downward_force_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_body_name: str = "ATI_Axia90_M50",
    z_threshold: float = -60.0,  # CHANGED: -10.0 → -60.0 (local frame, hand weight ~34N)
) -> torch.Tensor:
    """Penalize downward force in LOCAL frame (hand pressing on surfaces).

    USE CASE: Detect when hand is pressing down on table or object
    - Local -Z axis = "down" relative to ATI sensor
    - Large negative Fz = hand pushing downward

    NOTE: In local frame, hand weight (~34N) always appears as -Z force.
    Threshold -60N = 34N static + 26N margin for movement dynamics.
    Only triggers on actual pressing (collision with table).

    Args:
        env: The environment.
        asset_cfg: Robot asset configuration
        sensor_body_name: ATI sensor body name
        z_threshold: Downward force threshold in local frame (negative)

    Returns:
        Penalty [batch]: 0 if OK, positive if pressing downward excessively
    """
    # Reuse cached force from observation
    if hasattr(env, '_ati_force_local') and env._ati_force_local is not None:
        force_local = env._ati_force_local
    else:
        # Fallback computation (same pattern as ati_excessive_force_penalty)
        robot: Articulation = env.scene[asset_cfg.name]
        cache_key = f"_ati_sensor_body_idx_{sensor_body_name}"

        if not hasattr(env, cache_key):
            body_idx = robot.body_names.index(sensor_body_name)
            setattr(env, cache_key, body_idx)
            env._ati_force_smooth = None

        body_idx = getattr(env, cache_key)

        all_forces = robot.root_physx_view.get_link_incoming_joint_force()
        force_torque_world = all_forces[:, body_idx, :]

        # NaN/Inf safety check
        nan_mask = torch.isnan(force_torque_world) | torch.isinf(force_torque_world)
        if nan_mask.any():
            force_torque_world = torch.where(nan_mask, torch.zeros_like(force_torque_world), force_torque_world)

        # CHANGED (2025-11-23): Initialize with mean force (~80N) instead of first reading
        EXPECTED_MEAN_FORCE = 80.0
        if env._ati_force_smooth is None:
            env._ati_force_smooth = torch.zeros_like(force_torque_world)
            env._ati_force_smooth[:, 2] = -EXPECTED_MEAN_FORCE

        reset_mask = env.episode_length_buf == 0
        if reset_mask.any():
            env._ati_force_smooth[reset_mask, :] = 0.0
            env._ati_force_smooth[reset_mask, 2] = -EXPECTED_MEAN_FORCE

        smoothing_factor = 0.25
        env._ati_force_smooth = (
            smoothing_factor * force_torque_world +
            (1 - smoothing_factor) * env._ati_force_smooth
        )

        sensor_quat_w = robot.data.body_quat_w[:, body_idx, :]
        force_world = env._ati_force_smooth[:, 0:3]
        force_local = quat_apply_inverse(sensor_quat_w, force_world)

    # Get local Z-axis force (negative = pressing down)
    force_z = force_local[:, 2]

    # Penalty for excessive downward force
    # ReLU(-force_z - abs(threshold)) = penalty when pressing harder than threshold
    penalty = torch.nn.functional.relu(-force_z - abs(z_threshold))

    return penalty