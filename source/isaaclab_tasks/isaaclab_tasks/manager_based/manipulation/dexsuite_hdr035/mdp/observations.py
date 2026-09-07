# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul, subtract_frame_transforms

from .utils import reject_force_outliers, sample_object_point_cloud

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Object position in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 3)``: object position [x, y, z] expressed in the robot root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, object.data.root_pos_w - robot.data.root_pos_w)


def object_quat_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object orientation in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 4)``: object quaternion ``(w, x, y, z)`` in the robot root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_mul(quat_inv(robot.data.root_quat_w), object.data.root_quat_w)


def body_state_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body state (pos, quat, lin vel, ang vel) in the base asset's root frame.

    The state for each body is stacked horizontally as
    ``[position(3), quaternion(4)(wxyz), linvel(3), angvel(3)]`` and then concatenated over bodies.

    Args:
        env: The environment.
        body_asset_cfg: Scene entity for the articulated body whose links are observed.
        base_asset_cfg: Scene entity providing the reference (root) frame.

    Returns:
        Tensor of shape ``(num_envs, num_bodies * 13)`` with per-body states expressed in the base root frame.
    """
    body_asset: Articulation = env.scene[body_asset_cfg.name]
    base_asset: Articulation = env.scene[base_asset_cfg.name]
    # get world pose of bodies
    body_pos_w = body_asset.data.body_pos_w[:, body_asset_cfg.body_ids].view(-1, 3)
    body_quat_w = body_asset.data.body_quat_w[:, body_asset_cfg.body_ids].view(-1, 4)
    body_lin_vel_w = body_asset.data.body_lin_vel_w[:, body_asset_cfg.body_ids].view(-1, 3)
    body_ang_vel_w = body_asset.data.body_ang_vel_w[:, body_asset_cfg.body_ids].view(-1, 3)
    num_bodies = int(body_pos_w.shape[0] / env.num_envs)
    # get world pose of base frame
    root_pos_w = base_asset.data.root_link_pos_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 3)
    root_quat_w = base_asset.data.root_link_quat_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 4)
    # transform from world body pose to local body pose
    body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    body_lin_vel_b = quat_apply_inverse(root_quat_w, body_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(root_quat_w, body_ang_vel_w)
    # concate and return
    out = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=1)
    return out.view(env.num_envs, -1)


class object_point_cloud_b(ManagerTermBase):
    """Object surface point cloud expressed in a reference asset's root frame.

    Points are pre-sampled on the object's surface in its local frame and transformed to world,
    then into the reference (e.g., robot) root frame. Optionally visualizes the points.

    Args (from ``cfg.params``):
        object_cfg: Scene entity for the object to sample. Defaults to ``SceneEntityCfg("object")``.
        ref_asset_cfg: Scene entity providing the reference frame. Defaults to ``SceneEntityCfg("robot")``.
        num_points: Number of points to sample on the object surface. Defaults to ``10``.
        visualize: Whether to draw markers for the points. Defaults to ``True``.
        static: If ``True``, cache world-space points on reset and reuse them (no per-step resampling).

    Returns (from ``__call__``):
        If ``flatten=False``: tensor of shape ``(num_envs, num_points, 3)``.
        If ``flatten=True``: tensor of shape ``(num_envs, 3 * num_points)``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        num_points: int = cfg.params.get("num_points", 10)
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]
        # lazy initialize visualizer and point cloud
        if cfg.params.get("visualize", True):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

            ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationPointCloud")
            ray_cfg.markers["hit"].radius = 0.0025
            self.visualizer = VisualizationMarkers(ray_cfg)
        self.points_local = sample_object_point_cloud(
            env.num_envs, num_points, self.object.cfg.prim_path, device=env.device
        )
        self.points_w = torch.zeros_like(self.points_local)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        num_points: int = 10,
        flatten: bool = False,
        visualize: bool = True,
    ):
        """Compute the object point cloud in the reference asset's root frame.

        Note:
            Points are pre-sampled at initialization using ``self.num_points``; the ``num_points`` argument is
            kept for API symmetry and does not change the sampled set at runtime.

        Args:
            env: The environment.
            ref_asset_cfg: Reference frame provider (root). Defaults to ``SceneEntityCfg("robot")``.
            object_cfg: Object to sample. Defaults to ``SceneEntityCfg("object")``.
            num_points: Unused at runtime; see note above.
            flatten: If ``True``, return a flattened tensor ``(num_envs, 3 * num_points)``.
            visualize: If ``True``, draw markers for the points.

        Returns:
            Tensor of shape ``(num_envs, num_points, 3)`` or flattened if requested.
        """
        ref_pos_w = self.ref_asset.data.root_pos_w.unsqueeze(1).repeat(1, num_points, 1)
        ref_quat_w = self.ref_asset.data.root_quat_w.unsqueeze(1).repeat(1, num_points, 1)

        object_pos_w = self.object.data.root_pos_w.unsqueeze(1).repeat(1, num_points, 1)
        object_quat_w = self.object.data.root_quat_w.unsqueeze(1).repeat(1, num_points, 1)
        # apply rotation + translation
        self.points_w = quat_apply(object_quat_w, self.points_local) + object_pos_w
        if visualize:
            self.visualizer.visualize(translations=self.points_w.view(-1, 3))
        object_point_cloud_pos_b, _ = subtract_frame_transforms(ref_pos_w, ref_quat_w, self.points_w, None)

        return object_point_cloud_pos_b.view(env.num_envs, -1) if flatten else object_point_cloud_pos_b


def fingers_contact_force_b(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """base-frame contact forces from listed sensors, concatenated per env.

    Args:
        env: The environment.
        contact_sensor_names: Names of contact sensors in ``env.scene.sensors`` to read.

    Returns:
        Tensor of shape ``(num_envs, 3 * num_sensors)`` with forces stacked horizontally as
        ``[fx, fy, fz]`` per sensor.
    """
    force_w = [env.scene.sensors[name].data.force_matrix_w.view(env.num_envs, 3) for name in contact_sensor_names]
    force_w = torch.stack(force_w, dim=1)
    robot: Articulation = env.scene[asset_cfg.name]
    forces_b = quat_apply_inverse(robot.data.root_link_quat_w.unsqueeze(1).repeat(1, force_w.shape[1], 1), force_w)
    return forces_b


# ============================================================================
# ATI F/T SENSOR OBSERVATIONS (Forge approach - Improved)
# ============================================================================
#
# REFERENCE: isaaclab_tasks/direct/forge/forge_env.py
#
# KEY IMPROVEMENTS over original implementation:
# 1. smoothing_factor: 0.9 → 0.25 (Forge uses 0.25 for stability)
# 2. Buffer reset: Added episode reset handling to prevent NaN propagation
# 3. Local frame: Transform from World → ATI sensor local frame
# 4. Threshold observation: Include force threshold in observation (Forge style)
#
# ISSUE HISTORY (2025-11-23):
# - Original: smoothing_factor=0.9 caused noisy readings
# - Original: No buffer reset → NaN propagation across episodes
# - Original: World frame forces → inconsistent meaning when robot rotates
# - Result: abnormal_robot termination 100%, episode_length ~2.4 steps
#
# FUTURE IMPROVEMENTS (TODO):
# - [ ] Phase 2: Add randomized threshold (Forge: [5.0, 10.0] range)
# - [ ] Phase 2: Expand to 6D (Force + Torque) for grasp quality assessment
# - [ ] Phase 2: Add torque-based slip detection reward
# ============================================================================


def ati_force_sensor(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_body_name: str = "ATI_Axia90_M50",
    smoothing_factor: float = 0.25,  # CHANGED: 0.9 → 0.25 (Forge default)
) -> torch.Tensor:
    """Read ATI F/T sensor force using PhysX (Forge approach - Improved).

    REFERENCE: forge_env.py:95-101, forge_env_cfg.py:100 (ft_smoothing_factor=0.25)

    Uses get_link_incoming_joint_force() to read joint forces from PhysX.
    Applies exponential moving average (EMA) smoothing for stability.
    Transforms to LOCAL FRAME (ATI sensor reference) for consistent interpretation.

    EMA Formula: smoothed = α * current + (1-α) * previous
    - α = 0.25: 25% new value, 75% previous → smooth, stable signal
    - α = 0.9: 90% new value, 10% previous → noisy, responsive (BAD for our use case)

    Args:
        env: The environment.
        asset_cfg: Robot asset configuration
        sensor_body_name: Name of ATI sensor body in USD (default: "ATI_Axia90_M50")
        smoothing_factor: EMA alpha (default: 0.25, Forge reference)

    Returns:
        Force in LOCAL frame [batch, 3]: (Fx, Fy, Fz)
        - Local Z-axis: Force along sensor axis (typically downward = negative Z)
    """
    robot: Articulation = env.scene[asset_cfg.name]

    # =========================================================================
    # INITIALIZATION: Cache body index and create EMA buffer
    # =========================================================================
    cache_key = f"_ati_sensor_body_idx_{sensor_body_name}"
    if not hasattr(env, cache_key):
        try:
            body_idx = robot.body_names.index(sensor_body_name)
            setattr(env, cache_key, body_idx)
            # Initialize EMA buffer with None (will be set to first reading)
            # This prevents startup transient from zero initialization
            env._ati_force_smooth = None
            env._ati_force_computed_this_step = False  # Flag for reward function
            env._ati_debug_step_count = 0  # DEBUG counter
        except ValueError:
            raise ValueError(
                f"ATI sensor body '{sensor_body_name}' not found in robot.body_names. "
                f"Available bodies: {robot.body_names}"
            )

    body_idx = getattr(env, cache_key)

    # =========================================================================
    # READ FORCE/TORQUE FROM PHYSX
    # get_link_incoming_joint_force() returns [num_envs, num_bodies, 6]
    # where 6 = (Fx, Fy, Fz, Tx, Ty, Tz) in PARENT BODY FRAME (NOT world!)
    # IMPORTANT: "parent body" = Joint body0 (wrist_3 for ATI), NOT world!
    # =========================================================================
    all_forces = robot.root_physx_view.get_link_incoming_joint_force()
    force_torque_world = all_forces[:, body_idx, :]  # [num_envs, 6] - MISLEADING NAME!

    # =========================================================================
    # DEBUG (ADDED 2025-12-01): Track PhysX raw force BEFORE any processing
    # Purpose: Identify if extreme values come from PhysX or our processing
    # =========================================================================
    if not hasattr(env, '_debug_physx_raw_force'):
        env._debug_physx_raw_force = None
    env._debug_physx_raw_force = force_torque_world[:, 0:3].clone()  # Store for later debug

    # =========================================================================
    # NaN/Inf SAFETY CHECK
    # PhysX can return NaN/Inf on simulation instability
    # Replace with zeros to prevent propagation
    # =========================================================================
    nan_mask = torch.isnan(force_torque_world) | torch.isinf(force_torque_world)
    if nan_mask.any():
        force_torque_world = torch.where(nan_mask, torch.zeros_like(force_torque_world), force_torque_world)
        # DEBUG: Log NaN occurrences (first 10 times only)
        if not hasattr(env, '_ati_nan_count'):
            env._ati_nan_count = 0
        if env._ati_nan_count < 10:
            print(f"[WARNING] ATI F/T sensor: NaN/Inf detected and replaced with zeros")
            env._ati_nan_count += 1

    # =========================================================================
    # BUFFER INITIALIZATION & RESET
    # CHANGED (2025-11-23): Initialize with mean force value (~80N) instead of first reading
    # Reason: First reading can be extreme outlier (2000-8000N), corrupting EMA buffer
    # Mean force ~80N is typical static load (DG5F hand weight ~5kg × gravity)
    # =========================================================================
    EXPECTED_MEAN_FORCE = 80.0  # Typical mean force in Newtons

    if env._ati_force_smooth is None:
        # First call ever - initialize with expected mean force (not first reading)
        # This prevents extreme outliers from corrupting buffer from the start
        env._ati_force_smooth = torch.zeros_like(force_torque_world)
        env._ati_force_smooth[:, 2] = -EXPECTED_MEAN_FORCE  # Typical -Z force (gravity)

    # Reset buffer for environments that just started new episode
    # Use expected mean force, not current reading (which may be outlier)
    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
        env._ati_force_smooth[reset_mask, :] = 0.0
        env._ati_force_smooth[reset_mask, 2] = -EXPECTED_MEAN_FORCE

    # =========================================================================
    # EMA SMOOTHING
    # Reference: forge_env.py:100-101
    # alpha=0.25 means: new_smooth = 0.25*current + 0.75*previous
    # This provides stable readings while still being responsive to changes
    # =========================================================================
    env._ati_force_smooth = (
        smoothing_factor * force_torque_world +
        (1 - smoothing_factor) * env._ati_force_smooth
    )

    # =========================================================================
    # DEBUG (ADDED 2025-12-01): Track EMA smoothed force
    # =========================================================================
    if not hasattr(env, '_debug_ema_smoothed_force'):
        env._debug_ema_smoothed_force = None
    env._debug_ema_smoothed_force = env._ati_force_smooth[:, 0:3].clone()

    # =========================================================================
    # TRANSFORM TO LOCAL FRAME (ATI sensor reference)
    # Why local frame?
    # - World frame: "Down" direction changes with robot pose
    # - Local frame: "Down" is always sensor's -Z axis, regardless of robot pose
    #
    # Get ATI sensor body orientation and transform force vector
    # =========================================================================
    sensor_quat_w = robot.data.body_quat_w[:, body_idx, :]  # [num_envs, 4]
    force_world = env._ati_force_smooth[:, 0:3]  # [num_envs, 3]

    # Transform: Parent body frame → Local frame (inverse rotation)
    # NOTE: Despite variable name "force_world", this is actually parent body frame!
    force_local_raw = quat_apply_inverse(sensor_quat_w, force_world)

    # =========================================================================
    # REJECT FORCE OUTLIERS (REPLACE WITH PREVIOUS VALID VALUES)
    # MODIFIED (2025-12-01): Magnitude clamping → Outlier rejection
    #
    # Problem:
    #   - PhysX returns extreme constraint forces (220,000N) from ~0.5% envs
    #   - Caused by: Fixed joint instability, DR extremes, spawn collision
    #   - Clamping still corrupts RL normalization & policy
    #
    # Solution:
    #   - Detect outliers: magnitude > 5000N (극단적인 값만)
    #   - Replace with previous valid value (NOT clamp)
    #   - Isolate and neutralize outlier envs from affecting policy
    #
    # Threshold Rationale (5000N):
    #   - Normal grasp: 50-500N → 통과 ✓
    #   - Strong contact: 500-2000N → 통과 ✓
    #   - Abnormal spike: 5000-10000N → REJECTED
    #   - Extreme outlier: 220,000N → REJECTED
    #   - Previous 250N was TOO LOW, rejected normal strong grasps!
    #
    # Debug Results (2025-12-01):
    #   - PhysX_raw_max: 220,000N (extreme constraint force)
    #   - Affected envs: ~18/4096 (0.44%)
    #   - Normal envs mean: 3-4N
    #
    # Reference: utils.reject_force_outliers() for detailed logic
    # =========================================================================
    FORCE_OUTLIER_THRESHOLD = 5000.0  # Newtons - MODIFIED (2025-12-01): 250 → 5000

    force_local = reject_force_outliers(
        env,
        force_local_raw,
        threshold=FORCE_OUTLIER_THRESHOLD,
        buffer_key="_ati_force_valid"
    )

    # Keep raw magnitude calculation for debug tracking
    force_mag_raw = torch.norm(force_local_raw, dim=-1, keepdim=True)

    # =========================================================================
    # DEBUG: Log force values periodically (every 5 steps) to file
    # MODIFIED (2025-12-01): Added multi-stage tracking to identify extreme value source
    # =========================================================================
    env._ati_debug_step_count += 1
    if env._ati_debug_step_count % 5 == 0:
        # =====================================================================
        # STAGE 1: PhysX Raw (before any processing)
        # =====================================================================
        physx_raw_mag = torch.norm(env._debug_physx_raw_force, dim=-1)  # [num_envs]

        # =====================================================================
        # STAGE 2: EMA Smoothed (after smoothing, before frame transform)
        # =====================================================================
        ema_smoothed_mag = torch.norm(env._debug_ema_smoothed_force, dim=-1)  # [num_envs]

        # =====================================================================
        # STAGE 3: Frame Transformed (after local frame conversion)
        # =====================================================================
        force_mag_raw_flat = force_mag_raw.squeeze(-1)  # [num_envs]

        # =====================================================================
        # STAGE 4: Outlier Rejected (final observation value)
        # =====================================================================
        force_mag_rejected = torch.norm(force_local, dim=-1)  # [num_envs]

        # Count environments with outliers rejected (raw > threshold)
        rejected_count = (force_mag_raw_flat > FORCE_OUTLIER_THRESHOLD).sum().item()

        # Check for reset events (spike indicator)
        reset_count = (env.episode_length_buf == 0).sum().item()

        # Multi-stage debug output
        log_msg = (f"[Step {env._ati_debug_step_count:>8}] "
                   f"PhysX_raw_max={physx_raw_mag.max().item():>8.1f}N, "
                   f"EMA_max={ema_smoothed_mag.max().item():>8.1f}N, "
                   f"FrameTx_max={force_mag_raw_flat.max().item():>8.1f}N, "
                   f"Rejected_max={force_mag_rejected.max().item():>6.1f}N, "
                   f"mean={force_mag_rejected.mean().item():>5.1f}N, "
                   f"rejected={rejected_count:>3}/{env.num_envs}, "
                   f"resets={reset_count:>3}")

        # Print to terminal
        print(f"[DEBUG ATI Force] {log_msg}")

        # Save to file
        import os
        from datetime import datetime
        log_dir = "/home/dyros/IsaacLab/logs/rl_games/dexsuite_ur10e_dg5f_right"
        os.makedirs(log_dir, exist_ok=True)

        # Create log file with date (one file per day)
        log_file = os.path.join(log_dir, f"ati_force_debug_{datetime.now().strftime('%Y%m%d')}.txt")

        # Write header if file is new
        if not hasattr(env, '_ati_log_file_initialized'):
            with open(log_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"ATI Force Debug Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n")
            env._ati_log_file_initialized = True

        # Append log entry
        with open(log_file, 'a') as f:
            f.write(f"{log_msg}\n")

    # Mark that force was computed this step (for reward function to reuse)
    env._ati_force_computed_this_step = True
    env._ati_force_local = force_local  # Cache for reward function

    return force_local


def ati_force_threshold(
    env: ManagerBasedRLEnv,
    threshold_range: tuple[float, float] = (150.0, 200.0),  # MODIFIED (2025-11-30): Phase 2 구현, randomization 활성화
) -> torch.Tensor:
    """Return the ATI force threshold as observation (Forge style - with randomization).

    REFERENCE: forge_env.py:129 - "force_threshold": self.contact_penalty_thresholds[:, None]
              forge_env.py:319-321 - contact_rand + interpolation

    WHY INCLUDE THRESHOLD IN OBSERVATION?
    - Policy knows "how much force is acceptable"
    - Enables adaptation to different threshold settings
    - Essential when using randomized thresholds

    MODIFICATION HISTORY:
    - Original: 50.0N (고정값)
    - Phase 1 (11290200): 150.0N (고정값, reward threshold와 일치)
    - MODIFIED (2025-11-30): [150.0, 200.0]N (randomization, Phase 2 구현)

    RATIONALE (2025-11-30):
    문제: 고정 threshold 150.0N이 너무 낮음
      - Tensorboard 분석: ati_excessive_force_penalty = -0.02 (매 스텝마다 발생)
      - 의미: force가 지속적으로 ~154N (threshold 150N 초과)
      - 결과: Policy가 접촉 회피 학습 (good_finger_contact ≈ -0.03)

    해결책: Threshold randomization [150, 200]N
      - Episode마다 다른 threshold로 학습 → robust policy
      - 평균 threshold ~175N → 정상 grasping (110-130N) 허용
      - Forge 방식 (forge_env.py:319-321) 적용

    Args:
        env: The environment.
        threshold_range: (min, max) threshold range in Newtons (default: (150.0, 200.0))

    Returns:
        Threshold tensor [batch, 1]: Randomized per env, persists within episode
    """
    # MODIFIED (2025-11-30): Phase 2 구현 - Randomized threshold per environment
    # Initialize threshold buffer on first call
    if not hasattr(env, '_ati_force_threshold_buf'):
        # Randomize initial thresholds using Forge method
        # Reference: forge_env.py:319-321
        contact_rand = torch.rand((env.num_envs,), device=env.device)
        env._ati_force_threshold_buf = threshold_range[0] + contact_rand * (threshold_range[1] - threshold_range[0])

    # Re-randomize thresholds for environments that just reset
    # IMPORTANT: Episode reset 시에만 threshold 변경, episode 중에는 고정
    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
        contact_rand = torch.rand((reset_mask.sum().item(),), device=env.device)
        new_thresholds = threshold_range[0] + contact_rand * (threshold_range[1] - threshold_range[0])
        env._ati_force_threshold_buf[reset_mask] = new_thresholds

    # Return as [num_envs, 1] for observation
    return env._ati_force_threshold_buf.unsqueeze(1)


def ati_force_torque_sensor(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_body_name: str = "ATI_Axia90_M50",
    smoothing_factor: float = 0.25,  # CHANGED: 0.9 → 0.25 (Forge default)
) -> torch.Tensor:
    """Read ATI F/T sensor force AND torque - 6D version (for future use).

    REFERENCE: forge_env.py (uses 3D only, but 6D available)

    NOTE: Currently using 3D (force only) for simplicity.
    This 6D function is for FUTURE expansion:
    - Torque can indicate grasp quality
    - Object slipping causes torque changes
    - Could add torque-based slip detection reward

    Args:
        env: The environment.
        asset_cfg: Robot asset configuration
        sensor_body_name: Name of ATI sensor body in USD
        smoothing_factor: EMA alpha (default: 0.25)

    Returns:
        Force and torque in LOCAL frame [batch, 6]: (Fx, Fy, Fz, Tx, Ty, Tz)
    """
    robot: Articulation = env.scene[asset_cfg.name]

    # Initialize
    cache_key = f"_ati_sensor_body_idx_6d_{sensor_body_name}"  # Different key for 6D version
    if not hasattr(env, cache_key):
        try:
            body_idx = robot.body_names.index(sensor_body_name)
            setattr(env, cache_key, body_idx)
            env._ati_force_smooth_6d = None  # Separate buffer for 6D version
        except ValueError:
            raise ValueError(
                f"ATI sensor body '{sensor_body_name}' not found in robot.body_names. "
                f"Available bodies: {robot.body_names}"
            )

    body_idx = getattr(env, cache_key)

    # Read from PhysX
    all_forces = robot.root_physx_view.get_link_incoming_joint_force()
    force_torque_world = all_forces[:, body_idx, :]

    # NaN/Inf safety check
    nan_mask = torch.isnan(force_torque_world) | torch.isinf(force_torque_world)
    if nan_mask.any():
        force_torque_world = torch.where(nan_mask, torch.zeros_like(force_torque_world), force_torque_world)

    # Buffer initialization & reset
    if env._ati_force_smooth_6d is None:
        env._ati_force_smooth_6d = force_torque_world.clone()
    else:
        reset_mask = env.episode_length_buf == 0
        if reset_mask.any():
            env._ati_force_smooth_6d[reset_mask] = force_torque_world[reset_mask]

    # EMA smoothing
    env._ati_force_smooth_6d = (
        smoothing_factor * force_torque_world +
        (1 - smoothing_factor) * env._ati_force_smooth_6d
    )

    # Transform to local frame
    sensor_quat_w = robot.data.body_quat_w[:, body_idx, :]
    force_world = env._ati_force_smooth_6d[:, 0:3]
    torque_world = env._ati_force_smooth_6d[:, 3:6]

    force_local = quat_apply_inverse(sensor_quat_w, force_world)
    torque_local = quat_apply_inverse(sensor_quat_w, torque_world)

    return torch.cat([force_local, torque_local], dim=-1)