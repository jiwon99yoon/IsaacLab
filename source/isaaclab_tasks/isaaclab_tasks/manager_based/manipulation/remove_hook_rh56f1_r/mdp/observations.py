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

from .utils import sample_object_point_cloud

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
                    If body_names is specified (e.g., ["left_ring"]), samples from that body only
                    and uses body_pos_w/body_quat_w for transformation.
        ref_asset_cfg: Scene entity providing the reference frame. Defaults to ``SceneEntityCfg("robot")``.
        num_points: Number of points to sample on the object surface. Defaults to ``10``.
        visualize: Whether to draw markers for the points. Defaults to ``True``.
        static: If ``True``, cache world-space points on reset and reuse them (no per-step resampling).

    Returns (from ``__call__``):
        If ``flatten=False``: tensor of shape ``(num_envs, num_points, 3)``.
        If ``flatten=True``: tensor of shape ``(num_envs, 3 * num_points)``.

    Note (2026-01-07 Bug Fix):
        - 기존: object.data.root_pos_w 사용 → Wire root와 실제 mesh 위치 불일치 (점이 위로 뜸)
        - 수정: body_names 지정 시 body_pos_w/body_quat_w 사용 + 해당 body prim만 샘플링
        - 예: object_cfg=SceneEntityCfg("object", body_names=["left_ring"])
              → /Wire/left_ring* mesh만 샘플링, left_ring body pose로 transform
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        num_points: int = cfg.params.get("num_points", 10)
        self.num_points = num_points
        self.object: Articulation = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]

        # Determine if we're sampling from a specific body or the whole object
        # body_names가 지정되어 있으면 해당 body의 prim path만 샘플링
        self.use_body_pose = False
        sample_prim_path = self.object.cfg.prim_path

        if self.object_cfg.body_ids is not None:
            self.use_body_pose = True
            # Get body name from body_ids to construct prim path
            # body_names는 SceneEntityCfg에서 body_ids로 변환됨 (resolve 후)
            # 원래 body_names를 가져오기 위해 cfg.params에서 직접 확인
            body_names = cfg.params.get("object_cfg", SceneEntityCfg("object")).body_names
            if body_names:
                # Construct prim path for the specific body
                # e.g., "{ENV_REGEX_NS}/Wire" + "/left_ring" → "{ENV_REGEX_NS}/Wire/left_ring"
                body_name = body_names[0] if isinstance(body_names, list) else body_names
                # left_ring* 패턴으로 하위 mesh 모두 포함
                sample_prim_path = f"{self.object.cfg.prim_path}/{body_name}"
                print(f"[PointCloud] Sampling from body: {body_name}, prim_path: {sample_prim_path}")

        # lazy initialize visualizer and point cloud
        self.visualizer = None
        if cfg.params.get("visualize", True):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

            ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationPointCloud")
            ray_cfg.markers["hit"].radius = 0.0025
            self.visualizer = VisualizationMarkers(ray_cfg)

        self.points_local = sample_object_point_cloud(
            env.num_envs, num_points, sample_prim_path, device=env.device
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
        # Use actual num_points from initialization
        num_points = self.num_points

        ref_pos_w = self.ref_asset.data.root_pos_w.unsqueeze(1).repeat(1, num_points, 1)
        ref_quat_w = self.ref_asset.data.root_quat_w.unsqueeze(1).repeat(1, num_points, 1)

        # body_names가 지정되어 있으면 body_pos_w 사용, 아니면 root_pos_w 사용
        if self.use_body_pose and self.object_cfg.body_ids is not None:
            # body_pos_w shape: (num_envs, num_bodies, 3)
            # body_quat_w shape: (num_envs, num_bodies, 4)
            if isinstance(self.object_cfg.body_ids, list):
                object_pos_w = self.object.data.body_pos_w[:, self.object_cfg.body_ids, :].squeeze(1)
                object_quat_w = self.object.data.body_quat_w[:, self.object_cfg.body_ids, :].squeeze(1)
            elif isinstance(self.object_cfg.body_ids, slice):
                object_pos_w = self.object.data.body_pos_w[:, self.object_cfg.body_ids, :][:, 0, :]
                object_quat_w = self.object.data.body_quat_w[:, self.object_cfg.body_ids, :][:, 0, :]
            else:
                object_pos_w = self.object.data.body_pos_w[:, self.object_cfg.body_ids, :]
                object_quat_w = self.object.data.body_quat_w[:, self.object_cfg.body_ids, :]

            object_pos_w = object_pos_w.unsqueeze(1).repeat(1, num_points, 1)
            object_quat_w = object_quat_w.unsqueeze(1).repeat(1, num_points, 1)
        else:
            # 기존 방식: root pose 사용
            object_pos_w = self.object.data.root_pos_w.unsqueeze(1).repeat(1, num_points, 1)
            object_quat_w = self.object.data.root_quat_w.unsqueeze(1).repeat(1, num_points, 1)

        # apply rotation + translation
        self.points_w = quat_apply(object_quat_w, self.points_local) + object_pos_w
        if visualize and self.visualizer is not None:
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
# HOOK-SPECIFIC OBSERVATION FUNCTIONS (Remove Hook Task)
# ============================================================================


def hook_position_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hook_cfg: SceneEntityCfg = SceneEntityCfg("object", body_names=["right_ring"]),
) -> torch.Tensor:
    """
    Hook position in the robot's root frame.

    For Remove Hook task, the hook is part of the wire articulation (not chassis).
    This function tracks the hook's position for grasping and transport.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        hook_cfg: Scene entity for the hook object. Defaults to ``SceneEntityCfg("object", body_names=["right_ring"])``
                  which references the right_ring body within the wire articulation.

    Returns:
        Tensor of shape ``(num_envs, 3)``: hook position [x, y, z] in robot root frame.

    Note:
        The hook is accessed as a body of the wire articulation.
        In the USD: wire_model/right_ring (top-level body)
        IsaacLab only exposes top-level bodies, not nested Xforms.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    wire_object: Articulation = env.scene[hook_cfg.name]

    # Get hook position from wire articulation body pose data
    # body_pos_w has shape (num_envs, num_bodies, 3)
    # hook_cfg.body_ids specifies which body index is the hook
    # Handle slice, list, and int cases
    if isinstance(hook_cfg.body_ids, slice):
        # If body_ids is a slice, extract the first matched body
        hook_pos_w = wire_object.data.body_pos_w[:, hook_cfg.body_ids, :][:, 0, :]
    elif isinstance(hook_cfg.body_ids, list):
        # If body_ids is a list, extract and squeeze
        hook_pos_w = wire_object.data.body_pos_w[:, hook_cfg.body_ids, :].squeeze(1)
    else:
        # Single index
        hook_pos_w = wire_object.data.body_pos_w[:, hook_cfg.body_ids, :]

    # ========== DEBUG: Wire diagnostics (문제 있을 때 10 step마다 출력) ==========
    # 조건: velocity 폭발 (>10) 또는 NaN/Inf 값 감지 시, 10 step마다만 출력
    if isinstance(hook_cfg.body_ids, list):
        hook_lin_vel_w = wire_object.data.body_lin_vel_w[:, hook_cfg.body_ids, :].squeeze(1)
    elif isinstance(hook_cfg.body_ids, slice):
        hook_lin_vel_w = wire_object.data.body_lin_vel_w[:, hook_cfg.body_ids, :][:, 0, :]
    else:
        hook_lin_vel_w = wire_object.data.body_lin_vel_w[:, hook_cfg.body_ids, :]

    # 문제 감지: NaN/Inf 또는 velocity 폭발
    has_nan = not torch.isfinite(hook_pos_w).all()
    max_vel = torch.max(torch.abs(hook_lin_vel_w)).item()
    has_vel_explosion = max_vel > 10.0

    # 10 step마다만 출력 (env_0 기준)
    current_step = env.episode_length_buf[0].item()
    should_print = (current_step % 10 == 0)

    if (has_nan or has_vel_explosion) and should_print:
        # 문제 있는 env 찾기
        abnormal_envs = []
        for i in range(env.num_envs):
            pos = hook_pos_w[i]
            vel = hook_lin_vel_w[i]
            if not torch.isfinite(pos).all() or torch.norm(vel) > 10.0:
                abnormal_envs.append(i)

        print(f"\n[WARNING] Wire Abnormal (step {current_step}): "
              f"NaN={has_nan}, vel_max={max_vel:.1f} m/s, "
              f"affected_envs={abnormal_envs}")
    # ==================================================================

    return quat_apply_inverse(robot.data.root_quat_w, hook_pos_w - robot.data.root_pos_w)


def hook_orientation_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hook_cfg: SceneEntityCfg = SceneEntityCfg("object", body_names=["right_ring"]),
) -> torch.Tensor:
    """
    Hook orientation in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot. Defaults to ``SceneEntityCfg("robot")``.
        hook_cfg: Scene entity for the hook. Defaults to ``SceneEntityCfg("object", body_names=["right_ring"])``
                  which references the right_ring body within the wire articulation.

    Returns:
        Tensor of shape ``(num_envs, 4)``: hook quaternion (w, x, y, z) in robot root frame.

    Note:
        For Remove Hook task, hook orientation may not be critical (grasping focus is on position).
        This function is provided for completeness.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    wire_object: Articulation = env.scene[hook_cfg.name]

    # Get hook orientation from wire articulation body pose data
    # body_quat_w has shape (num_envs, num_bodies, 4)
    # Handle slice, list, and int cases
    if isinstance(hook_cfg.body_ids, slice):
        # If body_ids is a slice, extract the first matched body
        hook_quat_w = wire_object.data.body_quat_w[:, hook_cfg.body_ids, :][:, 0, :]
    elif isinstance(hook_cfg.body_ids, list):
        # If body_ids is a list, extract and squeeze
        hook_quat_w = wire_object.data.body_quat_w[:, hook_cfg.body_ids, :].squeeze(1)
    else:
        # Single index
        hook_quat_w = wire_object.data.body_quat_w[:, hook_cfg.body_ids, :]

    return quat_mul(quat_inv(robot.data.root_quat_w), hook_quat_w)


def spring_position_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    chassis_cfg: SceneEntityCfg = SceneEntityCfg("chassis_assembly"),
) -> torch.Tensor:
    """
    Spring position in the robot's root frame (for collision avoidance).

    In the Remove Hook task, the spring is part of the chassis assembly and is static.
    This observation helps the policy learn to avoid colliding with the spring while
    grasping the hook.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot. Defaults to ``SceneEntityCfg("robot")``.
        chassis_cfg: Scene entity for the chassis assembly (which includes springs).
                     Defaults to ``SceneEntityCfg("chassis_assembly")``.

    Returns:
        Tensor of shape ``(num_envs, 3)``: spring/chassis position in robot root frame.

    Note:
        Since spring is fixed relative to chassis, this returns chassis position.
        For finer control, specific spring prim could be tracked separately.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    chassis: RigidObject = env.scene[chassis_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, chassis.data.root_pos_w - robot.data.root_pos_w)