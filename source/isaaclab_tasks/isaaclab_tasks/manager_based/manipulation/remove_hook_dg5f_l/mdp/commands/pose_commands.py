# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# MODIFIED for Remove Hook Task (DG5F_L Variant):
# - Added support for tracking specific body (e.g., right_ring) instead of root
# - object_body_names parameter allows ArticulationCfg objects to be tracked properly
# - Falls back to root_state_w when object_body_names is None (original behavior)


"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import pose_commands_cfg as dex_cmd_cfgs


class ObjectUniformPoseCommand(CommandTerm):
    """Uniform pose command generator for an object (in the robot base frame).

    This command term samples target object poses by:
      • Drawing (x, y, z) uniformly within configured Cartesian bounds, and
      • Drawing roll-pitch-yaw uniformly within configured ranges, then converting
        to a quaternion (w, x, y, z). Optionally makes quaternions unique by enforcing
        a positive real part.

    Frames:
        Targets are defined in the robot's *base frame*. For metrics/visualization,
        targets are transformed into the *world frame* using the robot root pose.

    Outputs:
        The command buffer has shape (num_envs, 7): `(x, y, z, qw, qx, qy, qz)`.

    Metrics:
        `position_error` and `orientation_error` are computed between the commanded
        world-frame pose and the object's current world-frame pose.

    Config:
        `cfg` must provide the sampling ranges, whether to enforce quaternion uniqueness,
        and optional visualization settings.

    MODIFIED for Remove Hook Task (DG5F_L):
        - Added object_body_names support to track specific body instead of root.
        - For wire articulation, tracks "right_ring" body position/orientation.
        - NOTE: DG5F_L (left hand) targets right_ring (vs RH56F1_R targets left_ring)
    """

    cfg: dex_cmd_cfgs.ObjectUniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: dex_cmd_cfgs.ObjectUniformPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object = env.scene[cfg.object_name]  # Can be RigidObject or Articulation

        # MODIFIED: Handle object_body_names for tracking specific body
        self.use_body_tracking = cfg.object_body_names is not None
        if self.use_body_tracking:
            # Resolve body indices for the specified body names
            self.object_body_ids = self.object.find_bodies(cfg.object_body_names)[0]
            if len(self.object_body_ids) == 0:
                raise ValueError(
                    f"No bodies found matching '{cfg.object_body_names}' in object '{cfg.object_name}'. "
                    f"Available bodies: {self.object.body_names}"
                )
            # Use first body if multiple specified
            self.object_body_id = self.object_body_ids[0]

        # MODIFIED: success_vis_asset is now optional (None for remove_hook task without table)
        self.success_vis_asset = None
        if cfg.success_vis_asset_name is not None:
            self.success_vis_asset = env.scene[cfg.success_vis_asset_name]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        # MODIFIED: Only create success visualizer if success_vis_asset exists
        if self.success_vis_asset is not None:
            self.success_visualizer = VisualizationMarkers(self.cfg.success_visualizer_cfg)
            self.success_visualizer.set_visibility(True)
        else:
            self.success_visualizer = None

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )

        # MODIFIED: Get object position/orientation based on tracking mode
        if self.use_body_tracking:
            # Track specific body (e.g., right_ring for wire articulation)
            object_pos_w = self.object.data.body_pos_w[:, self.object_body_id, :]
            object_quat_w = self.object.data.body_quat_w[:, self.object_body_id, :]
        else:
            # Original behavior: track root state (for RigidObject)
            object_pos_w = self.object.data.root_state_w[:, :3]
            object_quat_w = self.object.data.root_state_w[:, 3:7]

        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            object_pos_w,
            object_quat_w,
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

        # MODIFIED: Only visualize success if success_vis_asset exists
        if self.success_visualizer is not None and self.success_vis_asset is not None:
            success_id = self.metrics["position_error"] < 0.05
            if not self.cfg.position_only:
                success_id &= self.metrics["orientation_error"] < 0.5
            self.success_visualizer.visualize(self.success_vis_asset.data.root_pos_w, marker_indices=success_id.int())

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                # -- goal pose
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.curr_visualizer = VisualizationMarkers(self.cfg.curr_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_visualizer.set_visibility(True)
            self.curr_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)
                self.curr_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return

        # MODIFIED: Get object position/orientation based on tracking mode
        if self.use_body_tracking:
            object_pos_w = self.object.data.body_pos_w[:, self.object_body_id, :]
            object_quat_w = self.object.data.body_quat_w[:, self.object_body_id, :]
        else:
            object_pos_w = self.object.data.root_pos_w
            object_quat_w = self.object.data.root_quat_w

        # update the markers
        if not self.cfg.position_only:
            # -- goal pose
            self.goal_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
            # -- current object pose
            self.curr_visualizer.visualize(object_pos_w, object_quat_w)
        else:
            distance = torch.norm(self.pose_command_w[:, :3] - object_pos_w[:, :3], dim=1)
            success_id = (distance < 0.05).int()
            # note: since marker indices for position is 1(far) and 2(near), we can simply shift the success_id by 1.
            # -- goal position
            self.goal_visualizer.visualize(self.pose_command_w[:, :3], marker_indices=success_id + 1)
            # -- current object position
            self.curr_visualizer.visualize(object_pos_w, marker_indices=success_id + 1)
