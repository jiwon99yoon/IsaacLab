# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_measured_torques(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """관절축에 투영된 측정 joint 토크 — FR3 내장 관절 토크 센서의 sim 대응물.

    실물 FR3는 중력보상 위에서 외란 토크 추정치(tau_ext)를 제공한다. 로봇 링크의 중력을 끈
    (disable_gravity=True) 환경에서는 이 측정값에 링크 자중이 빠져 있어 실물 tau_ext와
    성격이 맞는다. 접촉/payload/동역학 성분이 담기므로 contact-aware 정책 학습용.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    measured = asset.root_physx_view.get_dof_projected_joint_forces()
    return measured[:, asset_cfg.joint_ids]


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b
