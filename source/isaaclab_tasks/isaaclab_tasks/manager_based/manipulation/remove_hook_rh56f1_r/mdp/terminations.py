# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the remove_hook task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def out_of_bound(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    in_bound_range: dict[str, tuple[float, float]] = {},
) -> torch.Tensor:
    """Termination condition for the object falls out of bound.

    Args:
        env: The environment.
        asset_cfg: The object configuration. Defaults to SceneEntityCfg("object").
                   If body_names is specified, uses body position instead of root position.
        in_bound_range: The range in x, y, z such that the object is considered in range

    Note (Bug fix 2026-01-07):
        - 기존: root_pos_w만 체크 → hook(left_ring) body가 폭발해도 감지 못함
        - 수정: body_names 지정 시 body_pos_w 체크 → hook body position으로 termination 가능
    """
    object: Articulation = env.scene[asset_cfg.name]
    range_list = [in_bound_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)

    # body_names가 지정되어 있으면 body position 사용, 아니면 root position 사용
    if asset_cfg.body_ids is not None:
        # body_pos_w shape: (num_envs, num_bodies, 3)
        if isinstance(asset_cfg.body_ids, list):
            object_pos_w = object.data.body_pos_w[:, asset_cfg.body_ids, :].squeeze(1)
        elif isinstance(asset_cfg.body_ids, slice):
            object_pos_w = object.data.body_pos_w[:, asset_cfg.body_ids, :][:, 0, :]
        else:
            object_pos_w = object.data.body_pos_w[:, asset_cfg.body_ids, :]
    else:
        object_pos_w = object.data.root_pos_w

    object_pos_local = object_pos_w - env.scene.env_origins
    outside_bounds = ((object_pos_local < ranges[:, 0]) | (object_pos_local > ranges[:, 1])).any(dim=1)
    return outside_bounds


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminating environment when violation of velocity limits detects, this usually indicates unstable physics caused
    by very bad, or aggressive action"""
    robot: Articulation = env.scene[asset_cfg.name]
    # MODIFIED (2026-01-12): multiplier 1000 (임시 - 학습 우선)
    # TODO: 학습 완료 후 sim-to-real 전이 시 실제 로봇 속도 한계에 맞게 조정 필요
    # 주의: multiplier가 너무 크면 학습된 정책이 실제 로봇에서 위험할 수 있음
    return (robot.data.joint_vel.abs() > (robot.data.joint_vel_limits * 1000)).any(dim=1)