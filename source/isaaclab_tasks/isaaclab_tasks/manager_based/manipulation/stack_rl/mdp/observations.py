# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
20251211 추가: Stack RL 환경을 위한 상대 좌표 observation 함수들

이유: IsaacGym 분석 결과, 절대 좌표(world frame) 대신 상대 좌표 사용이 더 효과적
      - IsaacGym: cubeA_to_cubeB_pos (cube 간 상대 벡터) 사용
      - IsaacLab: object_position_in_robot_root_frame (robot 기준) 사용
      - 우리: 절대 좌표만 사용 → Policy가 관계를 학습하기 어려움

해결: 상대 벡터 observation 추가
      - cube_2_to_cube_1_pos: cube_2에서 cube_1으로의 벡터 (Phase 1에 직접 사용)
      - cube_3_to_cube_2_pos: cube_3에서 cube_2로의 벡터 (Phase 2에 직접 사용)

참고: DIFFERENCE_LIFT_AND_GYM_STACK_AND_WHATTODO_IN_OUR.md
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_relative_positions(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    20251211 추가: 큐브 간 상대 위치 벡터 반환 (6-dim)

    Returns:
        torch.Tensor: [N, 6] - cube_2_to_cube_1 (3) + cube_3_to_cube_2 (3)

    이유:
        - IsaacGym 성공 요인: cubeA_to_cubeB_pos 사용 (task-specific, 직접적)
        - 절대 좌표는 policy가 관계를 학습해야 하지만, 상대 벡터는 바로 사용 가능
        - Phase 1: cube_2를 cube_1 위에 올리기 → cube_2_to_cube_1 직접 사용
        - Phase 2: cube_3를 cube_2 위에 올리기 → cube_3_to_cube_2 직접 사용

    비교:
        - 기존: cube_positions (world frame, 9-dim) → policy가 subtraction 학습 필요
        - 새로: relative vectors (6-dim) → policy가 바로 사용
    """
    cube_1: RigidObject = env.scene["cube_1"]
    cube_2: RigidObject = env.scene["cube_2"]
    cube_3: RigidObject = env.scene["cube_3"]

    # cube_2 → cube_1 상대 벡터 (Phase 1에서 직접 사용)
    # 이 벡터가 작아질수록 cube_2가 cube_1에 가까워짐
    cube_2_to_cube_1 = cube_1.data.root_pos_w - cube_2.data.root_pos_w

    # cube_3 → cube_2 상대 벡터 (Phase 2에서 직접 사용)
    # 이 벡터가 작아질수록 cube_3가 cube_2에 가까워짐
    cube_3_to_cube_2 = cube_2.data.root_pos_w - cube_3.data.root_pos_w

    # 결합: [N, 6]
    relative_positions = torch.cat([cube_2_to_cube_1, cube_3_to_cube_2], dim=-1)

    return relative_positions


def cube_1_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    20251211 추가: cube_1의 절대 위치 반환 (3-dim)

    Returns:
        torch.Tensor: [N, 3] - cube_1 position in world frame

    이유:
        - cube_1은 base cube로, 절대 위치 필요
        - 다른 큐브들은 상대 위치로만 표현
        - IsaacGym도 base cube(cubeA)는 절대 위치 사용

    역할:
        - 전체 stack의 기준점 제공
        - Robot이 어디서 작업할지 파악
    """
    cube_1: RigidObject = env.scene["cube_1"]
    return cube_1.data.root_pos_w


def cube_orientations_compact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    20251211 수정: 큐브 orientation 반환 (12-dim, 기존과 동일)

    Returns:
        torch.Tensor: [N, 12] - 3 cubes × quaternion (4)

    참고:
        - Orientation은 상대값으로 변환하기 어려움
        - 절대 quaternion 그대로 사용 (IsaacGym도 동일)
        - 추후 euler angles로 변환 고려 가능 (12 → 9-dim)
    """
    cube_1: RigidObject = env.scene["cube_1"]
    cube_2: RigidObject = env.scene["cube_2"]
    cube_3: RigidObject = env.scene["cube_3"]

    cube_1_quat = cube_1.data.root_quat_w
    cube_2_quat = cube_2.data.root_quat_w
    cube_3_quat = cube_3.data.root_quat_w

    return torch.cat([cube_1_quat, cube_2_quat, cube_3_quat], dim=-1)
