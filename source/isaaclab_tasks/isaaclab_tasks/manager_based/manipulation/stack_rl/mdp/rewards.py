# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_grasped_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    """Reward for grasping an object."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        assert len(gripper_joint_ids) == 2, "Observations only support parallel gripper for now"

        grasped = torch.logical_and(
            pose_diff < diff_threshold,
            torch.abs(
                robot.data.joint_pos[:, gripper_joint_ids[0]]
                - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            )
            > env.cfg.gripper_threshold,
        )
        grasped = torch.logical_and(
            grasped,
            torch.abs(
                robot.data.joint_pos[:, gripper_joint_ids[1]]
                - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            )
            > env.cfg.gripper_threshold,
        )
        return grasped.float()
    else:
        raise ValueError("No gripper_joint_names found in environment config")


def object_placed_on_table(
    env: ManagerBasedRLEnv,
    target_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    height_tolerance: float = 0.02,
) -> torch.Tensor:
    """Reward for placing an object on the table at target height."""
    object: RigidObject = env.scene[object_cfg.name]
    object_height = object.data.root_pos_w[:, 2]

    # Check if object is at target height (on table)
    at_target_height = torch.abs(object_height - target_height) < height_tolerance

    # Check if object velocity is low (stable)
    object_vel = torch.norm(object.data.root_lin_vel_w, dim=1)
    is_stable = object_vel < 0.1

    placed = torch.logical_and(at_target_height, is_stable)
    return placed.float()


def object_stacked_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
) -> torch.Tensor:
    """Reward for stacking one object on another."""
    robot: Articulation = env.scene[robot_cfg.name]
    upper_object: RigidObject = env.scene[upper_object_cfg.name]
    lower_object: RigidObject = env.scene[lower_object_cfg.name]

    pos_diff = upper_object.data.root_pos_w - lower_object.data.root_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    stacked = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        assert len(gripper_joint_ids) == 2, "Observations only support parallel gripper for now"
        stacked = torch.logical_and(
            torch.isclose(
                robot.data.joint_pos[:, gripper_joint_ids[0]],
                torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                atol=1e-4,
                rtol=1e-4,
            ),
            stacked,
        )
        stacked = torch.logical_and(
            torch.isclose(
                robot.data.joint_pos[:, gripper_joint_ids[1]],
                torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                atol=1e-4,
                rtol=1e-4,
            ),
            stacked,
        )
    else:
        raise ValueError("No gripper_joint_names found in environment config")

    return stacked.float()


def cubes_stacked_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Success bonus when cube_2 is successfully stacked on cube_1."""
    robot: Articulation = env.scene["robot"]
    cube_1: RigidObject = env.scene["cube_1"]
    cube_2: RigidObject = env.scene["cube_2"]

    pos_diff = cube_2.data.root_pos_w - cube_1.data.root_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    # Stacking criteria
    xy_threshold = 0.05
    height_threshold = 0.005
    height_diff = 0.0468

    stacked = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

    # Check if gripper is open (released the cube)
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        stacked = torch.logical_and(
            torch.isclose(
                robot.data.joint_pos[:, gripper_joint_ids[0]],
                torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                atol=1e-4,
                rtol=1e-4,
            ),
            stacked,
        )

    return stacked.float()


def three_cubes_stacked_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Success bonus when all 3 cubes are successfully stacked (cube_1 base, cube_2 on cube_1, cube_3 on cube_2)."""
    robot: Articulation = env.scene["robot"]
    cube_1: RigidObject = env.scene["cube_1"]
    cube_2: RigidObject = env.scene["cube_2"]
    cube_3: RigidObject = env.scene["cube_3"]

    # Check cube_2 on cube_1
    pos_diff_2_on_1 = cube_2.data.root_pos_w - cube_1.data.root_pos_w
    height_dist_2_on_1 = torch.linalg.vector_norm(pos_diff_2_on_1[:, 2:], dim=1)
    xy_dist_2_on_1 = torch.linalg.vector_norm(pos_diff_2_on_1[:, :2], dim=1)

    # Check cube_3 on cube_2
    pos_diff_3_on_2 = cube_3.data.root_pos_w - cube_2.data.root_pos_w
    height_dist_3_on_2 = torch.linalg.vector_norm(pos_diff_3_on_2[:, 2:], dim=1)
    xy_dist_3_on_2 = torch.linalg.vector_norm(pos_diff_3_on_2[:, :2], dim=1)

    # Stacking criteria
    xy_threshold = 0.05
    height_threshold = 0.005
    height_diff = 0.0468

    stacked_2_on_1 = torch.logical_and(
        xy_dist_2_on_1 < xy_threshold, (height_dist_2_on_1 - height_diff) < height_threshold
    )
    stacked_3_on_2 = torch.logical_and(
        xy_dist_3_on_2 < xy_threshold, (height_dist_3_on_2 - height_diff) < height_threshold
    )

    # Both stacks must be successful
    all_stacked = torch.logical_and(stacked_2_on_1, stacked_3_on_2)

    # Check gripper is open (released the cubes)
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        all_stacked = torch.logical_and(
            torch.isclose(
                robot.data.joint_pos[:, gripper_joint_ids[0]],
                torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                atol=1e-4,
                rtol=1e-4,
            ),
            all_stacked,
        )

    return all_stacked.float()


# ========================================
# 20251210 1601 수정: 새로운 Helper 함수들
# ========================================

def check_cube_2_on_cube_1(env: ManagerBasedRLEnv, xy_threshold: float = 0.02, z_threshold: float = 0.003) -> torch.Tensor:
    """
    20251210 1601 수정: Masking용 - cube_2가 cube_1 위에 쌓였는지 확인

    Args:
        env: 환경
        xy_threshold: XY 정렬 임계값 (m)
        z_threshold: 높이 정렬 임계값 (m)

    Returns:
        Boolean tensor [N] - cube_2가 cube_1 위에 쌓였으면 True
    """
    cube_1: RigidObject = env.scene["cube_1"]
    cube_2: RigidObject = env.scene["cube_2"]

    pos_diff = cube_2.data.root_pos_w - cube_1.data.root_pos_w
    xy_dist = torch.norm(pos_diff[:, :2], dim=1)
    z_dist = torch.abs(pos_diff[:, 2] - 0.0468)  # cube height

    stacked = torch.logical_and(
        xy_dist < xy_threshold,
        z_dist < z_threshold
    )

    return stacked


def check_object_grasped(env: ManagerBasedRLEnv, robot: Articulation, object: RigidObject, diff_threshold: float = 0.06) -> torch.Tensor:
    """
    20251210 1601 수정: 물체를 잡았는지 확인

    Args:
        env: 환경
        robot: 로봇
        object: 확인할 물체
        diff_threshold: EE-object 거리 임계값 (m)

    Returns:
        Boolean tensor [N] - 잡았으면 True
    """
    ee_frame: FrameTransformer = env.scene["ee_frame"]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.norm(object_pos - end_effector_pos, dim=1)

    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)

        grasped = torch.logical_and(
            pose_diff < diff_threshold,
            torch.abs(
                robot.data.joint_pos[:, gripper_joint_ids[0]]
                - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            )
            > env.cfg.gripper_threshold,
        )
        grasped = torch.logical_and(
            grasped,
            torch.abs(
                robot.data.joint_pos[:, gripper_joint_ids[1]]
                - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            )
            > env.cfg.gripper_threshold,
        )
        return grasped
    else:
        raise ValueError("No gripper_joint_names found in environment config")


def check_gripper_open(env: ManagerBasedRLEnv, robot: Articulation) -> torch.Tensor:
    """
    20251210 1601 수정: 그리퍼가 열려있는지 확인

    Args:
        env: 환경
        robot: 로봇

    Returns:
        Boolean tensor [N] - 그리퍼가 열려있으면 True
    """
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)

        gripper_open = torch.isclose(
            robot.data.joint_pos[:, gripper_joint_ids[0]],
            torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
            atol=1e-4,
            rtol=1e-4,
        )
        return gripper_open
    else:
        raise ValueError("No gripper_joint_names found in environment config")


# ========================================
# 20251210 1601 수정: Observation 함수
# ========================================

def cube_dimensions_in_world_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    20251210 1601 수정: 각 큐브의 3D 크기 반환 (9-dim)

    Policy가 큐브 크기를 알아야 적절한 높이로 들어올려 쌓을 수 있음

    Returns:
        Tensor [N, 9] - [cube_1_xyz, cube_2_xyz, cube_3_xyz]
    """
    # 현재 모든 큐브가 동일한 크기: [0.0405, 0.0405, 0.0468]
    size = torch.tensor([0.0405, 0.0405, 0.0468], dtype=torch.float32, device=env.device)
    # 3개 큐브에 대해 반복
    sizes = size.unsqueeze(0).repeat(env.num_envs, 3)  # [N, 9]
    return sizes


# ========================================
# 20251210 1601 수정: 새로운 Reward 함수들
# ========================================

def object_is_lifted_after_grasp(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    initial_table_height: float = 0.0203,
) -> torch.Tensor:
    """
    20251210 1601 수정: 잡은 후 충분히 들어올렸는지 확인

    Args:
        env: 환경
        minimal_height: 최소 들어올림 높이 (초기 높이 대비)
        object_cfg: 물체 설정
        robot_cfg: 로봇 설정
        initial_table_height: 테이블 위 초기 큐브 높이

    Returns:
        Float tensor [N] - 잡고 + 들어올렸으면 1.0
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # 잡았는지 확인
    grasped = check_object_grasped(env, robot, object)

    # 들어올렸는지 확인 (초기 높이 + minimal_height)
    lifted = object.data.root_pos_w[:, 2] > (initial_table_height + minimal_height)

    # 잡고 + 들어올렸을 때만 보상
    return (grasped * lifted).float()


def stacking_hybrid_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    approach_weight: float = 5.0,
    success_weight: float = 30.0,
    xy_threshold: float = 0.02,
    height_threshold: float = 0.003,
    height_diff: float = 0.0468,
) -> torch.Tensor:
    """
    20251210 1601 수정: Hybrid reward (Dense approach + Sparse one-time success)

    Dense: 잡고 있을 때 목표 위치에 가까워질수록 보상
    Sparse: 쌓고 그리퍼 열면 한 번만 큰 보상 (reward hacking 방지)

    Args:
        env: 환경
        robot_cfg: 로봇 설정
        upper_object_cfg: 위 물체 설정
        lower_object_cfg: 아래 물체 설정
        approach_weight: Approach reward 가중치
        success_weight: Success reward 가중치
        xy_threshold: XY 정렬 임계값
        height_threshold: 높이 정렬 임계값
        height_diff: 목표 높이 차이 (큐브 높이)

    Returns:
        Float tensor [N] - approach_reward + success_reward
    """
    robot: Articulation = env.scene[robot_cfg.name]
    upper: RigidObject = env.scene[upper_object_cfg.name]
    lower: RigidObject = env.scene[lower_object_cfg.name]

    # === Part 1: Dense approach (잡고 있을 때만) ===
    holding = check_object_grasped(env, robot, upper)

    # 목표 위치: lower_cube 위 height_diff 만큼
    target_pos = lower.data.root_pos_w.clone()
    target_pos[:, 2] += height_diff

    distance_to_target = torch.norm(
        upper.data.root_pos_w - target_pos, dim=1
    )
    approach_reward = torch.exp(-5.0 * distance_to_target)  # 0~1
    approach_reward = approach_reward * holding.float()

    # === Part 2: Sparse success (한 번만) ===
    pos_diff = upper.data.root_pos_w - lower.data.root_pos_w
    xy_dist = torch.norm(pos_diff[:, :2], dim=1)
    z_dist = torch.abs(pos_diff[:, 2] - height_diff)

    stacked = torch.logical_and(
        xy_dist < xy_threshold,
        z_dist < height_threshold
    )

    # 그리퍼 열려야 함
    gripper_open = check_gripper_open(env, robot)
    success = stacked * gripper_open

    # One-time trigger: 환경에 flag 저장
    flag_name = f"_{upper_object_cfg.name}_stacked_flag"
    if not hasattr(env, flag_name):
        setattr(env, flag_name, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))

    flag = getattr(env, flag_name)
    newly_stacked = success * (~flag)
    flag[:] = torch.logical_or(flag, success)  # Update flag

    # === Combine ===
    return approach_weight * approach_reward + success_weight * newly_stacked.float()


def reaching_cube_3_masked(env: ManagerBasedRLEnv, std: float = 0.1) -> torch.Tensor:
    """
    20251210 1601 수정: cube_3 reaching (cube_2가 cube_1 위에 쌓였을 때만 활성화)

    Args:
        env: 환경
        std: Tanh kernel std

    Returns:
        Float tensor [N] - reaching reward (masked)
    """
    # 기본 reaching reward
    base_reward = object_ee_distance(env, std=std, object_cfg=SceneEntityCfg("cube_3"))

    # Mask: cube_2가 cube_1 위에 있는지
    mask = check_cube_2_on_cube_1(env)

    return base_reward * mask.float()


def grasping_cube_3_masked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    20251210 1601 수정: cube_3 grasping (cube_2가 cube_1 위에 쌓였을 때만 활성화)

    Returns:
        Float tensor [N] - grasping reward (masked)
    """
    base_reward = object_grasped_reward(
        env, robot_cfg, ee_frame_cfg, SceneEntityCfg("cube_3")
    )
    mask = check_cube_2_on_cube_1(env)
    return base_reward * mask.float()


def lifting_cube_3_masked(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    20251210 1601 수정: cube_3 lifting (cube_2가 cube_1 위에 쌓였을 때만 활성화)

    Returns:
        Float tensor [N] - lifting reward (masked)
    """
    base_reward = object_is_lifted_after_grasp(
        env, minimal_height, SceneEntityCfg("cube_3"), robot_cfg
    )
    mask = check_cube_2_on_cube_1(env)
    return base_reward * mask.float()


def stacking_cube_3_on_2_masked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    approach_weight: float = 5.0,
    success_weight: float = 50.0,
    xy_threshold: float = 0.02,
    height_threshold: float = 0.003,
) -> torch.Tensor:
    """
    20251210 1601 수정: cube_3 stacking on cube_2 (cube_2가 cube_1 위에 쌓였을 때만 활성화)

    Returns:
        Float tensor [N] - stacking reward (masked)
    """
    base_reward = stacking_hybrid_reward(
        env,
        robot_cfg,
        SceneEntityCfg("cube_3"),
        SceneEntityCfg("cube_2"),
        approach_weight=approach_weight,
        success_weight=success_weight,
        xy_threshold=xy_threshold,
        height_threshold=height_threshold,
    )
    mask = check_cube_2_on_cube_1(env)
    return base_reward * mask.float()


def object_stays_on_table(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    initial_height: float,
    tolerance: float,
) -> torch.Tensor:
    """
    20251210 1601 수정: 물체가 초기 높이 근처에 있으면 보상 (cube_1 테이블 위 유지)

    Args:
        env: 환경
        object_cfg: 물체 설정
        initial_height: 초기 높이
        tolerance: 허용 오차

    Returns:
        Float tensor [N] - 테이블 위에 있으면 1.0
    """
    object: RigidObject = env.scene[object_cfg.name]

    height_diff = torch.abs(object.data.root_pos_w[:, 2] - initial_height)
    on_table = height_diff < tolerance

    return on_table.float()


def object_movement_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    20251210 1601 수정: 물체가 움직이면 페널티 (cube_1 이동 방지)

    Args:
        env: 환경
        object_cfg: 물체 설정

    Returns:
        Float tensor [N] - 속도가 높을수록 음수 페널티
    """
    object: RigidObject = env.scene[object_cfg.name]

    # 선형 + 각속도
    lin_vel = torch.norm(object.data.root_lin_vel_w, dim=1)
    ang_vel = torch.norm(object.data.root_ang_vel_w, dim=1)

    # 속도가 높을수록 페널티 (각속도는 작게 가중)
    movement = lin_vel + 0.1 * ang_vel

    return -movement  # Negative penalty


def stack_stability_when_released(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    cube_cfgs: list,
    velocity_threshold: float,
) -> torch.Tensor:
    """
    20251210 1601 수정: 그리퍼 열렸을 때 큐브들이 안정적이면 보상

    조작 중에는 간섭하지 않고, 놓은 후에만 안정성 체크

    Args:
        env: 환경
        robot_cfg: 로봇 설정
        cube_cfgs: 큐브 설정 리스트
        velocity_threshold: 속도 임계값 (m/s)

    Returns:
        Float tensor [N] - 그리퍼 열림 + 안정적이면 1.0
    """
    robot: Articulation = env.scene[robot_cfg.name]

    # 그리퍼 열려있는지
    gripper_open = check_gripper_open(env, robot)

    # 모든 큐브의 속도 체크
    total_velocity = torch.zeros(env.num_envs, device=env.device)
    for cube_cfg in cube_cfgs:
        cube: RigidObject = env.scene[cube_cfg.name]
        vel = torch.norm(cube.data.root_lin_vel_w, dim=1)
        total_velocity += vel

    # 평균 속도
    avg_velocity = total_velocity / len(cube_cfgs)
    stable = avg_velocity < velocity_threshold

    # 그리퍼 열렸을 때만 체크
    return (stable * gripper_open).float()


def three_cubes_stacked_success_v2(
    env: ManagerBasedRLEnv,
    cube_1_height_tolerance: float = 0.01,
) -> torch.Tensor:
    """
    20251210 1601 수정: 3-cube stacking success (cube_1 테이블 체크 추가)

    성공 조건:
    1. cube_2 on cube_1
    2. cube_3 on cube_2
    3. cube_1 on table (추가!)
    4. gripper open

    Args:
        env: 환경
        cube_1_height_tolerance: cube_1 테이블 높이 허용 오차

    Returns:
        Float tensor [N] - 모든 조건 만족 시 1.0
    """
    robot: Articulation = env.scene["robot"]
    cube_1: RigidObject = env.scene["cube_1"]
    cube_2: RigidObject = env.scene["cube_2"]
    cube_3: RigidObject = env.scene["cube_3"]

    # 기존 체크 (cube_2 on cube_1, cube_3 on cube_2)
    xy_threshold = 0.02  # 20251210 1601 수정: 0.05 → 0.02 (더 엄격)
    height_threshold = 0.003
    height_diff = 0.0468

    # cube_2 on cube_1
    pos_diff_2_on_1 = cube_2.data.root_pos_w - cube_1.data.root_pos_w
    xy_dist_2_on_1 = torch.norm(pos_diff_2_on_1[:, :2], dim=1)
    z_dist_2_on_1 = torch.abs(pos_diff_2_on_1[:, 2] - height_diff)
    stack_2_on_1 = torch.logical_and(
        xy_dist_2_on_1 < xy_threshold,
        z_dist_2_on_1 < height_threshold
    )

    # cube_3 on cube_2
    pos_diff_3_on_2 = cube_3.data.root_pos_w - cube_2.data.root_pos_w
    xy_dist_3_on_2 = torch.norm(pos_diff_3_on_2[:, :2], dim=1)
    z_dist_3_on_2 = torch.abs(pos_diff_3_on_2[:, 2] - height_diff)
    stack_3_on_2 = torch.logical_and(
        xy_dist_3_on_2 < xy_threshold,
        z_dist_3_on_2 < height_threshold
    )

    all_stacked = torch.logical_and(stack_2_on_1, stack_3_on_2)

    # 20251210 1601 수정: cube_1이 테이블 위에 있는지 추가 체크
    cube_1_on_table = torch.abs(
        cube_1.data.root_pos_w[:, 2] - 0.0203
    ) < cube_1_height_tolerance
    all_stacked = torch.logical_and(all_stacked, cube_1_on_table)

    # 그리퍼 열림
    gripper_open = check_gripper_open(env, robot)

    return (all_stacked * gripper_open).float()


# ========================================
# 20251211 추가: IsaacGym 방식 Reward 함수들
# ========================================
# 배경: 기존 hybrid reward 방식의 문제점 발견
#   - Continuous approach reward로 인한 exploit (들고 이동만 해도 계속 보상)
#   - Gripper opening 유인 없음
#   - Mutual exclusion 없어서 성공 후에도 approach reward 계속 받음
#
# 해결: IsaacGym의 torch.where() 방식 도입
#   - 성공 시: sparse success reward만
#   - 실패 시: dense approach rewards만
#   - Gripper opening reward 추가
#   - One-way transition (성공하면 approach 사라짐)
# ========================================


def reaching_or_aligning_isaacgym(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    minimal_height: float = 0.04,
) -> torch.Tensor:
    """
    20251211 추가: IsaacGym 방식 - max(reaching, aligning)

    IsaacGym 철학:
        - Reaching: EE → object (집기 전)
        - Aligning: object → target (들고 난 후)
        - max()로 둘 중 하나만 받음 (동시에 두 개 받는 것 방지)

    Args:
        env: 환경
        std: Tanh kernel std
        object_cfg: 잡을 물체 (cube_2 or cube_3)
        target_cfg: 목표 물체 (cube_1 or cube_2)
        minimal_height: Align 활성화 최소 높이

    Returns:
        Float tensor [N] - max(reach_reward, align_reward)

    참고:
        IsaacGym franka_cube_stack.py line 718-735
        dist_reward = max(dist_reward, align_reward)
    """
    object: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    ee_frame: FrameTransformer = env.scene["ee_frame"]

    # === Reaching: EE → object ===
    object_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    reach_dist = torch.norm(object_pos - ee_pos, dim=1)
    # 20251211 수정 (3차): 10.0 → 3.0
    # 이유: Tanh saturation 문제로 gradient vanishing 발생
    #       거리 0.3m에서 tanh(3.0) = 0.995 → gradient ≈ 0 (학습 불가능)
    #       3.0으로 감소 시 tanh(0.9) = 0.716 → gradient 충분 (학습 가능)
    #       IsaacGym도 10.0 사용하지만, 초기 거리가 더 가까움 (0.1-0.15m)
    reach_reward = 1 - torch.tanh(3.0 * reach_dist)

    # === Aligning: object → target (lifted일 때만) ===
    # 목표: target 위 height_diff 위치
    object_height = object.data.root_pos_w[:, 2]
    lifted = object_height > minimal_height

    # Target 위치 계산
    offset = torch.zeros_like(object_pos)
    offset[:, 2] = 0.0468  # Cube height
    target_pos = target.data.root_pos_w + offset

    align_dist = torch.norm(object_pos - target_pos, dim=1)
    # 20251211 수정 (3차): 10.0 → 3.0
    # 이유: Epoch 2500까지 policy가 observation 무시하고 hard-coded 방향 이동
    #       스크린샷 분석: cube_1이 왼쪽인데 robot이 오른쪽으로 이동 (반대!)
    #       Gradient가 너무 약해서 (0.001) policy가 cube_relative_positions 학습 못 함
    #       3.0으로 감소: 거리 0.3m에서 reward 0.284 (기존 0.005의 57배!)
    #       → Policy가 observation 사용하여 올바른 방향 학습 가능
    align_reward = (1 - torch.tanh(3.0 * align_dist)) * lifted.float()

    # === Max (IsaacGym 방식) ===
    # 한 번에 하나만 최대값 받음
    return torch.max(reach_reward, align_reward)


def object_is_lifted_simple(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    20251211 추가: IsaacGym 방식 - Binary lift reward

    IsaacGym 철학:
        - Lifted 여부만 체크 (binary 0 or 1)
        - Continuous하게 주지 않음 (exploit 방지)
        - Grasping 체크 없음 (단순하게)

    Args:
        env: 환경
        minimal_height: 최소 들어올림 높이
        object_cfg: 물체 설정

    Returns:
        Float tensor [N] - 들어올렸으면 1.0, 아니면 0.0

    참고:
        IsaacGym franka_cube_stack.py line 723-726
        cubeA_lifted = (cubeA_height - cubeA_size) > 0.04
        lift_reward = cubeA_lifted
    """
    object: RigidObject = env.scene[object_cfg.name]

    # Table 위 높이
    object_height = object.data.root_pos_w[:, 2]
    table_height = 0.0203

    lifted = (object_height - table_height) > minimal_height

    return lifted.float()


def gripper_opening_at_target(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    xy_threshold: float = 0.10,
    z_threshold: float = 0.01,
) -> torch.Tensor:
    """
    20251211 추가: Gripper opening reward (새로운 개념)

    문제:
        - 기존: Gripper open 조건은 있지만 유인 없음
        - Policy가 왜 그리퍼를 열어야 하는지 모름

    해결:
        - 올바른 위치에 있을 때 그리퍼 열면 보상
        - Opening 행동 자체에 대한 인센티브 제공

    Args:
        env: 환경
        robot_cfg: 로봇 설정
        upper_object_cfg: 위 물체 (cube_2 or cube_3)
        lower_object_cfg: 아래 물체 (cube_1 or cube_2)
        xy_threshold: XY 정렬 임계값
        z_threshold: Z 정렬 임계값

    Returns:
        Float tensor [N] - 올바른 위치 + 그리퍼 열면 1.0

    철학:
        - IsaacGym은 "gripper away" 조건만 있음
        - 우리는 "gripper opening" 행동 자체를 보상
        - Opening이 학습되면 자연스럽게 success로 이어짐
    """
    robot: Articulation = env.scene[robot_cfg.name]
    upper: RigidObject = env.scene[upper_object_cfg.name]
    lower: RigidObject = env.scene[lower_object_cfg.name]

    # === 1. 올바른 위치 체크 ===
    pos_diff = upper.data.root_pos_w - lower.data.root_pos_w
    xy_dist = torch.norm(pos_diff[:, :2], dim=1)
    z_dist = torch.abs(pos_diff[:, 2] - 0.0468)  # Target height

    at_target = xy_dist < xy_threshold
    # Z threshold 제거:
    #   - XY 정렬만으로 충분 (올바른 위치 위에 있음)
    #   - Z는 들고 있으면 자연스럽게 맞춰짐
    #   - Success condition에서 정밀 체크하므로 guidance는 느슨하게
    # 20251211 수정 (4차): Z threshold 제거
    # 이유: Z 조건(1cm)이 너무 엄격하여 gripper opening 학습 불가
    #       XY 정렬만으로도 "올바른 위치"를 충분히 나타냄
    #       Z는 들고 있는 상태에서 자연스럽게 적절한 높이
    #       Success condition (xy<2cm, z<3mm)에서 정밀 체크하므로
    #       Reward는 guidance 목적으로 느슨하게 설정

    # at_target = torch.logical_and(
    #     xy_dist < xy_threshold,
    #     z_dist < z_threshold
    # )

    # === 2. 그리퍼 열림 체크 ===
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)

        gripper_open = torch.isclose(
            robot.data.joint_pos[:, gripper_joint_ids[0]],
            torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
            atol=1e-3,  # 약간 완화 (1e-4 → 1e-3)
            rtol=1e-3,
        )

        # === 3. 조건 결합 ===
        # 올바른 위치 + 그리퍼 열기 = 보상
        return (at_target * gripper_open).float()
    else:
        raise ValueError("No gripper_joint_names found in environment config")


def stacking_success_isaacgym(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    gripper_away_threshold: float = 0.04,
    xy_threshold: float = 0.02,
    height_threshold: float = 0.02,
) -> torch.Tensor:
    """
    20251211 추가: IsaacGym 방식 - Stacking success with gripper away

    IsaacGym 철학:
        - 성공 조건: (1) 위치 정렬 + (2) 그리퍼 멀리 떨어짐
        - "Gripper away" = 놓고 떨어져야 함 (단순히 open이 아님!)
        - Binary reward (0 or 1)

    Args:
        env: 환경
        robot_cfg: 로봇 설정
        upper_object_cfg: 위 물체
        lower_object_cfg: 아래 물체
        gripper_away_threshold: EE가 물체에서 떨어진 거리 (m)
        xy_threshold: XY 정렬 임계값
        height_threshold: 높이 정렬 임계값

    Returns:
        Float tensor [N] - 성공 시 1.0

    참고:
        IsaacGym franka_cube_stack.py line 737-741
        cubeA_align_cubeB = (norm(cubeA_to_cubeB_pos[:, :2]) < 0.02)
        cubeA_on_cubeB = abs(cubeA_height - target_height) < 0.02
        gripper_away_from_cubeA = (d > 0.04)
        stack_reward = cubeA_align & cubeA_on_cubeB & gripper_away
    """
    robot: Articulation = env.scene[robot_cfg.name]
    upper: RigidObject = env.scene[upper_object_cfg.name]
    lower: RigidObject = env.scene[lower_object_cfg.name]
    ee_frame: FrameTransformer = env.scene["ee_frame"]

    # === 1. Position alignment ===
    pos_diff = upper.data.root_pos_w - lower.data.root_pos_w
    xy_dist = torch.norm(pos_diff[:, :2], dim=1)

    # Target height (cube size)
    cube_size = 0.0468
    z_dist = torch.abs(pos_diff[:, 2] - cube_size)

    position_aligned = torch.logical_and(
        xy_dist < xy_threshold,
        z_dist < height_threshold
    )

    # === 2. Gripper away from object (IsaacGym 핵심!) ===
    upper_pos = upper.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    ee_to_object_dist = torch.norm(upper_pos - ee_pos, dim=1)

    gripper_away = ee_to_object_dist > gripper_away_threshold

    # === 3. Success = position + gripper away ===
    success = torch.logical_and(position_aligned, gripper_away)

    return success.float()


def phase1_mutual_exclusive_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    20251211 추가: Phase 1 Mutual Exclusive Reward (IsaacGym 핵심!)

    IsaacGym 철학:
        rewards = torch.where(
            stack_reward,
            stack_scale * stack_reward,      # 성공 → sparse만
            dist_scale * dist + lift_scale * lift + align_scale * align  # 실패 → dense만
        )

    핵심:
        - 성공하면 approach rewards 사라짐 (mutual exclusion!)
        - One-way transition: 성공 → sparse, 실패 → dense
        - Exploit 불가능: 성공 후 다시 approach 못 받음

    구성:
        - Approach: reaching_or_aligning(2.0) + lifting(3.0) + gripper_opening(5.0) = max 10.0
        - Success: stacking_success(16.0)
        - torch.where()로 mutual exclusion

    Args:
        env: 환경
        robot_cfg: 로봇 설정

    Returns:
        Float tensor [N] - mutually exclusive reward

    참고:
        IsaacGym franka_cube_stack.py line 748-753
    """
    # === Approach rewards (dense) ===
    reach_align = reaching_or_aligning_isaacgym(
        env,
        std=0.1,
        object_cfg=SceneEntityCfg("cube_2"),
        target_cfg=SceneEntityCfg("cube_1"),
        minimal_height=0.04,
    )

    lift = object_is_lifted_simple(
        env,
        minimal_height=0.04,
        object_cfg=SceneEntityCfg("cube_2"),
    )

    gripper_open = gripper_opening_at_target(
        env,
        robot_cfg=robot_cfg,
        upper_object_cfg=SceneEntityCfg("cube_2"),
        lower_object_cfg=SceneEntityCfg("cube_1"),
        xy_threshold=0.10,
        z_threshold=0.01,
    )

    # Approach total (weight는 RewardsCfg에서 적용)
    # 20251211 수정 (3차): reach_align weight 2.0 → 10.0
    # 이유: Lift (3.0)가 너무 dominant해서 align signal이 묻힘
    #       Policy가 "cube 들고 있으면 됨" 학습, "어디로 가야할지" 학습 못 함
    #       10.0으로 증가: align (최대 2.84) vs lift (3.0) - 이제 comparable!
    #       Credit assignment 가능: observation → action → reward 연결 학습
    # 20251211 수정 (4차): reach_align weight 10.0 → 15.0
    # 이유: Robot이 cube를 높이 들어올리는 문제 (40-60cm)
    #       Lift (3.0) 보장되니까 높이 올리는 local optimum
    #       Align을 lift보다 강하게 만들어 "낮게 들고 정렬" 유도
    #       15.0으로 증가: align (최대 4.26) > lift (3.0)
    #       Height penalty와 함께 작동하여 적절한 높이 학습
    approach_rewards = (
        15.0 * reach_align +  # 10.0 → 15.0 (align이 lift보다 강함!)
        3.0 * lift +          # IsaacGym: 1.5 lift
        5.0 * gripper_open    # 새로 추가 (opening 유도)
    )

    # === Success reward (sparse) ===
    success = stacking_success_isaacgym(
        env,
        robot_cfg=robot_cfg,
        upper_object_cfg=SceneEntityCfg("cube_2"),
        lower_object_cfg=SceneEntityCfg("cube_1"),
        gripper_away_threshold=0.04,
        xy_threshold=0.02,
        height_threshold=0.02,
    )

    success_reward = 16.0 * success  # IsaacGym: 16.0

    # === Mutual Exclusion (IsaacGym 핵심!) ===
    # Check if cube_2 is stacked on cube_1
    cube_2_stacked = check_cube_2_on_cube_1(
        env,
        xy_threshold=0.02,
        z_threshold=0.003
    )

    # torch.where: 성공 시 sparse만, 실패 시 dense만
    total_reward = torch.where(
        cube_2_stacked,
        success_reward,      # Stacked → only success (16.0)
        approach_rewards     # Not stacked → only approach (max 10.0)
    )

    return total_reward


def phase2_mutual_exclusive_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    20251211 추가: Phase 2 Mutual Exclusive Reward (Phase 1과 동일 구조)

    Phase 2: cube_3 on cube_2

    차이점:
        - cube_2가 cube_1 위에 쌓였을 때만 활성화 (masking)
        - Success reward 더 높음 (20.0, Phase 1의 16.0보다 높음)

    Args:
        env: 환경
        robot_cfg: 로봇 설정

    Returns:
        Float tensor [N] - mutually exclusive reward (masked)
    """
    # === Masking: cube_2 on cube_1 체크 ===
    mask = check_cube_2_on_cube_1(env, xy_threshold=0.02, z_threshold=0.003)

    # === Approach rewards (dense) ===
    reach_align = reaching_or_aligning_isaacgym(
        env,
        std=0.1,
        object_cfg=SceneEntityCfg("cube_3"),
        target_cfg=SceneEntityCfg("cube_2"),
        minimal_height=0.08,  # cube_2 위로 들어야 함
    )

    lift = object_is_lifted_simple(
        env,
        minimal_height=0.08,
        object_cfg=SceneEntityCfg("cube_3"),
    )

    gripper_open = gripper_opening_at_target(
        env,
        robot_cfg=robot_cfg,
        upper_object_cfg=SceneEntityCfg("cube_3"),
        lower_object_cfg=SceneEntityCfg("cube_2"),
        xy_threshold=0.10,
        z_threshold=0.01,
    )

    # 20251211 수정 (3차): Phase 1과 동일하게 reach_align weight 10.0 적용
    # 이유: Phase 2도 Phase 1과 동일한 문제 겪을 것 (tanh saturation, weak gradient)
    #       미리 같이 수정하여 Phase 2 진입 시 바로 학습 가능하도록
    # 20251211 수정 (4차): reach_align weight 10.0 → 15.0 (Phase 1과 동일)
    # 이유: Phase 2도 높이 문제 겪을 것 (cube_3를 높이 들어올림)
    #       Align을 lift보다 강하게 하여 "낮게 들고 정렬" 유도
    #       Phase 1과 일관성 유지
    approach_rewards = (
        15.0 * reach_align +  # 10.0 → 15.0 (Phase 1과 동일)
        3.0 * lift +
        5.0 * gripper_open
    )

    # === Success reward (sparse, 더 높음) ===
    success = stacking_success_isaacgym(
        env,
        robot_cfg=robot_cfg,
        upper_object_cfg=SceneEntityCfg("cube_3"),
        lower_object_cfg=SceneEntityCfg("cube_2"),
        gripper_away_threshold=0.04,
        xy_threshold=0.02,
        height_threshold=0.02,
    )

    success_reward = 20.0 * success  # Phase 2는 더 높은 보상

    # === Check if cube_3 is stacked on cube_2 ===
    cube_3: RigidObject = env.scene["cube_3"]
    cube_2: RigidObject = env.scene["cube_2"]

    pos_diff = cube_3.data.root_pos_w - cube_2.data.root_pos_w
    xy_dist = torch.norm(pos_diff[:, :2], dim=1)
    z_dist = torch.abs(pos_diff[:, 2] - 0.0468)

    cube_3_stacked = torch.logical_and(
        xy_dist < 0.02,
        z_dist < 0.003
    )

    # === Mutual Exclusion ===
    total_reward = torch.where(
        cube_3_stacked,
        success_reward,
        approach_rewards
    )

    # === Masking ===
    # cube_2가 안 쌓이면 0
    return total_reward * mask.float()


def cube_height_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    max_reasonable_height: float = 0.25,
    penalty_scale: float = 5.0,
) -> torch.Tensor:
    """
    20251211 추가 (4차): Cube 높이 제한 penalty

    배경:
        - 문제 발견: Robot이 cube를 잡은 후 비정상적으로 높이 들어올림 (40-60cm)
        - 원인: Lift reward가 binary (4cm 이상이면 무조건 1.0)
                높이 올릴수록 안전하다고 학습 (collision 회피)
                Align reward가 saturate되어 높이 차이를 구분 못 함
        - 스크린샷 증거: moving_up_motion_after_pickup_cube2.png

    해결:
        - Cube가 합리적인 높이(25cm) 이상 올라가면 quadratic penalty
        - 높을수록 가파르게 패널티 증가
        - Policy가 "낮게 들고 이동"하도록 유도

    Args:
        env: 환경
        object_cfg: Cube 설정 (cube_2 or cube_3)
        max_reasonable_height: 합리적인 최대 높이 (default 0.25m = 25cm)
                               이 높이까지는 penalty 없음
        penalty_scale: 패널티 강도 (default 5.0)
                      높을수록 강한 패널티

    Returns:
        Float tensor [N] - Penalty (음수 또는 0)
                          0: 합리적인 높이
                          음수: 너무 높음 (제곱 패널티)

    예시:
        max_height = 0.25, scale = 5.0일 때:
        - 10cm: penalty = 0 (good!)
        - 25cm: penalty = 0 (경계)
        - 30cm: penalty = -5.0 × (0.05)² = -0.0125
        - 40cm: penalty = -5.0 × (0.15)² = -0.1125
        - 50cm: penalty = -5.0 × (0.25)² = -0.3125 (강함!)
        - 60cm: penalty = -5.0 × (0.35)² = -0.6125 (매우 강함!)

    철학:
        - 너무 높이 들면 align 어려움 (거리 멀어짐)
        - 적당한 높이 유지가 최적 (10-20cm)
        - Quadratic penalty로 높을수록 가파르게 불이익

    참고:
        - Session 2025-12-11: Height penalty discussion
        - 기존 lift reward는 binary로 유지 (높이 구분 없음)
        - 이 penalty가 높이 조절 담당
    """
    object: RigidObject = env.scene[object_cfg.name]

    # Cube의 현재 높이
    cube_height = object.data.root_pos_w[:, 2]
    table_height = 0.0203

    height_above_table = cube_height - table_height

    # 합리적인 높이 초과분 계산
    # max_reasonable_height 이하면 0, 이상이면 양수
    excess_height = torch.clamp(height_above_table - max_reasonable_height, min=0.0)

    # Quadratic penalty (제곱)
    # 높을수록 가파르게 증가
    penalty = -penalty_scale * excess_height ** 2

    return penalty


# ==============================================================================
# Anca et al. (2023) Style Reward Functions - Option B Implementation
# ==============================================================================
# 논문: "Achieving Goals using Reward Shaping and Curriculum Learning"
# arXiv:2206.02462
#
# 20251212 추가 (5차 수정): Stage-based Curriculum Learning
#
# 핵심 개념:
#   1. Curriculum Stage: 1, 2, 3 (epoch 기반 전환)
#   2. Gated Rewards: Stage에 따라 active cube만 reward 받음
#   3. Sub-goal Bonus: 각 stage 완료 시 one-time 큰 보상 (λ=150)
#   4. Dense Shaping: Continuous guidance (λ=5)
#
# Curriculum Timeline:
#   - Epoch 0-1000: Stage 1 (cube_2 → cube_1만)
#   - Epoch 1000-5000: Stage 2 (cube_2 완료 + cube_3 → cube_2)
#   - Epoch 5000+: Stage 3 (all cubes)
# ==============================================================================


def get_curriculum_stage(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    현재 curriculum stage 반환 (1, 2, or 3)

    20251212 수정: Success-based curriculum로 변경
        - 기존: Epoch 기반 고정 전환 (common_step_counter 문제로 작동 안 함)
        - 새로: Success-based per-environment stages
        - stack_stages_success() curriculum이 각 env의 stage 자동 관리
        - 이 함수는 단순히 현재 stage를 읽어서 반환

    Returns:
        torch.Tensor: [num_envs] with stage number (1, 2, or 3)
            - Stage가 아직 설정 안 됐으면 모두 1로 초기화
            - curriculum_stages가 있으면 그것을 사용 (per-env stages)
    """
    # Check if success-based curriculum has set curriculum_stages
    if hasattr(env, "curriculum_stages"):
        # Use per-environment stages set by stack_stages_success()
        return env.curriculum_stages
    else:
        # Fallback: Initialize to stage 1 for all environments
        # This will be overridden by curriculum manager on first reset
        if "curriculum_stage" not in env.extras:
            env.extras["curriculum_stage"] = torch.ones(
                env.num_envs, dtype=torch.long, device=env.device
            )
        return env.extras["curriculum_stage"]


def init_subgoal_tracking(env: ManagerBasedRLEnv):
    """
    Sub-goal one-time bonus tracking 초기화

    각 cube별로 sub-goal 달성 여부를 tracking하여
    한 번만 bonus를 주도록 함

    Storage in env.extras:
        - subgoal_cube_1_done: [num_envs], bool
        - subgoal_cube_2_done: [num_envs], bool
        - subgoal_cube_3_done: [num_envs], bool
    """
    if "subgoal_cube_1_done" not in env.extras:
        env.extras["subgoal_cube_1_done"] = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    if "subgoal_cube_2_done" not in env.extras:
        env.extras["subgoal_cube_2_done"] = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    if "subgoal_cube_3_done" not in env.extras:
        env.extras["subgoal_cube_3_done"] = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )


def reset_subgoal_tracking(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """
    특정 environment들의 sub-goal tracking reset

    Episode 종료 시 호출하여 다음 episode를 위해 초기화
    """
    init_subgoal_tracking(env)  # Ensure initialized

    env.extras["subgoal_cube_1_done"][env_ids] = False
    env.extras["subgoal_cube_2_done"][env_ids] = False
    env.extras["subgoal_cube_3_done"][env_ids] = False


# ==============================================================================
# Anca Reward Term 1: Box-to-Goal Distance (Dense Shaping)
# ==============================================================================

def box_to_goal_distance_anca(
    env: ManagerBasedRLEnv,
    cube_1_goal_height: float = 0.0203,
    cube_2_goal_offset: float = 0.0406,
    cube_3_goal_offset: float = 0.0812,
) -> torch.Tensor:
    """
    Anca et al. r_dense: Box-to-goal distance reward

    논문 식:
        r_dense = -Σ ||s_goal - s_obj||² / 2

    Curriculum gating:
        - Stage 1: cube_2만 active
        - Stage 2: cube_2 + cube_3 active
        - Stage 3: all cubes active (우리는 cube_1은 고정)

    Args:
        cube_1_goal_height: Cube_1 목표 높이 (테이블 위)
        cube_2_goal_offset: Cube_2 목표 높이 (cube_1 위)
        cube_3_goal_offset: Cube_3 목표 높이 (cube_2 위)

    Returns:
        torch.Tensor: [num_envs] Dense distance reward

    Weight (Anca): λ = 5.0
    """
    stage = get_curriculum_stage(env)

    # Scene objects
    cube_1: RigidObject = env.scene["cube_1"]
    cube_2: RigidObject = env.scene["cube_2"]
    cube_3: RigidObject = env.scene["cube_3"]

    # Current positions
    cube_1_pos = cube_1.data.root_pos_w  # [num_envs, 3]
    cube_2_pos = cube_2.data.root_pos_w
    cube_3_pos = cube_3.data.root_pos_w

    # Goal positions
    # Cube_1: 테이블 위 초기 위치 (base, 고정)
    cube_1_goal = cube_1_pos.clone()
    cube_1_goal[:, 2] = cube_1_goal_height

    # Cube_2: Cube_1 위
    cube_2_goal = cube_1_pos.clone()
    cube_2_goal[:, 2] = cube_1_pos[:, 2] + cube_2_goal_offset

    # Cube_3: Cube_2 위 (cube_2 현재 위치 기준)
    cube_3_goal = cube_2_pos.clone()
    cube_3_goal[:, 2] = cube_2_pos[:, 2] + cube_3_goal_offset

    # Distance squared
    dist_1_sq = torch.sum((cube_1_pos - cube_1_goal) ** 2, dim=-1)
    dist_2_sq = torch.sum((cube_2_pos - cube_2_goal) ** 2, dim=-1)
    dist_3_sq = torch.sum((cube_3_pos - cube_3_goal) ** 2, dim=-1)

    # Gated by curriculum stage
    reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # Cube_1은 항상 고정이므로 reward에서 제외 (또는 아주 작은 weight)
    # reward -= 0.5 * dist_1_sq  # Optional

    # Cube_2: Stage 1+
    mask_2 = (stage >= 1)
    reward -= 0.5 * dist_2_sq * mask_2.float()

    # Cube_3: Stage 2+
    mask_3 = (stage >= 2)
    reward -= 0.5 * dist_3_sq * mask_3.float()

    return reward


# ==============================================================================
# Anca Reward Term 2: EE-to-Box Distance (Dense Shaping)
# ==============================================================================

def ee_to_box_distance_anca(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Anca et al. r_guide: End-effector to box distance reward

    논문 식:
        r_guide = -Σ ||s_ee - s_obj||² / 2

    Curriculum gating:
        - Stage 1: EE → cube_2만
        - Stage 2: EE → cube_3만 (cube_2는 이미 쌓임)
        - Stage 3: 모두

    Returns:
        torch.Tensor: [num_envs] EE guidance reward

    Weight (Anca): λ = 5.0
    """
    stage = get_curriculum_stage(env)

    # End-effector position
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # [num_envs, 3]

    # Cube positions
    cube_2: RigidObject = env.scene["cube_2"]
    cube_3: RigidObject = env.scene["cube_3"]

    cube_2_pos = cube_2.data.root_pos_w
    cube_3_pos = cube_3.data.root_pos_w

    # Distance squared
    dist_ee_to_2_sq = torch.sum((ee_pos - cube_2_pos) ** 2, dim=-1)
    dist_ee_to_3_sq = torch.sum((ee_pos - cube_3_pos) ** 2, dim=-1)

    # Gated reward
    reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # Stage 1: EE → cube_2
    mask_1 = (stage == 1)
    reward -= 0.5 * dist_ee_to_2_sq * mask_1.float()

    # Stage 2+: EE → cube_3
    mask_2_plus = (stage >= 2)
    reward -= 0.5 * dist_ee_to_3_sq * mask_2_plus.float()

    return reward


# ==============================================================================
# Anca Reward Term 3: Action Penalty
# ==============================================================================

def action_penalty_anca(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Anca et al. r_action: Action magnitude + joint deviation penalty

    논문 식:
        r_action = -||a||² / 2 - ||θ - θ_0||² / 2

    Returns:
        torch.Tensor: [num_envs] Action penalty

    Weight (Anca): λ = 0.01
    """
    robot: Articulation = env.scene[robot_cfg.name]

    # Action magnitude (if available in env)
    # IsaacLab stores actions in action_manager
    if hasattr(env, "action_manager"):
        actions = env.action_manager.action  # [num_envs, action_dim]
        action_norm_sq = torch.sum(actions ** 2, dim=-1)
    else:
        action_norm_sq = torch.zeros(env.num_envs, device=env.device)

    # Joint deviation from home pose
    # Home pose: 보통 [0, -0.785, 0, -2.356, 0, 1.571, 0.785] for Franka
    joint_pos = robot.data.joint_pos[:, :7]  # [num_envs, 7]

    # Home pose (IsaacLab default for Franka)
    joint_home = torch.tensor(
        [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        dtype=torch.float32,
        device=env.device,
    ).unsqueeze(0).expand(env.num_envs, -1)

    joint_dev_sq = torch.sum((joint_pos - joint_home) ** 2, dim=-1)

    # Combined penalty
    penalty = -0.5 * (action_norm_sq + joint_dev_sq)

    return penalty


# ==============================================================================
# Anca Reward Term 4: Table Collision Penalty
# ==============================================================================

def table_collision_penalty_anca(
    env: ManagerBasedRLEnv,
    table_height: float = 0.0,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Anca et al. r_table: Penalty for EE hitting table

    논문 식:
        r_table = -1 if z_ee <= table_height else 0

    Args:
        table_height: 테이블 상단 높이 (default 0.0, ground)

    Returns:
        torch.Tensor: [num_envs] Table collision penalty

    Weight (Anca): λ = 5.0
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # [num_envs, 3]

    z = ee_pos[:, 2]

    # Penalty if EE below table
    hit = (z <= table_height)
    penalty = -hit.float()

    return penalty


# ==============================================================================
# Anca Reward Term 5: EE Orientation Penalty
# ==============================================================================

def ee_orientation_penalty_anca(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Anca et al. r_orientation: Penalty for deviating from initial orientation

    논문 식:
        r_orientation = -(ω_0 * ω_ee)  # quaternion inner product

    Initial orientation 저장:
        env.extras["ee_quat_init"]

    Returns:
        torch.Tensor: [num_envs] Orientation penalty

    Weight (Anca): λ = 0.1
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]  # [num_envs, 4]

    # Normalize quaternion
    ee_quat = ee_quat / torch.norm(ee_quat, dim=-1, keepdim=True)

    # Initial quaternion (stored on first call or reset)
    if "ee_quat_init" not in env.extras:
        env.extras["ee_quat_init"] = ee_quat.clone()

    ee_quat_init = env.extras["ee_quat_init"]

    # Inner product (cosine similarity)
    dot = torch.sum(ee_quat_init * ee_quat, dim=-1)

    # Penalty (negative of dot product)
    # dot = 1 (aligned) → penalty = -1
    # dot = 0 (orthogonal) → penalty = 0
    penalty = -dot

    return penalty


# ==============================================================================
# Anca Reward Term 6: Sub-goal Sparse Bonus (One-time)
# ==============================================================================

def subgoal_sparse_bonus_anca(
    env: ManagerBasedRLEnv,
    goal_eps: float = 0.03,
    cube_size: float = 0.0203,
) -> torch.Tensor:
    """
    Anca et al. Sub-goal one-time bonus

    각 cube가 목표 위치에 도달한 순간 한 번만 큰 보상 부여

    Curriculum gating:
        - Stage 1: cube_2 on cube_1 달성 시 +1
        - Stage 2: cube_3 on cube_2 달성 시 +1
        - Stage 3: 모두

    One-time tracking:
        env.extras["subgoal_cube_X_done"]

    Args:
        goal_eps: 목표 도달 판정 threshold (3cm)
        cube_size: Cube 한 변 크기 (2.03cm)

    Returns:
        torch.Tensor: [num_envs] Sub-goal bonus (0 or 1)

    Weight (Anca): λ = 150.0 (매우 큼!)
    """
    stage = get_curriculum_stage(env)
    init_subgoal_tracking(env)

    # Cubes
    cube_1: RigidObject = env.scene["cube_1"]
    cube_2: RigidObject = env.scene["cube_2"]
    cube_3: RigidObject = env.scene["cube_3"]

    cube_1_pos = cube_1.data.root_pos_w
    cube_2_pos = cube_2.data.root_pos_w
    cube_3_pos = cube_3.data.root_pos_w

    reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # ========================================
    # Sub-goal 1: Cube_2 on Cube_1
    # ========================================
    if (stage >= 1).any():
        # Goal: cube_2가 cube_1 위에 있어야 함
        # XY alignment
        xy_dist = torch.norm(cube_2_pos[:, :2] - cube_1_pos[:, :2], dim=-1)
        # Z alignment
        z_expected = cube_1_pos[:, 2] + cube_size * 2  # cube_1 top + cube_2 height
        z_dist = torch.abs(cube_2_pos[:, 2] - z_expected)

        # Goal reached
        reached = (xy_dist < goal_eps) & (z_dist < goal_eps)

        # Stage active
        stage_active = (stage >= 1)

        # First time達成 (newly done)
        subgoal_done = env.extras["subgoal_cube_2_done"]
        newly_done = reached & (~subgoal_done) & stage_active

        # Update tracking
        env.extras["subgoal_cube_2_done"] = subgoal_done | newly_done

        # Reward
        reward += newly_done.float()

    # ========================================
    # Sub-goal 2: Cube_3 on Cube_2
    # ========================================
    if (stage >= 2).any():
        # Goal: cube_3가 cube_2 위에 있어야 함
        xy_dist = torch.norm(cube_3_pos[:, :2] - cube_2_pos[:, :2], dim=-1)
        z_expected = cube_2_pos[:, 2] + cube_size * 2
        z_dist = torch.abs(cube_3_pos[:, 2] - z_expected)

        reached = (xy_dist < goal_eps) & (z_dist < goal_eps)
        stage_active = (stage >= 2)

        subgoal_done = env.extras["subgoal_cube_3_done"]
        newly_done = reached & (~subgoal_done) & stage_active

        env.extras["subgoal_cube_3_done"] = subgoal_done | newly_done

        reward += newly_done.float()

    return reward


# ==============================================================================
# Anca Reward Term 7: Global Success Bonus (All cubes stacked)
# ==============================================================================

def global_success_bonus_anca(
    env: ManagerBasedRLEnv,
    goal_eps: float = 0.03,
    cube_size: float = 0.0203,
) -> torch.Tensor:
    """
    Anca et al. Global success bonus

    모든 cube가 제대로 쌓였을 때 큰 보상

    이미 구현된 three_cubes_stacked_success_v2()를 재사용하되,
    one-time bonus로 만들기 위해 tracking 추가

    Returns:
        torch.Tensor: [num_envs] Global success (0 or 1)

    Weight (Anca): λ = 150.0
    """
    # Reuse existing success function
    success = three_cubes_stacked_success_v2(
        env,
        cube_1_height_tolerance=goal_eps,
    )

    # One-time tracking
    if "global_success_done" not in env.extras:
        env.extras["global_success_done"] = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    global_done = env.extras["global_success_done"]
    newly_done = (success > 0.5) & (~global_done)

    env.extras["global_success_done"] = global_done | newly_done

    return newly_done.float()


# ==============================================================================
# Curriculum Success Tracking (for success-based curriculum)
# ==============================================================================

def track_curriculum_success(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
) -> torch.Tensor:
    """
    Track success flags for curriculum system.

    This function checks if each stage's goal is achieved and stores
    the success flags in env.extras for the curriculum manager to use.

    Success criteria:
    - cube2_stacked: Cube2 is successfully stacked on Cube1 (Stage 1 complete)
    - cube3_stacked: Cube3 is successfully stacked on Cube2 (Stage 2 complete)
    - tower_complete: All 3 cubes are stacked (Stage 3 complete)

    Args:
        env: Environment
        xy_threshold: XY alignment threshold
        height_threshold: Height alignment threshold
        height_diff: Expected height difference between cubes

    Returns:
        torch.Tensor: Dummy return (always 0) - this function just tracks success flags
    """
    robot: Articulation = env.scene["robot"]
    cube_1: RigidObject = env.scene["cube_1"]
    cube_2: RigidObject = env.scene["cube_2"]
    cube_3: RigidObject = env.scene["cube_3"]

    # Get gripper state
    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    gripper_open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)

    # ==== Stage 1: Cube2 on Cube1 ====
    pos_diff_2_on_1 = cube_2.data.root_pos_w - cube_1.data.root_pos_w
    xy_dist_2_on_1 = torch.norm(pos_diff_2_on_1[:, :2], dim=1)
    z_dist_2_on_1 = torch.abs(pos_diff_2_on_1[:, 2] - height_diff)

    cube2_aligned = torch.logical_and(
        xy_dist_2_on_1 < xy_threshold,
        z_dist_2_on_1 < height_threshold
    )

    # Gripper must be open (cube released)
    gripper_open_0 = torch.isclose(
        robot.data.joint_pos[:, gripper_joint_ids[0]],
        gripper_open_val,
        atol=1e-4, rtol=1e-4
    )
    gripper_open_1 = torch.isclose(
        robot.data.joint_pos[:, gripper_joint_ids[1]],
        gripper_open_val,
        atol=1e-4, rtol=1e-4
    )
    gripper_open = torch.logical_and(gripper_open_0, gripper_open_1)

    cube2_stacked = torch.logical_and(cube2_aligned, gripper_open)

    # ==== Stage 2: Cube3 on Cube2 (requires cube2 already stacked) ====
    pos_diff_3_on_2 = cube_3.data.root_pos_w - cube_2.data.root_pos_w
    xy_dist_3_on_2 = torch.norm(pos_diff_3_on_2[:, :2], dim=1)
    z_dist_3_on_2 = torch.abs(pos_diff_3_on_2[:, 2] - height_diff)

    cube3_aligned = torch.logical_and(
        xy_dist_3_on_2 < xy_threshold,
        z_dist_3_on_2 < height_threshold
    )

    cube3_stacked = torch.logical_and(
        torch.logical_and(cube3_aligned, gripper_open),
        cube2_stacked  # Prerequisite: cube2 must be stacked first
    )

    # ==== Stage 3: Complete tower (all 3 cubes stacked) ====
    tower_complete = torch.logical_and(cube2_stacked, cube3_stacked)

    # Store in extras for curriculum manager
    env.extras["cube2_stacked"] = cube2_stacked
    env.extras["cube3_stacked"] = cube3_stacked
    env.extras["tower_complete"] = tower_complete

    # Also store success rates for logging
    env.extras["cube2_stacked_rate"] = cube2_stacked.float().mean()
    env.extras["cube3_stacked_rate"] = cube3_stacked.float().mean()
    env.extras["tower_complete_rate"] = tower_complete.float().mean()

    # Return 0 (this is just a tracking function, not a reward)
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
