# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the dexsuite task.

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
        in_bound_range: The range in x, y, z such that the object is considered in range
    """
    object: RigidObject = env.scene[asset_cfg.name]
    range_list = [in_bound_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)

    object_pos_local = object.data.root_pos_w - env.scene.env_origins
    outside_bounds = ((object_pos_local < ranges[:, 0]) | (object_pos_local > ranges[:, 1])).any(dim=1)
    return outside_bounds


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminating environment when violation of velocity limits detects, this usually indicates unstable physics caused
    by very bad, or aggressive action

    Velocity limit multiplier by robot type:
    - Kuka-Allegro (original dexsuite): ×2 (baseline for manipulation tasks)
    - HDR20, UH035 (industrial robots): ×1000 (high-speed capable, prioritize sim stability)
    - UR10e (collaborative robot): ×1000 (CHANGED: 5→1000, 2025-11-24 디버깅용)
    """
    robot: Articulation = env.scene[asset_cfg.name]

    # Velocity limit multiplier - 디버깅 후 적절한 값으로 조정
    VELOCITY_LIMIT_MULTIPLIER = 1000  # 디버깅용: 5로 설정하여 어떤 joint가 초과하는지 확인
    # 디버깅 완료 (1139) - 손가락 겹치는게 문제 -> 일단 1000으로 바꿔서 ft sensor 확인부터 !
    # 1000에선 잘 됨 : 이제 5로 바꿈 (11250215)
    # 1000으로 일단 다시 바꿈 : 학습 잘되는 거 보여주기 위함 (11271514)

    # Calculate velocity violations
    vel_abs = robot.data.joint_vel.abs()
    vel_limits = robot.data.joint_vel_limits * VELOCITY_LIMIT_MULTIPLIER
    violations = vel_abs > vel_limits

    # =========================================================================
    # [DEBUG] Velocity Violation Logging - 2025-11-24
    # 목적: 어떤 joint가 velocity limit을 초과하는지 분석
    # 삭제 방법: 아래 "# [DEBUG START]" ~ "# [DEBUG END]" 블록 전체 삭제
    # 로그 파일 위치: /home/dyros/IsaacLab/logs/rl_games/dexsuite_ur10e_dg5f_right/vel_debug_YYYYMMDD.txt
    # =========================================================================
    # [DEBUG START]
    import os
    from datetime import datetime

    if not hasattr(env, '_vel_debug_step_count'):
        env._vel_debug_step_count = 0
        env._vel_debug_violation_history = []  # 위반 이력 저장
        # 로그 파일 초기화
        env._vel_debug_log_dir = "/home/dyros/IsaacLab/logs/rl_games/dexsuite_ur10e_dg5f_right"
        os.makedirs(env._vel_debug_log_dir, exist_ok=True)
        env._vel_debug_log_file = os.path.join(
            env._vel_debug_log_dir,
            f"vel_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        # 파일 헤더 작성
        with open(env._vel_debug_log_file, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"Velocity Violation Debug Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"VELOCITY_LIMIT_MULTIPLIER = {VELOCITY_LIMIT_MULTIPLIER}\n")
            f.write(f"{'='*80}\n\n")
        print(f"[DEBUG VEL] Log file created: {env._vel_debug_log_file}")

    env._vel_debug_step_count += 1

    # 50 step마다 로그 출력
    if env._vel_debug_step_count % 50 == 0:
        # 위반이 있는 환경 찾기
        any_violation = violations.any(dim=1)
        num_violations = any_violation.sum().item()

        log_lines = []  # 파일에 쓸 내용 저장

        if num_violations > 0:
            # 위반이 있는 환경 중 첫 번째 분석
            violation_env_idx = any_violation.nonzero(as_tuple=True)[0][0].item()

            # 해당 환경의 각 joint 분석
            env_vel = vel_abs[violation_env_idx]  # [num_joints]
            env_limits = vel_limits[violation_env_idx]  # [num_joints]
            env_violations = violations[violation_env_idx]  # [num_joints]

            # 위반하는 joint 인덱스
            violating_joints = env_violations.nonzero(as_tuple=True)[0]

            header = f"\n{'='*80}\n"
            header += f"[DEBUG VEL] Step {env._vel_debug_step_count} | "
            header += f"Violations: {num_violations}/{env.num_envs} envs ({100*num_violations/env.num_envs:.1f}%)\n"
            header += f"{'='*80}\n"
            print(header)
            log_lines.append(header)

            # Joint 이름 가져오기
            joint_names = robot.joint_names

            joint_header = f"[DEBUG VEL] Violating joints in env {violation_env_idx}:\n"
            print(joint_header, end='')
            log_lines.append(joint_header)

            for joint_idx in violating_joints[:10]:  # 최대 10개만 출력
                joint_idx = joint_idx.item()
                joint_name = joint_names[joint_idx] if joint_idx < len(joint_names) else f"joint_{joint_idx}"
                vel_value = env_vel[joint_idx].item()
                limit_value = env_limits[joint_idx].item()
                ratio = vel_value / (limit_value / VELOCITY_LIMIT_MULTIPLIER)  # 원래 limit 대비 배수

                line = (f"  [{joint_idx:2d}] {joint_name:30s} | "
                        f"vel={vel_value:8.2f} rad/s | "
                        f"limit×{VELOCITY_LIMIT_MULTIPLIER}={limit_value:8.2f} | "
                        f"ratio=×{ratio:.1f}\n")
                print(line, end='')
                log_lines.append(line)

            # 전체 joint 중 최대 초과 비율 찾기
            all_ratios = env_vel / (env_limits / VELOCITY_LIMIT_MULTIPLIER + 1e-6)
            max_ratio_idx = all_ratios.argmax().item()
            max_ratio = all_ratios[max_ratio_idx].item()
            max_joint_name = joint_names[max_ratio_idx] if max_ratio_idx < len(joint_names) else f"joint_{max_ratio_idx}"

            max_line = f"\n[DEBUG VEL] Max violation: {max_joint_name} at ×{max_ratio:.1f} of original limit\n"
            print(max_line, end='')
            log_lines.append(max_line)

            # Episode 시작 직후인지 확인 (episode_length_buf로)
            ep_len = env.episode_length_buf[violation_env_idx].item()
            ep_line = f"[DEBUG VEL] Episode length of violating env: {ep_len} steps\n"
            print(ep_line, end='')
            log_lines.append(ep_line)

            if ep_len < 5:
                early_line = f"[DEBUG VEL] ⚠️  Early episode violation! (step {ep_len}) - 초기 spawn 문제 가능성\n"
                print(early_line, end='')
                log_lines.append(early_line)

            footer = f"{'='*80}\n\n"
            print(footer)
            log_lines.append(footer)

        else:
            # 위반 없을 때도 간단히 출력
            no_viol_line = f"[DEBUG VEL] Step {env._vel_debug_step_count} | No violations\n"
            print(no_viol_line, end='')
            log_lines.append(no_viol_line)

        # 파일에 저장
        with open(env._vel_debug_log_file, 'a') as f:
            f.writelines(log_lines)
    # [DEBUG END]
    # =========================================================================

    return violations.any(dim=1)
