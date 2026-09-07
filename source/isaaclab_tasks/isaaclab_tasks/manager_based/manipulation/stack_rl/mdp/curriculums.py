# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for the stack_rl task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stack_stages_success(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    success_threshold: float = 0.7,
    window_size: int = 100,
) -> torch.Tensor:
    """Success-based curriculum for stacking task.

    This curriculum automatically adjusts the task difficulty (stage) based on the
    success rate of each environment. It uses a rolling window to track recent
    performance and transitions between stages when the success rate exceeds
    the threshold.

    Stages:
    - Stage 1: Stack cube2 on cube1
    - Stage 2: Stack cube3 on cube2 (cube1 and cube2 already stacked)
    - Stage 3: Complete tower (all 3 cubes stacked)

    Args:
        env: The learning environment.
        env_ids: The environment IDs for which to update the curriculum.
        success_threshold: Success rate (0.0-1.0) required to advance to next stage.
        window_size: Number of recent episodes to consider for success rate calculation.

    Returns:
        The mean curriculum stage across all environments.
    """
    # Initialize curriculum buffers if not exists
    if not hasattr(env, "curriculum_stages"):
        # Current stage for each environment (1, 2, or 3)
        env.curriculum_stages = torch.ones(env.num_envs, dtype=torch.long, device=env.device)
        # Success history for rolling window [num_envs, window_size]
        env.curriculum_success_history = torch.zeros(
            env.num_envs, window_size, dtype=torch.bool, device=env.device
        )
        # Current index in the rolling window
        env.curriculum_window_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        # Episode count for each env (to know when window is full)
        env.curriculum_episode_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # Check if we have the required success flags in extras
    # These should be set by the reward function
    cube2_stacked = env.extras.get("cube2_stacked", torch.zeros_like(env_ids, dtype=torch.bool))
    cube3_stacked = env.extras.get("cube3_stacked", torch.zeros_like(env_ids, dtype=torch.bool))
    tower_complete = env.extras.get("tower_complete", torch.zeros_like(env_ids, dtype=torch.bool))

    # Update success history for each environment
    for i, env_id in enumerate(env_ids):
        current_stage = env.curriculum_stages[env_id].item()

        # Determine if this episode was successful based on current stage
        if current_stage == 1:
            success = cube2_stacked[i]
        elif current_stage == 2:
            success = cube3_stacked[i]
        else:  # stage 3
            success = tower_complete[i]

        # Update rolling window
        window_idx = env.curriculum_window_idx[env_id]
        env.curriculum_success_history[env_id, window_idx] = success
        env.curriculum_window_idx[env_id] = (window_idx + 1) % window_size
        env.curriculum_episode_count[env_id] += 1

        # Calculate success rate (only if we have enough data)
        episodes_recorded = min(env.curriculum_episode_count[env_id].item(), window_size)
        if episodes_recorded >= min(20, window_size):  # At least 20 episodes or window_size
            success_rate = env.curriculum_success_history[env_id, :episodes_recorded].float().mean()

            # Stage transition logic
            if current_stage == 1 and success_rate >= success_threshold:
                # Advance from stage 1 to stage 2
                env.curriculum_stages[env_id] = 2
                # Reset window for new stage
                env.curriculum_success_history[env_id] = False
                env.curriculum_window_idx[env_id] = 0
                env.curriculum_episode_count[env_id] = 0
                print(f"[Curriculum] Env {env_id}: Stage 1 → 2 (success_rate={success_rate:.2%})")

            elif current_stage == 2 and success_rate >= success_threshold:
                # Advance from stage 2 to stage 3
                env.curriculum_stages[env_id] = 3
                # Reset window for new stage
                env.curriculum_success_history[env_id] = False
                env.curriculum_window_idx[env_id] = 0
                env.curriculum_episode_count[env_id] = 0
                print(f"[Curriculum] Env {env_id}: Stage 2 → 3 (success_rate={success_rate:.2%})")

            elif success_rate < success_threshold * 0.3 and current_stage > 1:
                # Regression: if success rate drops too low, go back one stage
                # This helps if the agent "forgets" earlier skills
                env.curriculum_stages[env_id] = current_stage - 1
                # Reset window for regressed stage
                env.curriculum_success_history[env_id] = False
                env.curriculum_window_idx[env_id] = 0
                env.curriculum_episode_count[env_id] = 0
                print(f"[Curriculum] Env {env_id}: Stage {current_stage} → {current_stage-1} (success_rate={success_rate:.2%})")

    # Store current stage in extras for tensorboard logging
    env.extras["curriculum_stage"] = env.curriculum_stages.float().mean()

    # Also store per-stage statistics
    for stage in [1, 2, 3]:
        stage_mask = env.curriculum_stages == stage
        count = stage_mask.sum().item()
        env.extras[f"curriculum_stage_{stage}_count"] = count
        env.extras[f"curriculum_stage_{stage}_ratio"] = count / env.num_envs

    # Return mean stage for tracking
    return env.curriculum_stages.float().mean()
