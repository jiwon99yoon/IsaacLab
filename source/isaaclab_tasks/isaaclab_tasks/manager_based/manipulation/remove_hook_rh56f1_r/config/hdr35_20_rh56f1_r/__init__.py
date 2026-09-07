# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Remove Hook environment configurations for HDR35_20 + RH56F1_R (Inspire Hand Right).

Task: Remove wire hook from spring and move to target position.
Robot: HDR35_20 (35kg payload arm) + RH56F1_R (6-DOF under-actuated inspire hand right)

Future Extension:
- HDR35_20 + DG5F_L (left hand) configuration can be added as separate config
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# ============================================================================
# Remove Hook Task - HDR35_20 + RH56F1_R (Inspire Hand Right)
# ============================================================================

gym.register(
    id="Isaac-RemoveHook-Hdr35-RH56F1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.remove_hdr35_20_rh56f1_r_env_cfg:RemoveHookHdr35Env_Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteUh035InspirePPORunnerCfg",
    },
)

# Play/Inference mode (longer episodes, no training)
gym.register(
    id="Isaac-RemoveHook-Hdr35-RH56F1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.remove_hdr35_20_rh56f1_r_env_cfg:RemoveHookHdr35Env_Cfg_PLAY",
    },
)

# ============================================================================
# Future: Remove Hook Task - HDR35_20 + DG5F_L (Left Hand)
# ============================================================================
# Uncomment when DG5F_L configuration is ready
#
# gym.register(
#     id="Isaac-RemoveHook-Hdr35-DG5F-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.remove_hdr35_20_dg5f_l_env_cfg:RemoveHookHdr35DG5FEnv_Cfg",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteDG5FPPORunnerCfg",
#     },
# )
