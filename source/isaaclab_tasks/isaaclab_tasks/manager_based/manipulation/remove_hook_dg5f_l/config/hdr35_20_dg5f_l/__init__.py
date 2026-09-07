# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HDR35_20 + DG5F_L robot configuration for Remove Hook task."""

import gymnasium as gym

from . import agents
from .remove_hdr35_20_dg5f_l_env_cfg import RemoveHookHdr35Dg5fEnv_Cfg, RemoveHookHdr35Dg5fEnv_Cfg_PLAY

##
# Register Gym environments.
##

gym.register(
    id="Isaac-RemoveHook-Hdr35-DG5F-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RemoveHookHdr35Dg5fEnv_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-RemoveHook-Hdr35-DG5F-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RemoveHookHdr35Dg5fEnv_Cfg_PLAY,
    },
)
