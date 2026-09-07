# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Hdr035 + ATI + DG5F Left Hand environments.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# Base environments (without F/T sensor)
##

gym.register(
    id="Isaac-Dexsuite-Hdr035-Dg5fLeft-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_hdr035_dg5f_left_env_cfg:DexsuiteHdr035Dg5fLeftLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteHdr035Dg5fLeftPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexsuite-Hdr035-Dg5fLeft-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_hdr035_dg5f_left_env_cfg:DexsuiteHdr035Dg5fLeftLiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteHdr035Dg5fLeftPPORunnerCfg",
    },
)

# ##
# # ATI F/T Sensor environments (Forge approach)
# ##

# gym.register(
#     id="Isaac-Dexsuite-Hdr035-Dg5fLeft-Lift-FT-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.dexsuite_hdr035_dg5f_left_env_cfg:DexsuiteHdr035Dg5fLeftLiftEnvCfg_FT",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteHdr035Dg5fLeftPPORunnerCfg",
#     },
# )

# gym.register(
#     id="Isaac-Dexsuite-Hdr035-Dg5fLeft-Lift-FT-Play-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.dexsuite_hdr035_dg5f_left_env_cfg:DexsuiteHdr035Dg5fLeftLiftEnvCfg_FT_PLAY",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteHdr035Dg5fLeftPPORunnerCfg",
#     },
# )
