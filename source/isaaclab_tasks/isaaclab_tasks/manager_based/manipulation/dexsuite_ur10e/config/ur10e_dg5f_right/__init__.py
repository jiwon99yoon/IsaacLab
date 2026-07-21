# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
UR10e + ATI + DG5F Right Hand environments.
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
    id="Isaac-Dexsuite-Ur10e-Dg5fRight-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_ur10e_dg5f_right_env_cfg:DexsuiteUr10eDg5fRightLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteUr10eDg5fRightPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexsuite-Ur10e-Dg5fRight-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_ur10e_dg5f_right_env_cfg:DexsuiteUr10eDg5fRightLiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteUr10eDg5fRightPPORunnerCfg",
    },
)

##
# ATI F/T Sensor environments (Forge approach)
##

gym.register(
    id="Isaac-Dexsuite-Ur10e-Dg5fRight-Lift-FT-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_ur10e_dg5f_right_env_cfg:DexsuiteUr10eDg5fRightLiftEnvCfg_FT",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteUr10eDg5fRightPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexsuite-Ur10e-Dg5fRight-Lift-FT-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_ur10e_dg5f_right_env_cfg:DexsuiteUr10eDg5fRightLiftEnvCfg_FT_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteUr10eDg5fRightPPORunnerCfg",
    },
)
