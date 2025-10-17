# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher # <- isaacsim 불러오기 위한 코드 

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args headless mode나 같은 것들

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# app_launcher 실행하면 앱 실행

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# app_launcher가지고서 parsing된 환경 불러올 수 있음

"""Rest everything follows."""

# simulation must run before importing this

from isaaclab.sim import SimulationCfg, SimulationContext

# 위부분먼저 하면 isaac sim이 꺼짐

def main():
    """Main function."""

    # Initialize the simulation context
    # isaac sim config simulation gravity, time 설정 하는 class
    # dt만 0.01로 setting 
    sim_cfg = SimulationCfg(dt=0.01) # simulation physics, time-step 등등
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0]) #position in the 30 space, target point the camera should look at

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    # 시뮬레이션이 돌아가는 구문
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
