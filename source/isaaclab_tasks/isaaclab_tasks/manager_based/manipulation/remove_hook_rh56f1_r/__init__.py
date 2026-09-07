# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Remove Hook Task Environments.

Task Description:
- Remove wire hook from spring on front chassis assembly
- Move hook to target position

Robot Configurations:
- HDR35_20 + RH56F1_R (Inspire Hand Right) - Primary configuration
- HDR35_20 + DG5F_L (DG5F Left) - Future extension

Implementation inspired by Dexsuite benchmark:

Reorient:
@article{petrenko2023dexpbt,
  title={Dexpbt: Scaling up dexterous manipulation for hand-arm systems with population based training},
  author={Petrenko, Aleksei and Allshire, Arthur and State, Gavriel and Handa, Ankur and Makoviychuk, Viktor},
  journal={arXiv preprint arXiv:2305.12127},
  year={2023}
}

Lift:
@article{singh2024dextrah,
  title={Dextrah-rgb: Visuomotor policies to grasp anything with dexterous hands},
  author={Singh, Ritvik and Allshire, Arthur and Handa, Ankur and Ratliff, Nathan and Van Wyk, Karl},
  journal={arXiv preprint arXiv:2412.01791},
  year={2024}
}
"""

# Explicitly import config modules to register environments
from .config import hdr35_20_rh56f1_r  # noqa: F401
