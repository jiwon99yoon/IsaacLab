# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Remove Hook Task Environments - DG5F Left Hand Variant.

Task Description:
- Remove wire hook from spring on front chassis assembly
- Move hook to target position

Robot Configuration:
- HDR35_20 + DG5F_L (Tesollo DG5F Left, 20-DOF fully-actuated)
- Target: right_ring (왼손이므로 오른쪽 고리를 타겟)

Key Differences from RH56F1_R variant:
- Hand DoF: 20 (fully-actuated) vs 6 (under-actuated)
- Higher precision control possible
- Different fingertip body names (ll_dg_X_4 vs right_xxx_X)
- Different palm body name (ll_dg_palm vs gripper_base_link)
"""

# Explicitly import config modules to register environments
from .config import hdr35_20_dg5f_l  # noqa: F401
