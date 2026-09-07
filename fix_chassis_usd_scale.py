#!/usr/bin/env python3
"""
Fix ScaleOrientation issues in chassis USD file.
Run this script in Isaac Sim Python console or with ./isaaclab.sh -p fix_chassis_usd_scale.py
"""

from pxr import Usd, UsdGeom, Gf
import omni.usd

# USD 파일 경로
usd_path = "/home/dyros/IsaacLab/hyundai/1_factory_usd_file/flattended_usd/env_diated_decomposed_chassis_spring_wire_flattened.usd"
output_path = usd_path.replace(".usd", "_scale_fixed.usd")

print(f"Loading USD: {usd_path}")
stage = Usd.Stage.Open(usd_path)

# 문제가 되는 prims
problem_prims = [
    "/env_decomposed_chassis_spring/wire-revolute-collision-flattened/wire_model/left_ring",
    "/env_decomposed_chassis_spring/wire-revolute-collision-flattened/wire_model/right_ring",
]

for prim_path in problem_prims:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"❌ Prim not found: {prim_path}")
        continue

    print(f"\n{'='*60}")
    print(f"Processing: {prim_path}")
    print(f"{'='*60}")

    xformable = UsdGeom.Xformable(prim)

    # 현재 transform ops 확인
    ops = xformable.GetOrderedXformOps()
    print(f"Current transform ops: {len(ops)}")

    for op in ops:
        op_name = op.GetOpName()
        op_type = op.GetOpType()
        op_value = op.Get()

        print(f"  - {op_name}: {op_type} = {op_value}")

        # ScaleOrientation 제거
        if "orient" in op_name.lower() or op_type == UsdGeom.XformOp.TypeOrient:
            print(f"    🔧 Removing ScaleOrientation")
            # Orient를 identity quaternion으로 설정
            if op_value is not None:
                op.Set(Gf.Quatf(1, 0, 0, 0))

        # Non-uniform scale을 uniform으로 변경
        if op_type == UsdGeom.XformOp.TypeScale:
            if op_value and len(op_value) == 3:
                if abs(op_value[0] - op_value[1]) > 0.01 or abs(op_value[1] - op_value[2]) > 0.01:
                    # Non-uniform scale 감지
                    avg_scale = (op_value[0] + op_value[1] + op_value[2]) / 3.0
                    uniform_scale = Gf.Vec3f(avg_scale, avg_scale, avg_scale)
                    print(f"    🔧 Changing scale from {op_value} to {uniform_scale}")
                    op.Set(uniform_scale)
                else:
                    print(f"    ✅ Scale is already uniform: {op_value}")

# 저장
print(f"\n{'='*60}")
print(f"Saving fixed USD to: {output_path}")
print(f"{'='*60}")

stage.Export(output_path)
print("✅ Done! Use the new USD file in your config:")
print(f"   usd_path=\"{output_path}\"")
