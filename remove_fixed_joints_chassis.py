#!/usr/bin/env python3
"""
Remove FixedJoints from chassis USD file that cause PhysX errors.

This script removes FixedJoint prims from the chassis assembly because:
1. PhysX cannot create joints between static/kinematic bodies
2. FixedJoints are unnecessary for a fixed chassis structure
3. The chassis should be pure collision geometry (AssetBaseCfg)

Run with: ./isaaclab.sh -p remove_fixed_joints_chassis.py
"""

from pxr import Usd, Sdf

# Input and output paths
input_usd = "/home/dyros/IsaacLab/hyundai/1_factory_usd_file/flattended_usd/env_diated_decomposed_chassis_tilted_flattened.usd"
output_usd = "/home/dyros/IsaacLab/hyundai/1_factory_usd_file/flattended_usd/env_diated_decomposed_chassis_tilted_flattened_no_joints.usd"

print(f"Loading USD: {input_usd}")
stage = Usd.Stage.Open(input_usd)

if not stage:
    print("❌ Failed to open USD file!")
    exit(1)

# FixedJoints to remove
fixed_joints_to_remove = [
    "/env_model/right_strut_spring/FixedJoint",
    "/env_model/left_strut_spring/FixedJoint",
    "/env_model/left_frame/FixedJoint",
]

print(f"\n{'='*70}")
print("Removing FixedJoints from chassis assembly")
print(f"{'='*70}\n")

removed_count = 0
for joint_path in fixed_joints_to_remove:
    prim = stage.GetPrimAtPath(joint_path)

    if prim.IsValid():
        print(f"🔍 Found FixedJoint: {joint_path}")

        # Remove the prim
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(joint_path, Sdf.Path.emptyPath)

        if stage.GetRootLayer().Apply(edit):
            print(f"✅ Removed: {joint_path}")
            removed_count += 1
        else:
            print(f"❌ Failed to remove: {joint_path}")
    else:
        print(f"⚠️  Not found (may already be removed): {joint_path}")

print(f"\n{'='*70}")
print(f"Summary: Removed {removed_count} FixedJoint(s)")
print(f"{'='*70}\n")

# Save the modified USD
print(f"Saving modified USD to: {output_usd}")
stage.Export(output_usd)

print(f"\n✅ Done!\n")
print("Next steps:")
print("1. Update your config to use the new USD file:")
print(f'   usd_path="{output_usd}"')
print("2. Run the training script again")
print("\nThe chassis will now work as pure collision geometry (AssetBaseCfg)")
print("without PhysX joint errors.")
