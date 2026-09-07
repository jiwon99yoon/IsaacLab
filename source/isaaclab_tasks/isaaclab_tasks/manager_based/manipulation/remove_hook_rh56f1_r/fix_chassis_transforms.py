#!/usr/bin/env python3
"""
Script to fix chassis USD transform hierarchy.

Problem: Child meshes have absolute transforms that don't respect parent XForm.
Solution: Flatten all transforms so geometry is relative to root XForm at origin.
"""

from pxr import Usd, UsdGeom, Gf
import sys


def inspect_usd_transforms(usd_path: str):
    """Inspect USD file to see current transform hierarchy."""
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"ERROR: Could not open USD file: {usd_path}")
        return

    print("=" * 80)
    print(f"Inspecting USD: {usd_path}")
    print("=" * 80)

    # Get default prim or root prims
    default_prim = stage.GetDefaultPrim()
    if default_prim:
        print(f"\nDefault Prim: {default_prim.GetPath()}")
        root_prims = [default_prim]
    else:
        root_prims = [stage.GetPrimAtPath(p) for p in stage.GetPseudoRoot().GetChildren()]

    # Recursively print all XForms and their transforms
    def print_xform_hierarchy(prim, indent=0):
        if not prim.IsValid():
            return

        indent_str = "  " * indent

        # Check if it's an XForm
        if prim.IsA(UsdGeom.Xform):
            xform = UsdGeom.Xform(prim)

            # Get local transform
            local_transform = xform.GetLocalTransformation()

            # Extract translation
            translation = local_transform.ExtractTranslation()

            print(f"{indent_str}{prim.GetPath()} (XForm)")
            print(f"{indent_str}  Translation: ({translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f})")
            print(f"{indent_str}  Type: {prim.GetTypeName()}")

        elif prim.IsA(UsdGeom.Mesh):
            print(f"{indent_str}{prim.GetPath()} (Mesh)")

        # Recurse to children
        for child in prim.GetChildren():
            print_xform_hierarchy(child, indent + 1)

    for root in root_prims:
        print_xform_hierarchy(root)

    print("\n" + "=" * 80)


def fix_chassis_transforms(input_usd: str, output_usd: str):
    """
    Fix chassis USD by ensuring all transforms are relative to root at origin.

    Strategy:
    1. Open the USD file
    2. Find the root XForm
    3. Zero out the root XForm's transform (set to identity)
    4. Move all geometry under the root to be positioned at the original root location
    5. Save as a new USD file
    """
    print(f"\nFixing transforms...")
    print(f"Input:  {input_usd}")
    print(f"Output: {output_usd}")

    # Open source stage
    stage = Usd.Stage.Open(input_usd)
    if not stage:
        print(f"ERROR: Could not open USD file: {input_usd}")
        return False

    # Get default prim (should be the chassis root)
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        print("ERROR: No default prim found. Please set a default prim in the USD file.")
        return False

    print(f"\nDefault Prim: {default_prim.GetPath()}")

    # Get the root XForm's current transform
    if not default_prim.IsA(UsdGeom.Xform):
        print(f"ERROR: Default prim is not an XForm, it's a {default_prim.GetTypeName()}")
        return False

    root_xform = UsdGeom.Xform(default_prim)
    root_transform = root_xform.GetLocalTransformation()
    root_translation = root_transform.ExtractTranslation()

    print(f"Root XForm current translation: ({root_translation[0]:.4f}, {root_translation[1]:.4f}, {root_translation[2]:.4f})")

    # Method 1: Clear all xformOps on root (make it identity)
    root_xform.ClearXformOpOrder()

    print(f"✓ Cleared root XForm transform (now identity)")

    # Method 2: Apply the root's original transform to all direct children
    # This preserves the visual appearance but makes transforms relative to root
    for child in default_prim.GetChildren():
        if child.IsA(UsdGeom.Xformable):
            child_xformable = UsdGeom.Xformable(child)

            # Get child's current local transform
            child_local = child_xformable.GetLocalTransformation()

            # Compose with root's original transform
            # new_transform = root_transform * child_local
            new_transform = root_transform * child_local

            # Clear and reset
            child_xformable.ClearXformOpOrder()

            # Create new transform ops
            translate_op = child_xformable.AddTranslateOp()
            translate_op.Set(new_transform.ExtractTranslation())

            # Extract and apply rotation if needed
            rotation = new_transform.ExtractRotation()
            if not rotation.GetIdentity().GetQuaternion().IsClose(rotation.GetQuaternion(), 1e-6):
                rotate_op = child_xformable.AddRotateXYZOp()
                rotate_op.Set(rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1)))

            print(f"  ✓ Updated child: {child.GetPath()}")

    # Save to new file
    stage.Export(output_usd)
    print(f"\n✓ Saved fixed USD to: {output_usd}")

    return True


def main():
    # Original chassis USD
    original_usd = "/home/dyros/IsaacLab/hyundai/1_factory_usd_file/flattended_usd/env_diated_decomposed_chassis_tilted_flattened.usd"

    # Fixed output USD
    fixed_usd = "/home/dyros/IsaacLab/hyundai/1_factory_usd_file/flattended_usd/env_diated_decomposed_chassis_tilted_flattened_FIXED.usd"

    # Step 1: Inspect current structure
    print("\n" + "=" * 80)
    print("STEP 1: Inspecting ORIGINAL USD")
    print("=" * 80)
    inspect_usd_transforms(original_usd)

    # Step 2: Fix transforms
    print("\n" + "=" * 80)
    print("STEP 2: Fixing Transforms")
    print("=" * 80)
    success = fix_chassis_transforms(original_usd, fixed_usd)

    if success:
        # Step 3: Inspect fixed structure
        print("\n" + "=" * 80)
        print("STEP 3: Inspecting FIXED USD")
        print("=" * 80)
        inspect_usd_transforms(fixed_usd)

        print("\n" + "=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print(f"\nFixed USD saved to:")
        print(f"  {fixed_usd}")
        print(f"\nNext steps:")
        print(f"  1. Update your config file to use the fixed USD:")
        print(f"     usd_path=\"{fixed_usd}\"")
        print(f"  2. Run your simulation again")
        print("=" * 80)
    else:
        print("\nERROR: Failed to fix transforms")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
