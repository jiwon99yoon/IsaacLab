#!/usr/bin/env python3
"""
Script to create UR10e + ATI FT Sensor + DG5F Hand USD file
Based on existing ur10_ati_inspire.usd structure

Usage:
    python create_ur10_dg5f_usd.py

This will create: /home/dyros/IsaacLab/hyundai/ur10_ati_dg5f.usd
"""

from pxr import Usd, UsdGeom, UsdPhysics, Sdf
import os

def create_ur10_dg5f_usd():
    """
    Create UR10 + ATI FT Sensor + DG5F Hand combined USD

    Method: Reference existing USD files and compose them together
    """

    # File paths
    ur10_usd_path = "/home/dyros/IsaacLab/hyundai/ur10_ati.usd"  # UR10 + ATI already combined
    dg5f_usd_path = "/home/dyros/IsaacLab/hyundai/dg_description/urdf/dg5f_right_new/dg5f_right_new.usd"
    output_path = "/home/dyros/IsaacLab/hyundai/ur10_ati_dg5f.usd"

    # Check if files exist
    if not os.path.exists(ur10_usd_path):
        print(f"Error: UR10 USD not found at {ur10_usd_path}")
        return False

    if not os.path.exists(dg5f_usd_path):
        print(f"Error: DG5F USD not found at {dg5f_usd_path}")
        return False

    print(f"Creating combined USD: {output_path}")
    print(f"  UR10+ATI: {ur10_usd_path}")
    print(f"  DG5F: {dg5f_usd_path}")

    # Create a new stage
    stage = Usd.Stage.CreateNew(output_path)

    # Set up metadata
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Create root prim
    root_prim = stage.DefinePrim("/ur10_ati_dg5f", "Xform")
    stage.SetDefaultPrim(root_prim)

    # Add UR10 + ATI as reference
    ur10_prim = stage.DefinePrim("/ur10_ati_dg5f/ur10_ati", "Xform")
    ur10_prim.GetReferences().AddReference(ur10_usd_path)

    # Add DG5F hand as reference
    # The hand should be attached to the UR10's end-effector (wrist_3_link or tool0)
    dg5f_prim = stage.DefinePrim("/ur10_ati_dg5f/dg5f_hand", "Xform")
    dg5f_prim.GetReferences().AddReference(dg5f_usd_path)

    # Set DG5F hand position relative to UR10 end-effector
    # You may need to adjust this transform based on your specific setup
    xformOp = UsdGeom.Xformable(dg5f_prim)
    # Example: Move hand to attach to wrist
    # xformOp.AddTranslateOp().Set((0.0, 0.0, 0.1))  # Adjust as needed

    print("\n⚠️  IMPORTANT: Manual adjustment required!")
    print("1. Open the generated USD in Isaac Sim")
    print("2. Create a fixed joint between UR10 wrist_3_link and DG5F base")
    print("3. Adjust the relative transform to properly align the hand")
    print("4. Save the file")

    # Save the stage
    stage.GetRootLayer().Save()
    print(f"\n✅ Created: {output_path}")
    print("\nNext steps:")
    print("1. Open Isaac Sim")
    print("2. Open the created USD file")
    print("3. Use the UI to properly attach DG5F to UR10 wrist")
    print("4. Save the final version")

    return True


def create_ur10e_dg5f_from_ur10e():
    """
    Alternative: Start from Isaac Sim's UR10e USD and add DG5F
    """

    # File paths
    ur10e_usd_path = "/home/dyros/isaacsim_assets/Assets/Isaac/5.0/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
    ati_usd_path = "/home/dyros/IsaacLab/hyundai/inspire/attached_torque_sensor-tashan/TS-F-A.usd"  # ATI FT sensor
    dg5f_usd_path = "/home/dyros/IsaacLab/hyundai/dg_description/urdf/dg5f_right_new/dg5f_right_new.usd"
    output_path = "/home/dyros/IsaacLab/hyundai/ur10e_ati_dg5f.usd"

    # Check if files exist
    for path, name in [(ur10e_usd_path, "UR10e"), (dg5f_usd_path, "DG5F")]:
        if not os.path.exists(path):
            print(f"Warning: {name} USD not found at {path}")

    print(f"\nCreating UR10e + ATI + DG5F combined USD: {output_path}")

    # Create a new stage
    stage = Usd.Stage.CreateNew(output_path)

    # Set up metadata
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Create root prim
    root_prim = stage.DefinePrim("/ur10e_ati_dg5f", "Xform")
    stage.SetDefaultPrim(root_prim)

    # Add UR10e as reference
    ur10e_prim = stage.DefinePrim("/ur10e_ati_dg5f/ur10e", "Xform")
    ur10e_prim.GetReferences().AddReference(ur10e_usd_path)

    # Add ATI FT sensor if exists
    if os.path.exists(ati_usd_path):
        ati_prim = stage.DefinePrim("/ur10e_ati_dg5f/ati_sensor", "Xform")
        ati_prim.GetReferences().AddReference(ati_usd_path)

    # Add DG5F hand
    dg5f_prim = stage.DefinePrim("/ur10e_ati_dg5f/dg5f_hand", "Xform")
    dg5f_prim.GetReferences().AddReference(dg5f_usd_path)

    # Save the stage
    stage.GetRootLayer().Save()
    print(f"✅ Created: {output_path}")
    print("\n⚠️  Manual assembly required in Isaac Sim!")

    return True


if __name__ == "__main__":
    print("=" * 70)
    print("UR10/UR10e + DG5F USD Creator")
    print("=" * 70)

    print("\nOption 1: Create ur10_ati_dg5f.usd (from existing ur10_ati.usd)")
    print("Option 2: Create ur10e_ati_dg5f.usd (from Isaac Sim's ur10e.usd)")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        create_ur10_dg5f_usd()
    elif choice == "2":
        create_ur10e_dg5f_from_ur10e()
    else:
        print("Invalid choice. Running both...")
        create_ur10_dg5f_usd()
        print("\n" + "=" * 70 + "\n")
        create_ur10e_dg5f_from_ur10e()

    print("\n" + "=" * 70)
    print("IMPORTANT: The generated USD requires manual assembly in Isaac Sim!")
    print("=" * 70)
    print("\nSteps to complete in Isaac Sim:")
    print("1. Open Isaac Sim")
    print("2. File -> Open -> Select the generated USD")
    print("3. In the Stage panel, create a Fixed Joint:")
    print("   - Parent: /ur10[e]_ati_dg5f/ur10[e]/wrist_3_link (or tool0)")
    print("   - Child: /ur10[e]_ati_dg5f/dg5f_hand/base_link")
    print("4. Adjust the relative transform (translation/rotation)")
    print("5. File -> Save")
    print("\nAlternatively, use the Isaac Sim URDF Importer to combine URDFs first!")
