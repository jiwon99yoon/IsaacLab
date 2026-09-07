#!/usr/bin/env python3
"""
Wire Bar to Rigid-Body Rope Conversion Script

Usage:
    cd ~/isaacsim
    ./python.sh /home/dyros/IsaacLab/create_wire_rope.py
"""

import argparse
from isaacsim import SimulationApp

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run headless")
args = parser.parse_args()

# Launch Isaac Sim
simulation_app = SimulationApp({"headless": args.headless})

# Now import USD modules
import omni.usd
from pxr import UsdGeom, Sdf, Gf, UsdPhysics, UsdShade, PhysxSchema, Usd
from omni.physx.scripts import physicsUtils

print("="*60)
print("Wire Bar -> Rigid-Body Rope Conversion")
print("="*60)

# Configuration
INPUT_USD = "/home/dyros/IsaacLab/hyundai/1_factory_usd_file/flattended_usd/wire_revolute_collision_flattened.usd"
OUTPUT_USD = "/home/dyros/IsaacLab/hyundai/1_factory_usd_file/flattended_usd/wire_rope_0.7m.usd"

TOTAL_LENGTH = 0.7
NUM_SEGMENTS = 14
SEGMENT_RADIUS = 0.004
SEGMENT_LENGTH = TOTAL_LENGTH / NUM_SEGMENTS

CONE_ANGLE_LIMIT = 25.0
ROPE_DAMPING = 0.5
ROPE_STIFFNESS = 0.1
DENSITY = 500.0

WIRE_START_X = -0.32
WIRE_Y = 0.0
WIRE_Z = 0.001

# Open USD file
print(f"\nOpening: {INPUT_USD}")
omni.usd.get_context().open_stage(INPUT_USD)
stage = omni.usd.get_context().get_stage()

if not stage:
    print("ERROR: Failed to open stage!")
    simulation_app.close()
    exit(1)

# Paths
base_path = Sdf.Path("/wire_model")
wire_path = base_path.AppendChild("wire")
rope_scope = wire_path.AppendChild("rope_segments")
joints_scope = wire_path.AppendChild("rope_joints")
material_path = base_path.AppendChild("RopeMaterial")


def create_rope_segment(path, position, color):
    """Create one rope segment (capsule)"""
    capsule = UsdGeom.Capsule.Define(stage, path)
    capsule.CreateHeightAttr(SEGMENT_LENGTH * 0.85)
    capsule.CreateRadiusAttr(SEGMENT_RADIUS)
    capsule.CreateAxisAttr("X")
    capsule.AddTranslateOp().Set(position)
    capsule.CreateDisplayColorAttr().Set([color])

    prim = capsule.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)

    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateDensityAttr().Set(DENSITY)

    UsdPhysics.CollisionAPI.Apply(prim)
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx_collision.CreateRestOffsetAttr().Set(0.0)
    physx_collision.CreateContactOffsetAttr().Set(0.002)

    physicsUtils.add_physics_material_to_prim(stage, prim, material_path)
    return prim


def create_d6_joint(joint_path, body0_path, body1_path, local_pos0, local_pos1):
    """Create D6 joint with 2 rotational DOF"""
    joint = UsdPhysics.Joint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([body0_path])
    joint.CreateBody1Rel().SetTargets([body1_path])
    joint.CreateLocalPos0Attr().Set(local_pos0)
    joint.CreateLocalPos1Attr().Set(local_pos1)

    d6_prim = joint.GetPrim()

    # Lock translation and rotX
    for dof in ["transX", "transY", "transZ", "rotX"]:
        limit_api = UsdPhysics.LimitAPI.Apply(d6_prim, dof)
        limit_api.CreateLowAttr(1.0)
        limit_api.CreateHighAttr(-1.0)

    # Unlock rotY, rotZ for bending
    for dof in ["rotY", "rotZ"]:
        limit_api = UsdPhysics.LimitAPI.Apply(d6_prim, dof)
        limit_api.CreateLowAttr(-CONE_ANGLE_LIMIT)
        limit_api.CreateHighAttr(CONE_ANGLE_LIMIT)

        drive_api = UsdPhysics.DriveAPI.Apply(d6_prim, dof)
        drive_api.CreateTypeAttr("force")
        drive_api.CreateDampingAttr(ROPE_DAMPING)
        drive_api.CreateStiffnessAttr(ROPE_STIFFNESS)

    return joint


# Step 1: Hide existing bar
print("\n[1] Hiding existing bar...")
bar_prim = stage.GetPrimAtPath(wire_path.AppendChild("bar"))
if bar_prim and bar_prim.IsValid():
    UsdGeom.Imageable(bar_prim).MakeInvisible()
    print("    Done")

# Step 2: Create physics material
print("\n[2] Creating physics material...")
if not stage.GetPrimAtPath(material_path):
    UsdShade.Material.Define(stage, material_path)
    mat = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(material_path))
    mat.CreateStaticFrictionAttr().Set(0.5)
    mat.CreateDynamicFrictionAttr().Set(0.5)
    mat.CreateRestitutionAttr().Set(0.0)
print("    Done")

# Step 3: Create scopes
print("\n[3] Creating scopes...")
if not stage.GetPrimAtPath(rope_scope):
    UsdGeom.Scope.Define(stage, rope_scope)
if not stage.GetPrimAtPath(joints_scope):
    UsdGeom.Scope.Define(stage, joints_scope)
print("    Done")

# Step 4: Create rope segments
print(f"\n[4] Creating {NUM_SEGMENTS} rope segments...")
segment_paths = []
orange = Gf.Vec3f(1.0, 0.5, 0.0)

for i in range(NUM_SEGMENTS):
    seg_path = rope_scope.AppendChild(f"seg_{i:02d}")
    x = WIRE_START_X + (i + 0.5) * SEGMENT_LENGTH
    pos = Gf.Vec3f(x, WIRE_Y, WIRE_Z)
    create_rope_segment(seg_path, pos, orange)
    segment_paths.append(seg_path)
print("    Done")

# Step 5: Create inter-segment joints
print(f"\n[5] Creating {NUM_SEGMENTS-1} joints...")
joint_offset = SEGMENT_LENGTH * 0.45

for i in range(NUM_SEGMENTS - 1):
    joint_path = joints_scope.AppendChild(f"joint_{i:02d}")
    create_d6_joint(
        joint_path,
        segment_paths[i],
        segment_paths[i + 1],
        Gf.Vec3f(joint_offset, 0, 0),
        Gf.Vec3f(-joint_offset, 0, 0)
    )
print("    Done")

# Step 6: Connect to hooks
print("\n[6] Connecting to hooks...")

right_4_path = wire_path.AppendChild("right_4")
if stage.GetPrimAtPath(right_4_path):
    create_d6_joint(
        joints_scope.AppendChild("joint_right4_to_rope"),
        right_4_path,
        segment_paths[0],
        Gf.Vec3f(0.015, 0, 0),
        Gf.Vec3f(-joint_offset, 0, 0)
    )
    print("    right_4 -> seg_00: Done")

left_4_path = wire_path.AppendChild("left_4")
if stage.GetPrimAtPath(left_4_path):
    create_d6_joint(
        joints_scope.AppendChild("joint_rope_to_left4"),
        segment_paths[-1],
        left_4_path,
        Gf.Vec3f(joint_offset, 0, 0),
        Gf.Vec3f(-0.015, 0, 0)
    )
    print("    seg_13 -> left_4: Done")

# Step 7: Save
print(f"\n[7] Saving to: {OUTPUT_USD}")
stage.GetRootLayer().Export(OUTPUT_USD)
print("    Done")

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)
print(f"""
Created:
  - {NUM_SEGMENTS} rope segments
  - {NUM_SEGMENTS - 1} inter-segment joints
  - 2 hook connections

Output: {OUTPUT_USD}

To test in Isaac Sim GUI:
  File -> Open -> {OUTPUT_USD}
  Press PLAY (Space)
""")

# Close
simulation_app.close()
