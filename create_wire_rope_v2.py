#!/usr/bin/env python3
"""
Wire Bar to Rigid-Body Rope Conversion Script (v2 - Fixed)

Fixes:
  - Rope segments at /wire_model/rope_segments (not under /wire_model/wire)
  - Proper joint body path references
  - Flattened USD export

Usage:
    cd ~/isaacsim
    ./python.sh /home/dyros/IsaacLab/create_wire_rope_v2.py
"""

import argparse
from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

import omni.usd
from pxr import UsdGeom, Sdf, Gf, UsdPhysics, UsdShade, PhysxSchema, Usd, UsdUtils
from omni.physx.scripts import physicsUtils

print("="*60)
print("Wire Bar -> Rigid-Body Rope Conversion (v2)")
print("="*60)

# Configuration
INPUT_USD = "/home/dyros/IsaacLab/hyundai/1_factory_usd_file/flattended_usd/wire_revolute_collision_flattened.usd"
OUTPUT_USD = "/home/dyros/IsaacLab/hyundai/1_factory_usd_file/flattended_usd/wire_rope_0.7m_flattened.usd"

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

# Open USD
print(f"\nOpening: {INPUT_USD}")
omni.usd.get_context().open_stage(INPUT_USD)
stage = omni.usd.get_context().get_stage()

if not stage:
    print("ERROR: Failed to open stage!")
    simulation_app.close()
    exit(1)

# Paths - rope_segments at TOP LEVEL (not under /wire)
base_path = Sdf.Path("/wire_model")
wire_path = base_path.AppendChild("wire")

# IMPORTANT: Create rope at top level to avoid RigidBody hierarchy conflict
rope_scope = base_path.AppendChild("rope_segments")
joints_scope = base_path.AppendChild("rope_joints")
material_path = base_path.AppendChild("RopeMaterial")


def create_rope_segment(path, position, color):
    """Create one rope segment (capsule) with RigidBody"""
    capsule = UsdGeom.Capsule.Define(stage, path)
    capsule.CreateHeightAttr(SEGMENT_LENGTH * 0.85)
    capsule.CreateRadiusAttr(SEGMENT_RADIUS)
    capsule.CreateAxisAttr("X")
    capsule.AddTranslateOp().Set(position)
    capsule.CreateDisplayColorAttr().Set([color])

    prim = capsule.GetPrim()

    # RigidBody API
    rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)

    # Mass API
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateDensityAttr().Set(DENSITY)

    # Collision API
    UsdPhysics.CollisionAPI.Apply(prim)
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx_collision.CreateRestOffsetAttr().Set(0.0)
    physx_collision.CreateContactOffsetAttr().Set(0.002)

    # Material binding
    physicsUtils.add_physics_material_to_prim(stage, prim, material_path)

    return prim


def create_d6_joint(joint_path, body0_path, body1_path, local_pos0, local_pos1):
    """Create D6 joint between two rigid bodies"""
    joint = UsdPhysics.Joint.Define(stage, joint_path)

    # Set body relationships using string paths
    body0_rel = joint.CreateBody0Rel()
    body0_rel.SetTargets([body0_path])

    body1_rel = joint.CreateBody1Rel()
    body1_rel.SetTargets([body1_path])

    # Local positions
    joint.CreateLocalPos0Attr().Set(local_pos0)
    joint.CreateLocalPos1Attr().Set(local_pos1)

    # Local rotations (identity)
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    d6_prim = joint.GetPrim()

    # Lock translation and rotX
    for dof in ["transX", "transY", "transZ", "rotX"]:
        limit_api = UsdPhysics.LimitAPI.Apply(d6_prim, dof)
        limit_api.CreateLowAttr(1.0)
        limit_api.CreateHighAttr(-1.0)

    # Unlock rotY, rotZ
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
    print("    Done - bar hidden")
else:
    print("    Warning: bar not found")

# Also hide bar_case if exists
bar_case = stage.GetPrimAtPath(wire_path.AppendChild("bar_case"))
if bar_case and bar_case.IsValid():
    UsdGeom.Imageable(bar_case).MakeInvisible()
    print("    Done - bar_case hidden")

# Step 2: Create physics material
print("\n[2] Creating physics material...")
if not stage.GetPrimAtPath(material_path):
    UsdShade.Material.Define(stage, material_path)
    mat = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(material_path))
    mat.CreateStaticFrictionAttr().Set(0.5)
    mat.CreateDynamicFrictionAttr().Set(0.5)
    mat.CreateRestitutionAttr().Set(0.0)
print("    Done")

# Step 3: Create scopes at TOP LEVEL
print("\n[3] Creating scopes at /wire_model level...")
if not stage.GetPrimAtPath(rope_scope):
    UsdGeom.Scope.Define(stage, rope_scope)
    print(f"    Created: {rope_scope}")
if not stage.GetPrimAtPath(joints_scope):
    UsdGeom.Scope.Define(stage, joints_scope)
    print(f"    Created: {joints_scope}")

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
    print(f"    seg_{i:02d} at x={x:.4f}")

print(f"    Created {NUM_SEGMENTS} segments")

# Step 5: Create inter-segment joints
print(f"\n[5] Creating {NUM_SEGMENTS-1} inter-segment joints...")
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

# Step 6: Connect to hooks (right_4, left_4)
print("\n[6] Connecting to hooks...")

# Check if right_4 exists and connect
right_4_path = wire_path.AppendChild("right_4")
right_4_prim = stage.GetPrimAtPath(right_4_path)
if right_4_prim and right_4_prim.IsValid():
    # Make sure right_4 has RigidBodyAPI
    if not right_4_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(right_4_prim)
        print(f"    Added RigidBodyAPI to right_4")

    joint_right = joints_scope.AppendChild("joint_right4_to_seg00")
    create_d6_joint(
        joint_right,
        right_4_path,
        segment_paths[0],
        Gf.Vec3f(0.015, 0, 0),
        Gf.Vec3f(-joint_offset, 0, 0)
    )
    print(f"    Connected: right_4 -> seg_00")
else:
    print(f"    Warning: right_4 not found at {right_4_path}")

# Check if left_4 exists and connect
left_4_path = wire_path.AppendChild("left_4")
left_4_prim = stage.GetPrimAtPath(left_4_path)
if left_4_prim and left_4_prim.IsValid():
    # Make sure left_4 has RigidBodyAPI
    if not left_4_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(left_4_prim)
        print(f"    Added RigidBodyAPI to left_4")

    joint_left = joints_scope.AppendChild("joint_seg13_to_left4")
    create_d6_joint(
        joint_left,
        segment_paths[-1],
        left_4_path,
        Gf.Vec3f(joint_offset, 0, 0),
        Gf.Vec3f(-0.015, 0, 0)
    )
    print(f"    Connected: seg_13 -> left_4")
else:
    print(f"    Warning: left_4 not found at {left_4_path}")

# Step 7: Save as FLATTENED USD
print(f"\n[7] Saving flattened USD...")

# First, flatten the stage
print("    Flattening stage...")
flattened_layer = stage.Flatten()

# Export the flattened layer
print(f"    Exporting to: {OUTPUT_USD}")
flattened_layer.Export(OUTPUT_USD)

print("    Done")

# Summary
print("\n" + "="*60)
print("COMPLETE!")
print("="*60)
print(f"""
Created:
  - {NUM_SEGMENTS} rope segments at /wire_model/rope_segments/
  - {NUM_SEGMENTS - 1} inter-segment D6 joints
  - 2 hook connection joints (right_4, left_4)
  - bar mesh hidden

Output (flattened): {OUTPUT_USD}

Structure:
  /wire_model
  ├── wire (existing - bar hidden)
  │   ├── left_4 (connected to seg_13)
  │   └── right_4 (connected to seg_00)
  ├── rope_segments/
  │   ├── seg_00 ... seg_13
  ├── rope_joints/
  │   ├── joint_00 ... joint_12
  │   ├── joint_right4_to_seg00
  │   └── joint_seg13_to_left4
  ├── right_ring
  └── left_ring

To test:
  1. Open Isaac Sim GUI
  2. File -> Open -> {OUTPUT_USD}
  3. Press PLAY (Space)
""")

simulation_app.close()
