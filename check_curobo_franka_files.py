#!/usr/bin/env python3
"""
CuRobo가 사용하는 Franka Panda 파일 확인
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

print("\n" + "=" * 70)
print("Checking Franka Panda files used by CuRobo")
print("=" * 70)

# 1. URDF 파일 경로
urdf_nucleus_path = f"{ISAACLAB_NUCLEUS_DIR}/Controllers/SkillGenAssets/FrankaPanda/franka_panda.urdf"
print(f"\n1. URDF file (from Nucleus):")
print(f"   Nucleus path: {urdf_nucleus_path}")

# URDF 다운로드
try:
    local_urdf = retrieve_file_path(urdf_nucleus_path, force_download=False)
    print(f"   ✓ Local path: {local_urdf}")

    if os.path.exists(local_urdf):
        file_size = os.path.getsize(local_urdf)
        print(f"   ✓ File exists, size: {file_size} bytes")

        # URDF 내용 일부 확인
        with open(local_urdf, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            print(f"   ✓ Total lines: {len(lines)}")

            # Robot name 확인
            for line in lines[:50]:
                if 'robot name' in line.lower():
                    print(f"   Robot definition: {line.strip()}")
                    break

            # Joint 개수 확인
            joint_count = content.count('<joint name=')
            link_count = content.count('<link name=')
            print(f"   Links: {link_count}, Joints: {joint_count}")
    else:
        print(f"   ✗ File not found at {local_urdf}")
except Exception as e:
    print(f"   ✗ Error retrieving URDF: {e}")
    import traceback
    traceback.print_exc()

# 2. CuRobo robot config 파일
print(f"\n2. CuRobo robot config YAML:")
print(f"   Base template: franka.yml (from CuRobo library)")

try:
    import curobo
    curobo_path = os.path.dirname(curobo.__file__)
    robot_config_dir = os.path.join(curobo_path, "content", "configs", "robot")
    franka_yml = os.path.join(robot_config_dir, "franka.yml")

    print(f"   CuRobo library path: {curobo_path}")

    if os.path.exists(franka_yml):
        print(f"   ✓ Found: {franka_yml}")

        # YAML 내용 확인
        with open(franka_yml, 'r') as f:
            lines = f.readlines()
            print(f"   Content preview (first 25 lines):")
            for i, line in enumerate(lines[:25], 1):
                print(f"     {i:2d}: {line.rstrip()}")
    else:
        print(f"   ✗ Not found at {franka_yml}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 3. Collision spheres 파일
print(f"\n3. Collision spheres configuration:")

try:
    import curobo
    curobo_path = os.path.dirname(curobo.__file__)
    spheres_dir = os.path.join(curobo_path, "content", "configs", "robot", "spheres")
    franka_spheres = os.path.join(spheres_dir, "franka_mesh.yml")

    if os.path.exists(franka_spheres):
        print(f"   ✓ Found: {franka_spheres}")

        with open(franka_spheres, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            print(f"   ✓ Total lines: {len(lines)}")

            # Sphere 개수 확인
            center_count = content.count('center:')
            position_count = content.count('position:')
            print(f"   Sphere definitions: ~{max(center_count, position_count)}")
    else:
        print(f"   ✗ Not found at {franka_spheres}")

        # 대체 파일 검색
        if os.path.exists(spheres_dir):
            print(f"   Available sphere files:")
            for file in os.listdir(spheres_dir):
                if 'franka' in file.lower():
                    print(f"     - {file}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 4. Temporary files check
print(f"\n4. Checking /tmp for CuRobo temporary files:")
tmp_files = []
try:
    for item in os.listdir('/tmp'):
        if 'franka' in item.lower() or 'curobo' in item.lower():
            full_path = os.path.join('/tmp', item)
            tmp_files.append((item, full_path))

    if tmp_files:
        print(f"   Found {len(tmp_files)} related files/dirs in /tmp:")
        for name, path in sorted(tmp_files)[:15]:  # 처음 15개만
            if os.path.isfile(path):
                size = os.path.getsize(path)
                print(f"     [FILE] {name} ({size} bytes)")
            else:
                try:
                    file_count = len(os.listdir(path))
                    print(f"     [DIR]  {name}/ ({file_count} files)")
                except:
                    print(f"     [DIR]  {name}/")
    else:
        print(f"   No CuRobo/Franka files found in /tmp")
except Exception as e:
    print(f"   Error accessing /tmp: {e}")

# 5. 환경 변수 확인
print(f"\n5. Environment:")
print(f"   ISAACLAB_NUCLEUS_DIR: {ISAACLAB_NUCLEUS_DIR}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
CuRobo uses these files for Franka Panda:
1. URDF: Downloaded from Nucleus to /tmp
   - Path: {NUCLEUS}/Controllers/SkillGenAssets/FrankaPanda/franka_panda.urdf

2. Robot config: franka.yml from CuRobo library
   - Base template copied and modified in /tmp
   - URDF path updated to point to downloaded file

3. Collision spheres: franka_mesh.yml from CuRobo library
   - Defines collision spheres for each link

IMPORTANT:
- If you modified Franka in IsaacLab assets, CuRobo won't see it
- CuRobo uses its own URDF from Nucleus
- To use custom Franka, you need to:
  a) Modify the Nucleus URDF, OR
  b) Pass custom URDF path to CuroboPlannerCfg
""")

simulation_app.close()
