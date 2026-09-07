# UR10e + ATI FT Sensor + DG5F/Inspire Hand USD 생성 가이드

## 준비된 파일들

```bash
# UR10e (Isaac Sim assets)
/home/dyros/isaacsim_assets/Assets/Isaac/5.0/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd

# 이미 결합된 참고 파일
/home/dyros/IsaacLab/hyundai/ur10_ati_inspire.usd  # UR10 + ATI + Inspire (참고!)

# DG5F Hand
/home/dyros/IsaacLab/hyundai/dg_description/urdf/dg5f_right_new/dg5f_right_new.usd
/home/dyros/IsaacLab/hyundai/dg_description/urdf/dg5f_right_new.urdf

# Inspire Hand (이미 UR10과 결합됨)
/home/dyros/IsaacLab/hyundai/inspire/URDF/RH56F1_L/urdf/RH56F1_L.urdf

# ATI FT Sensor
/home/dyros/IsaacLab/hyundai/inspire/attached_torque_sensor-tashan/TS-F-A.usd
```

---

## 방법 1: Isaac Sim GUI에서 조립 (강력 권장!)

### Step 1: Isaac Sim 실행
```bash
cd ~/isaacsim
./isaac-sim.sh
```

### Step 2: 새 Stage 생성
1. File → New
2. Physics Scene 설정 확인 (자동으로 추가됨)

### Step 3: UR10e 불러오기
1. **Drag & Drop 방식:**
   - Content Browser에서 `ur10e.usd` 찾기
   - Stage로 드래그

   또는

2. **Add Reference 방식:**
   - Stage 패널에서 우클릭
   - Add → Reference
   - Path: `/home/dyros/isaacsim_assets/Assets/Isaac/5.0/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd`

### Step 4: DG5F Hand 불러오기
1. Stage 패널에서 UR10e의 `wrist_3_link` 찾기
2. `wrist_3_link`에 우클릭 → Add → Reference
3. Path: `/home/dyros/IsaacLab/hyundai/dg_description/urdf/dg5f_right_new/dg5f_right_new.usd`

### Step 5: ATI FT Sensor 추가 (선택사항)
1. `wrist_3_link`와 DG5F 사이에 FT sensor 삽입
2. Reference 추가: `TS-F-A.usd`

### Step 6: Fixed Joint 생성
1. **Physics → Joints → Fixed Joint** 선택
2. **Body 0:** `/ur10e/wrist_3_link`
3. **Body 1:** `/dg5f_hand/base_link` (또는 DG5F의 root link)
4. Position/Orientation 조정:
   - Transform 값 조정하여 손이 올바르게 정렬되도록 함

### Step 7: 상대 Transform 조정
1. DG5F prim 선택
2. Property 패널에서 Transform 조정:
   - **Translation:** (0, 0, 0.1) 정도에서 시작
   - **Rotation:** (0, 0, 0) 또는 필요에 따라 조정
3. Viewport에서 시각적으로 확인하며 fine-tuning

### Step 8: 저장
```
File → Save As
Path: /home/dyros/IsaacLab/hyundai/ur10e_ati_dg5f.usd
```

---

## 방법 2: URDF 먼저 결합 후 USD 변환 (더 정확함)

### Step 1: UR10e + DG5F URDF 결합

UR10e URDF 찾기:
```bash
find /home/dyros/isaacsim_assets -name "*.urdf" | grep ur10
```

URDF 결합 스크립트 (`combine_ur10e_dg5f.urdf.xacro`):
```xml
<?xml version="1.0"?>
<robot name="ur10e_dg5f" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Import UR10e -->
  <xacro:include filename="$(find ur_description)/urdf/ur10e.urdf.xacro"/>
  <xacro:ur10e_robot prefix="" joint_limits_parameters_file="..."/>

  <!-- Import DG5F Hand -->
  <xacro:include filename="/home/dyros/IsaacLab/hyundai/dg_description/urdf/dg5f_right_new.urdf"/>

  <!-- Attach hand to wrist -->
  <joint name="wrist_to_hand" type="fixed">
    <parent link="tool0"/>
    <child link="dg5f_base_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>

</robot>
```

### Step 2: URDF를 USD로 변환 (Isaac Sim)
```bash
# Isaac Sim URDF Importer 사용
# File → Import → URDF
# Select: combined URDF file
# Import Settings:
#   - Fix Base Link: False (mobile robot이 아니므로)
#   - Self Collision: True
#   - Import Inertia Tensor: True
# Click Import
```

---

## 방법 3: 기존 ur10_ati_inspire.usd를 복사하여 Inspire → DG5F 교체

가장 빠른 방법! 이미 ur10_ati_inspire.usd가 있으므로:

### Step 1: Isaac Sim에서 열기
```bash
cd ~/isaacsim
./isaac-sim.sh
# File → Open → /home/dyros/IsaacLab/hyundai/ur10_ati_inspire.usd
```

### Step 2: Inspire Hand 제거
1. Stage 패널에서 Inspire hand prim 찾기
2. 선택 후 Delete

### Step 3: DG5F 추가
1. wrist_3_link에 DG5F reference 추가 (위 Step 4 참고)
2. Fixed Joint 다시 생성
3. Transform 조정

### Step 4: Save As
```
File → Save As
Path: /home/dyros/IsaacLab/hyundai/ur10_ati_dg5f.usd
```

---

## 검증 방법

### Isaac Sim에서 테스트:
1. **Articulation Inspector 열기:**
   - Window → Simulation → Articulation Inspector
2. **Joint Control 패널:**
   - 각 조인트를 수동으로 움직여 보기
   - UR10e: 6 joints
   - DG5F: 20 joints
   - Total: 26 DOF 확인
3. **Physics Simulation 실행:**
   - Play 버튼 클릭
   - 로봇이 떨어지지 않고 안정적인지 확인
4. **Self-Collision 확인:**
   - Joint를 극단적으로 움직여서 self-collision이 발생하는지 확인

### Isaac Lab에서 테스트:
```bash
cd /home/dyros/IsaacLab

# 테스트 스크립트 실행
python source/isaaclab_assets/isaaclab_assets/robots/test_ur10e_dg5f.py
```

---

## 트러블슈팅

### 문제 1: 손이 떨어짐 (Fixed Joint 없음)
**원인:** Fixed joint가 제대로 생성되지 않음
**해결:** Physics → Joints → Fixed Joint 다시 생성

### 문제 2: 손 위치가 맞지 않음
**원인:** Transform이 잘못됨
**해결:** DG5F prim의 Translation/Rotation 조정

### 문제 3: Self-collision 발생
**원인:** DG5F와 UR10e wrist가 겹침
**해결:**
- DG5F의 Translation.z 값 증가 (예: 0.05 → 0.1)
- Collision filter group 설정

### 문제 4: Joint가 제어되지 않음
**원인:** Articulation Root가 제대로 설정되지 않음
**해결:**
- Stage 패널에서 root prim 선택
- Physics → Articulation Root 추가
- Enabled Self Collisions: True

---

## 최종 파일 위치

생성될 파일들:
```
/home/dyros/IsaacLab/hyundai/ur10e_ati_dg5f.usd      # UR10e + ATI + DG5F
/home/dyros/IsaacLab/hyundai/ur10_ati_dg5f.usd       # UR10 + ATI + DG5F
/home/dyros/IsaacLab/hyundai/ur10e_ati_inspire.usd   # UR10e + ATI + Inspire (새로 만들기)
```

기존 파일:
```
/home/dyros/IsaacLab/hyundai/ur10_ati_inspire.usd    # UR10 + ATI + Inspire (이미 있음!)
```

---

## 다음 단계: Isaac Lab 환경 설정

USD 파일이 준비되면:
1. `source/isaaclab_assets/isaaclab_assets/robots/ur10e_dg5f.py` 생성
2. `source/isaaclab_tasks/.../dexsuite_ur10e/` 환경 생성
3. Contact sensor 설정
4. 학습 시작!

---

## 참고 자료

- Isaac Sim Documentation: https://docs.omniverse.nvidia.com/isaacsim/latest/
- URDF Importer: https://docs.omniverse.nvidia.com/isaacsim/latest/features/environment_setup/ext_omni_isaac_urdf.html
- Isaac Lab Assets: https://isaac-sim.github.io/IsaacLab/source/refs/assets.html
