




iiwa_allegro와 hdr35_20과 rh56f1 / dg5f 비교분석
● Stiffness 비교 분석

  암(Arm) 비교
  ┌─────────────────┬─────────────┬──────────┬───────┐
  │    파라미터     │ iiwa (Kuka) │ HDR35_20 │ 비율  │
  ├─────────────────┼─────────────┼──────────┼───────┤
  │ j1-j4 Stiffness │ 300         │ 30,000   │ 100배 │
  ├─────────────────┼─────────────┼──────────┼───────┤
  │ j5 Stiffness    │ 100         │ 3,000    │ 30배  │
  ├─────────────────┼─────────────┼──────────┼───────┤
  │ j6 Stiffness    │ 50          │ 1,500    │ 30배  │
  ├─────────────────┼─────────────┼──────────┼───────┤
  │ j7 Stiffness    │ 25          │ -        │ -     │
  ├─────────────────┼─────────────┼──────────┼───────┤
  │ j1-j4 Damping   │ 45          │ 2,000    │ 44배  │
  └─────────────────┴─────────────┴──────────┴───────┘
  핸드(Hand) 비교
  ┌──────────────┬─────────┬────────┬──────────┐
  │   파라미터   │ Allegro │ DG5F_L │ RH56F1_R │
  ├──────────────┼─────────┼────────┼──────────┤
  │ Stiffness    │ 3.0     │ 4.0    │ 2.5      │
  ├──────────────┼─────────┼────────┼──────────┤
  │ Damping      │ 0.1     │ 0.3    │ 0.25     │
  ├──────────────┼─────────┼────────┼──────────┤
  │ Effort Limit │ 0.5     │ 5.0    │ 1.0      │
  └──────────────┴─────────┴────────┴──────────┘
  암 vs 핸드 Stiffness 비율
  ┌────────────────┬──────────────┬────────────────┬────────────────┐
  │   로봇 조합    │ 암 Stiffness │ 핸드 Stiffness │ 비율 (암/핸드) │
  ├────────────────┼──────────────┼────────────────┼────────────────┤
  │ iiwa + Allegro │ 300          │ 3.0            │ 100배          │
  ├────────────────┼──────────────┼────────────────┼────────────────┤
  │ HDR35 + DG5F   │ 30,000       │ 4.0            │ 7,500배        │
  ├────────────────┼──────────────┼────────────────┼────────────────┤
  │ HDR35 + RH56F1 │ 30,000       │ 2.5            │ 12,000배       │
  └────────────────┴──────────────┴────────────────┴────────────────┘
  결론

  문제점 발견:
  - iiwa+Allegro: 암/핸드 stiffness 비율 = 100배
  - HDR35+DG5F: 암/핸드 stiffness 비율 = 7,500배 (75배 차이!)

  같은 action scale(0.1)을 쓰면:
  iiwa 토크: 300 × 0.1 = 30 Nm
  Allegro 토크: 3 × 0.1 = 0.3 Nm → 비율 100:1 ✓

  HDR35 토크: 30,000 × 0.1 = 3,000 Nm (너무 큼!)
  DG5F 토크: 4 × 0.1 = 0.4 Nm → 비율 7,500:1 ✗

  권장: 다른 action scale 사용
  # 암: scale = 0.001 (토크 30 Nm)
  # 핸드: scale = 0.1 (토크 0.4 Nm)


# remove_hook 환경 관련 issue 정리 (01071108 updated)

- hook_remove되도록 초기 자세 설정해야함



# issue관련해서 수정해야할 것
1. 환경 16개 -> 256개로 변경해야함
2. dg5f model 작성하기
3. wire_reset 관련 이슈 : 학습도중 wire가 날라가는 현상이 있음 <- 이거 reset 및 termination해야하는데 이를 위해선 spawn하는 usd(wire_revolute_collision_flattened)의 전반적 수정이 필요 
    - wire가 회전된 상태로 시작되도록 해야함 - rod 기준
    - 중간에 deformable object설정 
    -> reset_object와 reset_object_joints (stop here 01071616으로 주석처리해둠)

4. dexsuite 환경에 대한 분석 필요 :" action scale 등


stiffness의 차이?
각 joint마다 회전하는 것의 차이 

action scale의 경우 



# long term 수정 (or 할 것) 
9. ati sensor들어가있는 usd 완성 (urdf부터 완성해서 temp.usd로)
10. curobo와 연동
11. IsaacRos
12. 현재 IsaacLab엔 PD 제어기가 짜져있음. PD가 아닌 중력보상이나 Dynamic가능한지 check 





----------------------------------
# 이전 issue 기록해두기

1. 샤시모듈 : 원래 가운데 frame, left_strut, right_strut로 나눠서 (rigid_body로 설정, fixed joint로 관리했었는데) -> 이걸 Isaaclab에서 불러오는데 오류가 뜸
    rigid body가 아닌 그냥 collision check용 샤시모듈 usd file 제작 (env_diated_decomposed_chassis_tilted_flattened_no_rigid_instanceable.usd)
2. reward function 중 현재 right_ring으로 object_pos, ori 되어있는거 left_ring기준으로 바꿔야함 
- rh56f1_r이 달린 모델은 left_ring을 타겟하도록, dg5f_l이 달린 모델은 right_ring을 타겟하도록

3. observation에 어떤 것들이 있는지 check하고 실제 데이터로 얻을 수 있는 정보만 변경

4. 현재 contact force는 wire에 닿기만 하면 접촉점수를 받음 -> 이걸 left_ring으로 바꿈

5. wire의 usd 문제 
left_ring에 속한 RevoluteJoint의 

------------------------------------
# 큰 틀에서 앞으로 Todo

일단 현재 policy - history (5) tracking module 세팅은 해두었는데, 우리는 zivid camera로 위치 인식하고 이걸 단단하게 잡았다(titily grasp : 하나의 물체처럼)고 가정하면, vision 정보가 딱히 필요없음
- 우리 환경 구상하는 것과는 별개로 일단 isaaclab을 위한 전반적 framework를 구축했다 정도로 생각
- 이제 단단한 grasp가 있을 때 이걸 빼내는 학습 방법 생각하려함
- f/t sensor를 불러와서 isaacsim내에서 접근할 수 있는 코드에 대해선 실험해둔바있는데 이걸 어떻게 사용할지 고민

- 중간 rigid body -> deformable object가 되도록 설정해야함
