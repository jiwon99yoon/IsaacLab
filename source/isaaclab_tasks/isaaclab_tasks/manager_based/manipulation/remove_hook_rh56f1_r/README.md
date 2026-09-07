# Remove Hook 태스크 - HDR35_20 + RH56F1_R

## 태스크 설명

단일 암 로봇(HDR35_20 + RH56F1_R 오른손)이 스프링 어셈블리에서 와이어 훅을 잡아서 제거하는 매니퓰레이션 태스크입니다.

- **로봇**: HDR35_20 암 + RH56F1_R Inspire 오른손 (총 12 DoF: 암 6 + 핸드 6)
- **오브젝트**: 스프링과 와이어 훅이 있는 전면 샤시 어셈블리
- **타겟 훅**: `left_ring` (오른손이 왼쪽 훅을 타겟)
- **목표**: 훅을 잡고, 스프링에서 제거한 후, 목표 위치로 이동 (왼쪽 0.2m, 위 0.3m)

## 학습 명령어

```bash
python scripts/reinforcement_learning/rl_games/train.py --task Isaac-RemoveHook-Hdr35-RH56F1-v0
```

## 환경 설정

### Action Space
- **타입**: Relative Joint Position
- **Scale**: 스텝당 0.001 rad (액션당 최대 변화량, iiwa와 동일 토크)
- **조인트**: 모든 조인트 (`.*`)

### Observation Space
- 조인트 위치 및 속도
- 핸드 팁 상태 (손바닥 + 손가락 끝)
- 로봇 기준 오브젝트(훅) 자세
- 손가락 끝 접촉력

### 보상 구조
1. **fingers_to_hook** (weight=2.0): 손에서 훅까지의 거리
2. **good_finger_contact** (weight=3.0): 손가락 끝과 훅 사이의 접촉
3. **hook_to_target** (weight=10.0): 훅에서 목표 위치까지의 거리

### 종료 조건
- **time_out**: 8초
- **object_out_of_bound**: 훅 위치가 유효 범위 밖
- **abnormal_robot**: 조인트 속도가 한계 × 10 초과

## DG5F_L 변형과의 주요 차이점

| 파라미터 | DG5F_L (왼손) | RH56F1_R (오른손) |
|-----------|---------------|------------------|
| 핸드 DoF | 20 (fully-actuated) | 6 (under-actuated) |
| 타겟 훅 | right_ring | left_ring |
| 타겟 오프셋 | (+0.2, 0, +0.3) | (-0.2, 0, +0.3) |
| Palm Body | ll_dg_palm | gripper_base_link |
| 손가락 끝 | ll_dg_X_4 | right_xxx_X |

## 최근 변경사항 (2026-01-12)

자세한 기술적 분석은 [ANALYZE.md](./ANALYZE.md) 참조.

---

## 파일 구조

```
remove_hook_rh56f1_r/
├── __init__.py                 # 태스크 등록
├── remove_hook_env_cfg.py      # 기본 환경 설정
├── README.md                   # 개요 문서 (이 파일)
├── ANALYZE.md                  # 기술적 분석
├── config/
│   └── hdr35_20_rh56f1_r/
│       ├── __init__.py
│       ├── remove_hdr35_20_rh56f1_r_env_cfg.py  # 로봇별 설정
│       └── agents/
│           └── rl_games_ppo_cfg.yaml
└── mdp/
    ├── __init__.py
    ├── commands.py             # 타겟 위치 명령
    ├── observations.py         # 관측 함수
    ├── rewards.py              # 보상 함수
    └── terminations.py         # 종료 조건
```

---

**최종 업데이트:** 2026-01-12
