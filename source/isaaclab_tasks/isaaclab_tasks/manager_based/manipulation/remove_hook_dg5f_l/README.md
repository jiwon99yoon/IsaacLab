# Remove Hook 태스크 - HDR35_20 + DG5F_L

## 태스크 설명

단일 암 로봇(HDR35_20 + DG5F_L 왼손)이 스프링 어셈블리에서 와이어 훅을 잡아서 제거하는 매니퓰레이션 태스크입니다.

- **로봇**: HDR35_20 암 + DG5F_L Tesollo 왼손 (총 26 DoF: 암 6 + 핸드 20)
- **오브젝트**: 스프링과 와이어 훅이 있는 전면 샤시 어셈블리
- **타겟 훅**: `right_ring` (왼손이 오른쪽 훅을 타겟)
- **목표**: 훅을 잡고, 스프링에서 제거한 후, 목표 위치로 이동 (오른쪽 0.2m, 위 0.3m)

## 학습 명령어

```bash
python scripts/reinforcement_learning/rl_games/train.py --task Isaac-RemoveHook-Hdr35-DG5F-v0
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

## Dexsuite (iiwa)와의 주요 차이점

| 파라미터 | Dexsuite (iiwa) | Remove Hook (HDR35) | 비고 |
|-----------|-----------------|---------------------|------|
| 암 Stiffness | Kp = 300 | Kp = 30,000 | 100배 |
| Action Scale | 0.1 | 0.001 | 1/100 (토크 일치) |
| 속도한계 Multiplier | 2 | 2 | 동일 |
| 보상 std (reach) | 0.1 | 1.0 | - |
| 보상 std (target) | 0.02 | 0.2 | - |

**토크 일치**: `τ = Kp × scale` → iiwa: 300×0.1=30, HDR35: 30000×0.001=30

## 최근 변경사항 (2026-01-12)

자세한 기술적 분석은 [ANALYZE.md](./ANALYZE.md) 참조.
