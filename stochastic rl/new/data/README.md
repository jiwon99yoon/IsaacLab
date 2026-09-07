# TensorBoard CSV 플로터 사용법

## 🎯 간단 사용법

### 1. TensorBoard에서 CSV 다운로드
```bash
tensorboard --logdir=/home/dyros/IsaacLab/logs/rl_games/stochastic_rl
```
- 브라우저에서 원하는 그래프 열기
- 왼쪽 하단 "Show data download links" 클릭
- CSV 다운로드
- 파일명 변경 (예: `shaped_reward_dg5f.csv`)

### 2. Python 스크립트 실행
```bash
python plot_comparison.py \
    --dg5f shaped_reward_dg5f.csv \
    --inspire shaped_reward_inspire.csv \
    --ylabel "Total Shaped Reward" \
    --output shaped_reward.png
```

## 📋 CSV 파일 형식

TensorBoard CSV는 자동으로 이 형식으로 저장됩니다:
```csv
Wall time,Step,Value
1234567890.123,0,0.523
1234567890.456,36,1.234
...
```

**중요**: 스크립트는 `Step` (x축)과 `Value` (y축) 컬럼만 사용합니다.

## 🎨 파라미터 설명

| 파라미터 | 설명 | 예시 |
|---------|------|------|
| `--dg5f` | DG5F CSV 파일 경로 | `shaped_reward_dg5f.csv` |
| `--inspire` | Inspire CSV 파일 경로 | `shaped_reward_inspire.csv` |
| `--ylabel` | Y축 라벨 (자유롭게 입력) | `"Total Shaped Reward"` |
| `--output` | 출력 PNG 파일명 | `shaped_reward.png` |
| `--smoothing` | (선택) 부드러움 정도 (0-1) | `0.95` (기본값) |

## 📊 Smoothing 파라미터

- **0.95** (기본값): 대부분의 reward 그래프에 적합
- **0.90**: Loss 계열 (덜 부드럽게)
- **0.98**: Curriculum 계열 (더 부드럽게)

## 🚀 빠른 시작 예제

### Shaped Reward (Figure 6)
```bash
python plot_comparison.py \
    --dg5f shaped_reward_dg5f.csv \
    --inspire shaped_reward_inspire.csv \
    --ylabel "Total Shaped Reward" \
    --output fig6_shaped_reward.png
```

### Entropy (Figure 10a)
```bash
python plot_comparison.py \
    --dg5f entropy_dg5f.csv \
    --inspire entropy_inspire.csv \
    --ylabel "Policy Entropy" \
    --output fig10a_entropy.png
```

### Critic Loss (Figure 10b)
```bash
python plot_comparison.py \
    --dg5f critic_loss_dg5f.csv \
    --inspire critic_loss_inspire.csv \
    --ylabel "Critic Loss (TD Error)" \
    --output fig10b_critic_loss.png \
    --smoothing 0.90
```

## 🔧 일괄 실행

`plot_examples.sh` 파일에 모든 그래프를 생성하는 명령어가 들어있습니다:
```bash
chmod +x plot_examples.sh
# 파일 열어서 필요한 명령어만 복사해서 실행
```

## 📁 필요한 CSV 파일 목록

### Figure 6: Shaped Reward
- `shaped_reward_dg5f.csv`
- `shaped_reward_inspire.csv`

### Figure 10: Learning Dynamics (4개)
- `entropy_dg5f.csv` / `entropy_inspire.csv`
- `critic_loss_dg5f.csv` / `critic_loss_inspire.csv`
- `kl_divergence_dg5f.csv` / `kl_divergence_inspire.csv`
- `episode_length_dg5f.csv` / `episode_length_inspire.csv`

### Figure 11: ADR Curriculum
- `adr_curriculum_dg5f.csv` / `adr_curriculum_inspire.csv`

## 🎯 출력물

- **PNG 파일**: 300 DPI, publication-quality
- **통계**: 터미널에 final value 출력
- **미리보기**: 자동으로 그래프 창 표시

## ⚠️ Troubleshooting

### "missing 'Step' or 'Value' columns"
→ CSV 파일이 TensorBoard 형식이 아닙니다. 첫 줄에 `Step,Value`가 있는지 확인하세요.

### "FileNotFoundError"
→ CSV 파일 경로를 확인하세요. 상대 경로 또는 절대 경로 모두 가능합니다.

### 그래프가 너무 noisy함
→ `--smoothing` 값을 올려보세요 (예: 0.97)

### 그래프가 너무 뭉개짐
→ `--smoothing` 값을 낮춰보세요 (예: 0.90)
