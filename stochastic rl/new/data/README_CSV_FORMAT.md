# CSV 파일 포맷 가이드

## 📋 CSV 파일 형식

TensorBoard에서 다운로드한 CSV 파일은 다음 형식이어야 합니다:

### Shaped Reward CSV 예시:
```csv
step,shaped_reward
0,0.523
36,1.234
72,1.456
108,2.123
...
```

### 기타 Metric CSV 예시:
```csv
step,entropy
0,35.234
36,36.123
72,35.987
...
```

## 📂 필요한 CSV 파일 목록

### Figure 6: Shaped Reward (이미 완료)
- `shaped_step_reward_dg5f.csv` ← **step**, **shaped_reward** 컬럼
- `shaped_step_reward_inspire.csv`

### Figure 9: Reward Decomposition (Bar Chart)
epoch 7500 시점의 값들만 필요하므로 수동으로 표 만들기

### Figure 10: Learning Dynamics (2×2 Grid)
1. **Entropy:**
   - `entropy_dg5f.csv` ← **step**, **entropy**
   - `entropy_inspire.csv`

2. **Critic Loss:**
   - `critic_loss_dg5f.csv` ← **step**, **critic_loss**
   - `critic_loss_inspire.csv`

3. **KL Divergence:**
   - `kl_divergence_dg5f.csv` ← **step**, **kl**
   - `kl_divergence_inspire.csv`

4. **Episode Length:**
   - `episode_length_dg5f.csv` ← **step**, **episode_length**
   - `episode_length_inspire.csv`

### Figure 11: ADR Curriculum
- `adr_curriculum_dg5f.csv` ← **step**, **curriculum_difficulty**
- `adr_curriculum_inspire.csv`

## 🔧 TensorBoard에서 CSV 다운로드 방법

1. TensorBoard 실행:
   ```bash
   tensorboard --logdir=/home/dyros/IsaacLab/logs/rl_games/stochastic_rl
   ```

2. 브라우저에서 원하는 그래프 찾기 (예: `rewards/shaped`)

3. 그래프 왼쪽 하단의 **"Show data download links"** 아이콘 클릭

4. **CSV 다운로드** 클릭

5. 파일명 변경:
   - 원본: `run-2025-11-20_23-27-dg5f-tag-rewards_shaped.csv`
   - 변경: `shaped_step_reward_dg5f.csv`

6. 파일을 `/home/dyros/IsaacLab/stochastic rl/new/data/` 폴더에 저장

## 🎨 Python Script 사용법

### 방법 1: Shaped Reward 전용 스크립트
```bash
cd "/home/dyros/IsaacLab/stochastic rl/new/data"
python plot_shaped_reward.py
```

### 방법 2: 범용 스크립트 (대화형)
```bash
python plot_tensorboard_metric.py
# 메뉴에서 원하는 metric 선택 (1-7)
```

### 방법 3: 직접 Python 코드 수정
```python
# plot_tensorboard_metric.py 파일을 열어서
# configs 딕셔너리에 원하는 설정 추가
```

## 📊 출력 파일

각 script 실행 시 생성되는 파일:
- `{metric}_comparison.pdf` ← **LaTeX 논문에 사용** (벡터 그래픽, 확대해도 안 깨짐)
- `{metric}_comparison.png` ← 미리보기용 (래스터 그래픽)

## ⚙️ Smoothing 파라미터 조정

그래프가 너무 튀면 (noisy):
```python
smoothing = 0.97  # 더 부드럽게 (0.95 → 0.97)
```

그래프가 너무 뭉개지면:
```python
smoothing = 0.90  # 덜 부드럽게 (0.95 → 0.90)
```

권장값:
- Reward 계열: 0.95
- Entropy, KL: 0.93
- Loss 계열: 0.90
- Curriculum: 0.98 (원래 부드러우니까)

## 🎯 컬럼명 매칭

TensorBoard CSV의 컬럼명이 다를 경우, script에서 수정:

```python
# 예시: TensorBoard가 'Step' (대문자)로 저장한 경우
x_col = 'Step'  # 원래 'step'에서 변경

# 예시: TensorBoard가 'Value'로 저장한 경우
y_col = 'Value'  # 원래 'shaped_reward'에서 변경
```

또는 CSV 파일 첫 줄을 직접 수정:
```csv
Step,Value          ← 원본
step,shaped_reward  ← 수정
```

## 📝 Troubleshooting

### Error: "CSV file not found"
→ CSV 파일이 `/home/dyros/IsaacLab/stochastic rl/new/data/` 폴더에 있는지 확인

### Error: "missing required columns"
→ CSV 파일의 첫 줄 (헤더)에 `step`, `shaped_reward` 등이 있는지 확인

### 그래프가 이상하게 나옴
→ CSV 데이터에 NaN, Inf 값이 있는지 확인 (pandas로 체크)

### PDF 폰트가 안 나옴
→ Times New Roman 폰트가 시스템에 없을 수 있음. DejaVu Serif로 자동 fallback됨
