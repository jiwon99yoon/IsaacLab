# Conda CUDA Toolkit 제거 및 복구 가이드

## 문제 요약

```bash
conda install cuda-toolkit=12.8
```

이 명령이 다음을 설치했습니다:
- CUDA toolkit 12.8 (50개+ 패키지)
- **GCC 11.2.0** (구버전) → GCC_12.0.0 미지원
- 여러 의존성 라이브러리

결과:
- Isaac Sim 센서 라이브러리 로딩 실패
- `GCC_12.0.0' not found` 에러

## 해결 방법 비교

### Option 1: CUDA Toolkit 제거 ⭐ 추천

**장점:**
- 깔끔하게 문제 해결
- CuRobo는 시스템 CUDA (12.8) 사용 가능
- conda 환경 가벼워짐

**단점:**
- CuRobo 재설치 필요 (5-10분)

### Option 2: libgcc만 업데이트

**장점:**
- 빠름 (1-2분)
- CuRobo 재설치 불필요

**단점:**
- Conda CUDA와 시스템 CUDA 충돌 가능성
- 디스크 공간 낭비 (불필요한 CUDA toolkit)

## 방법 1: CUDA Toolkit 완전 제거 (추천)

### Step 1: 백업

```bash
# Conda 환경 백업
conda activate env_isaaclab
conda list --explicit > ~/env_isaaclab_backup.txt

# CuRobo 설치 경로 확인
pip show nvidia-curobo
```

### Step 2: CUDA Toolkit 제거

```bash
conda activate env_isaaclab

# 1. CUDA toolkit 메타패키지 제거
conda remove cuda-toolkit -y

# 2. 남은 CUDA 패키지들 확인
conda list | grep cuda

# 3. 필요시 개별 제거
conda remove cuda-libraries cuda-libraries-dev cuda-tools cuda-visual-tools -y

# 4. 구버전 GCC 제거
conda remove gcc_impl_linux-64 gcc_linux-64 gxx_impl_linux-64 gxx_linux-64 -y
```

### Step 3: 최신 libgcc 설치

```bash
# 최신 libgcc 설치
conda install -c conda-forge libgcc-ng=13 -y

# 확인
strings $CONDA_PREFIX/lib/libgcc_s.so.1 | grep "GCC_" | tail -5
# GCC_12.0.0이 나와야 함
```

### Step 4: 시스템 CUDA 사용 설정

```bash
# ~/.bashrc 수정
nano ~/.bashrc

# 아래 내용 추가:
export CUDA_HOME="/usr/local/cuda"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.0+PTX"

# 적용
source ~/.bashrc
```

### Step 5: CuRobo 재설치

```bash
conda activate env_isaaclab

# 기존 CuRobo 제거
pip uninstall nvidia-curobo -y

# 재설치 (시스템 CUDA 사용)
pip install -e "git+https://github.com/NVlabs/curobo.git@ebb71702f3f70e767f40fd8e050674af0288abe8#egg=nvidia-curobo" --no-build-isolation
```

### Step 6: 검증

```bash
# Isaac Sim 실행 테스트
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --device cuda:0 \
    --num_envs 1 \
    --generation_num_trials 1 \
    --input_file ./datasets/annotated_dataset_skillgen.hdf5 \
    --output_file ./datasets/test.hdf5 \
    --task Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0 \
    --use_skillgen

# 에러 확인
# "GCC_12.0.0 not found" 에러가 사라져야 함
```

## 방법 2: libgcc만 업데이트 (빠른 해결)

### 간단한 방법

```bash
conda activate env_isaaclab

# 최신 libgcc 설치 (기존 것 덮어쓰기)
conda install -c conda-forge libgcc-ng=13 --force-reinstall -y

# 또는 시스템 libgcc 사용
cd $CONDA_PREFIX/lib
mv libgcc_s.so.1 libgcc_s.so.1.backup
ln -s /lib/x86_64-linux-gnu/libgcc_s.so.1 libgcc_s.so.1
```

### 검증

```bash
# libgcc 버전 확인
strings $CONDA_PREFIX/lib/libgcc_s.so.1 | grep "GCC_" | tail -5

# Isaac Sim 테스트
./isaaclab.sh -p check_curobo_franka_files.py --headless 2>&1 | grep -i gcc
```

## 자동 스크립트 실행

```bash
# 자동 스크립트로 Option 1 실행
chmod +x remove_conda_cuda.sh
./remove_conda_cuda.sh
```

## 문제 발생 시 복구

### Conda 환경 복구

```bash
# 백업에서 복원
conda create --name env_isaaclab_new --file ~/env_isaaclab_backup.txt
```

### 처음부터 재설치

```bash
# 환경 삭제
conda deactivate
conda env remove -n env_isaaclab

# IsaacLab 재설치
cd ~/IsaacLab
./isaaclab.sh --install
```

## 권장 사항

**최선의 방법:**
1. ✅ **방법 1 (CUDA Toolkit 제거)** 실행
   - 깔끔하고 안정적
   - 향후 문제 최소화

**빠른 임시 해결:**
2. ⚡ **방법 2 (libgcc만 업데이트)**
   - 에러만 없애고 싶은 경우
   - 나중에 방법 1로 전환 권장

## 참고

**시스템 CUDA 확인:**
```bash
nvcc --version
# Cuda compilation tools, release 12.8, V12.8.93
```

**Conda CUDA vs 시스템 CUDA:**
- CuRobo는 둘 다 사용 가능
- 시스템 CUDA 사용 권장 (conda 환경 가볍게 유지)
- Conda CUDA는 Python 패키지 개발용

**에러가 사라져야 할 것들:**
- ✅ `GCC_12.0.0' not found`
- ✅ `libgeneric_mo_io.so: cannot open shared object file`
- ✅ `failed to load native plugin` (센서 관련)

**여전히 나올 수 있는 경고 (무시 가능):**
- ⚠️ Warp CUDA warnings (Warp 미사용 시)
