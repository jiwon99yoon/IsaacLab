#!/bin/bash
# CUDA toolkit 제거 및 시스템 CUDA 사용 스크립트

set -e

echo "==================================================================="
echo "Conda CUDA Toolkit 제거 및 시스템 CUDA 사용 설정"
echo "==================================================================="

# Conda 환경 활성화
source ~/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

echo -e "\n1. 현재 CUDA 패키지 목록:"
conda list | grep cuda | wc -l
echo "   개의 CUDA 관련 패키지 설치됨"

echo -e "\n2. Conda CUDA toolkit 제거 중..."
# 주의: 이 명령은 시간이 걸릴 수 있습니다
conda remove --force cuda-toolkit -y

echo -e "\n3. 추가 CUDA 패키지 제거 중..."
# 남은 CUDA 패키지들도 제거
conda remove --force $(conda list | grep '^cuda-' | awk '{print $1}') -y 2>/dev/null || true

echo -e "\n4. 구버전 GCC 제거 중..."
# GCC 11.2.0 제거
conda remove --force gcc_impl_linux-64 gcc_linux-64 gxx_impl_linux-64 gxx_linux-64 -y 2>/dev/null || true

echo -e "\n5. 최신 libgcc 설치 중..."
# 최신 libgcc 설치
conda install -c conda-forge libgcc-ng=13 -y

echo -e "\n6. 환경 변수 설정..."
# ~/.bashrc에 시스템 CUDA 경로 추가
cat >> ~/.bashrc << 'EOFBASH'

# System CUDA for CuRobo (added by remove_conda_cuda.sh)
export CUDA_HOME="/usr/local/cuda"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
EOFBASH

echo -e "\n==================================================================="
echo "완료!"
echo "==================================================================="
echo "
다음 단계:
1. 터미널 재시작 또는 source ~/.bashrc 실행
2. CuRobo 재설치 (시스템 CUDA 사용):
   source ~/.bashrc
   conda activate env_isaaclab
   pip install -e \"git+https://github.com/NVlabs/curobo.git@ebb71702f3f70e767f40fd8e050674af0288abe8#egg=nvidia-curobo\" --no-build-isolation

3. Isaac Sim 실행하여 에러 확인
   ./isaaclab.sh -p check_curobo_franka_files.py --headless
"
