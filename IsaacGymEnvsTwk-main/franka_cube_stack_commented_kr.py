# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np  # NumPy 라이브러리 임포트 (수치 계산용)
import os  # 파일 및 디렉토리 경로 처리용 OS 모듈
import torch  # PyTorch 딥러닝 프레임워크

from isaacgym import gymtorch  # Isaac Gym의 PyTorch 래퍼
from isaacgym import gymapi  # Isaac Gym API

# 유틸리티 함수들 임포트: 쿼터니언 곱셈, 텐서 변환, 텐서 클램핑, 쿼터니언 적용
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp, quat_apply
# 벡터화된 환경의 기본 클래스 임포트
from isaacgymenvs.tasks.base.vec_task import VecTask


@torch.jit.script  # JIT 컴파일을 통한 성능 최적화 데코레이터
def axisangle2quat(vec, eps=1e-6):
    """
    축-각도(axis-angle) 표현을 쿼터니언으로 변환하는 함수
    Args:
        vec (tensor): (..., 3) 형태의 텐서, 마지막 차원은 (ax,ay,az) 축-각도 지수 좌표
        eps (float): 작은 값을 0으로 매핑하기 위한 안정성 값

    Returns:
        tensor: (..., 4) 형태의 텐서, 마지막 차원은 (x,y,z,w) vec4 쿼터니언
    """
    # type: (Tensor, float) -> Tensor
    # 입력 형태를 저장하고 재구성
    input_shape = vec.shape[:-1]  # 마지막 차원을 제외한 모든 차원의 shape 저장
    vec = vec.reshape(-1, 3)  # (N, 3) 형태로 평탄화

    # 회전 각도 계산 (벡터의 노름)
    angle = torch.norm(vec, dim=-1, keepdim=True)  # L2 노름 계산

    # 반환할 쿼터니언 배열 생성 및 초기화
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0  # w 성분을 1로 초기화 (단위 쿼터니언)

    # 각도가 0이 아닌 인덱스를 찾아 쿼터니언 형태로 변환
    idx = angle.reshape(-1) > eps  # eps보다 큰 각도만 선택
    # 로드리게스 공식을 사용한 쿼터니언 변환
    # q = [sin(θ/2) * axis/||axis||, cos(θ/2)]
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],  # xyz 성분
        torch.cos(angle[idx, :] / 2.0)  # w 성분
    ], dim=-1)

    # 원래 형태로 재구성하여 반환
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class FrankaCubeStack(VecTask):
    """Franka 로봇을 사용한 큐브 쌓기 태스크 클래스"""

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg  # 설정(configuration) 딕셔너리 저장

        # 에피소드 최대 길이 설정
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        # 액션 스케일 및 노이즈 파라미터 설정
        self.action_scale = self.cfg["env"]["actionScale"]  # 액션 스케일링 계수
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]  # 큐브 시작 위치 노이즈
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]  # 큐브 시작 회전 노이즈
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]  # Franka 위치 노이즈
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]  # Franka 회전 노이즈
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]  # Franka DOF(자유도) 노이즈
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]  # 집계 모드 (물리 최적화용)

        # 보상 함수에 전달할 설정 딕셔너리 생성
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],  # 거리 보상 스케일
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],  # 들어올리기 보상 스케일
            "r_align_scale": self.cfg["env"]["alignRewardScale"],  # 정렬 보상 스케일
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],  # 쌓기 보상 스케일
        }

        # 제어기 타입 설정 및 검증
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"
        # osc: Operational Space Control (작업 공간 제어)
        # joint_tor: Joint Torque Control (관절 토크 제어)

        # 관측 및 액션 차원 설정
        # 관측: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)
        # OSC 모드: 19차원, Joint Torque 모드: 26차원 (관절 각도 7개 추가)
        self.cfg["env"]["numObservations"] = 19 if self.control_type == "osc" else 26
        # 액션: OSC의 경우 delta EEF (6) + 그리퍼 (1), Joint Torque의 경우 관절 토크 (7) + 그리퍼 (1)
        self.cfg["env"]["numActions"] = 7 if self.control_type == "osc" else 8

        # 런타임에 채워질 값들 초기화
        self.states = {}  # 보상 계산에 사용될 관련 상태들의 딕셔너리
        self.handles = {}  # 이름을 관련 시뮬레이션 핸들에 매핑하는 딕셔너리
        self.num_dofs = None  # 환경당 전체 DOF(자유도) 수
        self.actions = None  # 배포될 현재 액션
        self._init_cubeA_state = None  # 현재 환경에서 cubeA의 초기 상태
        self._init_cubeB_state = None  # 현재 환경에서 cubeB의 초기 상태
        self._cubeA_state = None  # 현재 환경에서 cubeA의 현재 상태
        self._cubeB_state = None  # 현재 환경에서 cubeB의 현재 상태
        self._cubeA_id = None  # 주어진 환경에서 cubeA에 해당하는 Actor ID
        self._cubeB_id = None  # 주어진 환경에서 cubeB에 해당하는 Actor ID

        # 텐서 플레이스홀더 초기화
        self._root_state = None  # 루트 바디의 상태 (n_envs, 13)
        self._dof_state = None  # 모든 관절의 상태 (n_envs, n_dof)
        self._q = None  # 관절 위치 (n_envs, n_dof)
        self._qd = None  # 관절 속도 (n_envs, n_dof)
        self._rigid_body_state = None  # 모든 강체의 상태 (n_envs, n_bodies, 13)
        self._contact_forces = None  # 시뮬레이션의 접촉력
        self._eef_state = None  # 엔드 이펙터 상태 (그립 포인트에서)
        self._eef_lf_state = None  # 엔드 이펙터 상태 (왼쪽 손가락 끝)
        self._eef_rf_state = None  # 엔드 이펙터 상태 (오른쪽 손가락 끝)
        self._j_eef = None  # 엔드 이펙터의 자코비안
        self._mm = None  # 질량 행렬 (Mass matrix)
        self._arm_control = None  # 팔 제어를 위한 텐서 버퍼
        self._gripper_control = None  # 그리퍼 제어를 위한 텐서 버퍼
        self._pos_control = None  # 위치 액션
        self._effort_control = None  # 토크 액션
        self._franka_effort_limits = None  # Franka의 액추에이터 힘 제한
        self._global_indices = None  # 평탄화된 배열에서 모든 환경에 해당하는 고유 인덱스

        # 디버그 시각화 활성화 여부
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # 위쪽 축 설정 (중력 방향의 반대)
        self.up_axis = "z"  # z축이 위쪽
        self.up_axis_idx = 2  # z축의 인덱스는 2

        # 부모 클래스 초기화
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Franka 기본 DOF 위치 설정 (9개: 7개 관절 + 2개 그리퍼 핑거)
        # 이 값은 로봇의 초기 자세를 정의함
        self.franka_default_dof_pos = to_torch(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035], device=self.device
        )

        # OSC (Operational Space Control) 게인 설정
        self.kp = to_torch([150.] * 6, device=self.device)  # 비례 게인 (위치 오차에 대한 강성)
        self.kd = 2 * torch.sqrt(self.kp)  # 미분 게인 (속도 오차에 대한 감쇠, 임계 감쇠)
        self.kp_null = to_torch([10.] * 7, device=self.device)  # 영공간 비례 게인
        self.kd_null = 2 * torch.sqrt(self.kp_null)  # 영공간 미분 게인
        #self.cmd_limit = None  # 나중에 채워질 예정

        # 제어 제한 설정
        # OSC 모드: [x, y, z, rx, ry, rz]의 위치/회전 명령 제한
        # Joint Torque 모드: 관절 토크 제한
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._franka_effort_limits[:7].unsqueeze(0)

        # 모든 환경 리셋
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # 텐서 새로고침
        self._refresh()

    def create_sim(self):
        """시뮬레이션 생성 및 설정"""
        # 시뮬레이션 파라미터 설정
        self.sim_params.up_axis = gymapi.UP_AXIS_Z  # z축을 위쪽으로 설정
        self.sim_params.gravity.x = 0  # x방향 중력 0
        self.sim_params.gravity.y = 0  # y방향 중력 0
        self.sim_params.gravity.z = -9.81  # z방향 중력 -9.81 m/s^2 (지구 중력)

        # 부모 클래스의 create_sim 호출하여 시뮬레이터 생성
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        # 바닥 평면 생성
        self._create_ground_plane()
        # 환경들 생성 (num_envs개, envSpacing 간격으로, sqrt(num_envs) 행렬 배치)
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        """바닥 평면 생성"""
        plane_params = gymapi.PlaneParams()  # 평면 파라미터 객체 생성
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)  # 평면의 법선 벡터 (z축 방향)
        self.gym.add_ground(self.sim, plane_params)  # 시뮬레이션에 바닥 추가

    def _create_envs(self, num_envs, spacing, num_per_row):
        """환경들을 생성하는 함수"""
        # 환경의 경계 박스 정의
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)  # 하한
        upper = gymapi.Vec3(spacing, spacing, spacing)  # 상한

        # 에셋 파일 경로 설정
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        # 설정 파일에 에셋 경로가 지정된 경우 해당 경로 사용
        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # Franka 에셋 로드 옵션 설정
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True  # 비주얼 첨부물 뒤집기
        asset_options.fix_base_link = True  # 베이스 링크 고정 (로봇이 떠다니지 않게)
        asset_options.collapse_fixed_joints = False  # 고정 조인트 유지
        asset_options.disable_gravity = True  # 중력 비활성화 (로봇이 스스로 서있도록)
        asset_options.thickness = 0.001  # 충돌 두께
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT  # 기본 DOF 구동 모드: 힘/토크
        asset_options.use_mesh_materials = True  # 메시 재질 사용
        # Franka 에셋 로드
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        # Franka DOF의 강성(stiffness)과 감쇠(damping) 설정
        # 처음 7개는 관절 (0), 마지막 2개는 그리퍼 (높은 강성과 감쇠)
        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # 테이블 에셋 생성
        table_pos = [0.0, 0.0, 1.0]  # 테이블 위치
        table_thickness = 0.05  # 테이블 두께
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True  # 테이블 고정
        # 1.2m x 1.2m x 0.05m 박스 형태의 테이블 생성
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        # 테이블 받침대 에셋 생성
        table_stand_height = 0.1  # 받침대 높이
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]  # 받침대 위치
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True  # 받침대 고정
        # 0.2m x 0.2m x 0.1m 박스 형태의 받침대 생성
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        # 큐브 크기 설정
        self.cubeA_size = 0.050  # cubeA 크기: 5cm
        self.cubeB_size = 0.070  # cubeB 크기: 7cm

        # cubeA 에셋 생성
        cubeA_opts = gymapi.AssetOptions()
        # 5cm x 5cm x 5cm 박스 생성
        cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)  # cubeA 색상: 붉은색 계열

        # cubeB 에셋 생성
        cubeB_opts = gymapi.AssetOptions()
        # 7cm x 7cm x 7cm 박스 생성
        cubeB_asset = self.gym.create_box(self.sim, *([self.cubeB_size] * 3), cubeB_opts)
        cubeB_color = gymapi.Vec3(0.0, 0.4, 0.1)  # cubeB 색상: 녹색 계열

        # Franka의 강체 개수와 DOF 개수 가져오기
        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        # 콘솔에 정보 출력
        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # Franka DOF 속성 설정
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []  # DOF 하한값 리스트
        self.franka_dof_upper_limits = []  # DOF 상한값 리스트
        self._franka_effort_limits = []  # DOF 힘/토크 제한 리스트

        for i in range(self.num_franka_dofs):
            # 구동 모드 설정: 그리퍼(인덱스 > 6)는 위치 제어, 관절은 토크 제어
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT

            # 물리 엔진에 따른 강성과 감쇠 설정
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:  # Flex 엔진 사용 시
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            # 각 DOF의 제한값 저장
            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        # 리스트를 텐서로 변환
        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)

        # DOF 속도 스케일 설정 (대부분 1.0)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1  # 그리퍼는 느리게 (10%)

        # 그리퍼의 힘 제한 설정
        franka_dof_props['effort'][7] = 200  # 왼쪽 핑거
        franka_dof_props['effort'][8] = 200  # 오른쪽 핑거

        # Franka 시작 포즈 정의
        franka_start_pose = gymapi.Transform()
        # 위치: 테이블 위 받침대 위
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # 회전 없음 (단위 쿼터니언)

        # 테이블 시작 포즈 정의
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        # 테이블 표면 위치 계산 (보상 계산에 사용)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # 테이블 받침대 시작 포즈 정의
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # 큐브 시작 포즈 정의 (reset()에서 재설정되므로 임시 위치)
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)  # 임시 위치
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        cubeB_start_pose = gymapi.Transform()
        cubeB_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)  # 임시 위치
        cubeB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # 집계(aggregate) 크기 계산 (물리 최적화용)
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 4  # Franka + 테이블 + 받침대 + cubeA + cubeB
        max_agg_shapes = num_franka_shapes + 4  # 동일

        # 환경 리스트 초기화
        self.frankas = []  # Franka 액터들
        self.envs = []  # 환경 포인터들

        # 환경 생성 루프
        for i in range(self.num_envs):
            # 환경 인스턴스 생성
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # 액터들을 생성하고 집계 그룹 적절히 정의
            # 주의: franka는 항상 시뮬레이션에서 먼저 로드되어야 함!
            if self.aggregate_mode >= 3:
                # 모든 액터를 하나의 집계 그룹으로
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Franka 생성
            # 위치 노이즈가 있으면 시작 포즈를 랜덤화
            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            # 회전 노이즈가 있으면 시작 회전을 랜덤화
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)
            # Franka 액터 생성 (이름: "franka", 충돌 그룹: 0)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            # DOF 속성 설정
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                # Franka만 별도 집계, 나머지는 함께
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # 테이블 생성 (충돌 그룹: 1)
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            # 받침대 생성 (충돌 그룹: 1)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 1, 0)

            if self.aggregate_mode == 1:
                # 테이블과 받침대만 별도 집계, 큐브는 따로
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # 큐브 생성
            # cubeA (충돌 그룹: 2)
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)
            # cubeB (충돌 그룹: 4)
            self._cubeB_id = self.gym.create_actor(env_ptr, cubeB_asset, cubeB_start_pose, "cubeB", i, 4, 0)

            # 큐브 색상 설정
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
            self.gym.set_rigid_body_color(env_ptr, self._cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color)

            if self.aggregate_mode > 0:
                # 집계 종료
                self.gym.end_aggregate(env_ptr)

            # 생성된 환경 포인터 저장
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # 초기 상태 버퍼 설정 (13: pos(3) + quat(4) + lin_vel(3) + ang_vel(3))
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cubeB_state = torch.zeros(self.num_envs, 13, device=self.device)

        # 데이터 초기화
        self.init_data()

    def init_data(self):
        """시뮬레이션 핸들 및 텐서 버퍼 초기화"""
        # 시뮬레이션 핸들 설정
        env_ptr = self.envs[0]  # 첫 번째 환경
        franka_handle = 0  # Franka 핸들

        # 강체 핸들 딕셔너리 설정
        self.handles = {
            # Franka 관련 핸들
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_grip_site"),
            # 큐브 핸들
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "box"),
            "cubeB_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeB_id, "box"),
        }

        # 전체 DOF 수 계산
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # 텐서 버퍼 설정
        # 액터 루트 상태 텐서 획득 및 래핑
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        # DOF 상태 텐서 획득 및 래핑
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # 강체 상태 텐서 획득 및 래핑
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # 텐서를 적절한 형태로 변환
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)

        # DOF 상태에서 위치와 속도 추출
        self._q = self._dof_state[..., 0]  # 관절 위치
        self._qd = self._dof_state[..., 1]  # 관절 속도

        # 엔드 이펙터 상태 추출
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]

        # 자코비안 텐서 획득
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        # 손 관절 인덱스 찾기
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['panda_hand_joint']
        # 엔드 이펙터 자코비안 (처음 7개 DOF만 사용)
        self._j_eef = jacobian[:, hand_joint_index, :, :7]

        # 질량 행렬 텐서 획득
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        # 팔의 질량 행렬 (7x7)
        self._mm = mm[:, :7, :7]

        # 큐브 상태 참조 설정
        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self._cubeB_state = self._root_state[:, self._cubeB_id, :]

        # 상태 딕셔너리 초기화 (큐브 크기 정보)
        self.states.update({
            "cubeA_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeA_size,
            "cubeB_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeB_size,
        })

        # 액션 버퍼 초기화
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # 제어 버퍼 초기화
        self._arm_control = self._effort_control[:, :7]  # 팔 제어 (토크)
        self._gripper_control = self._pos_control[:, 7:9]  # 그리퍼 제어 (위치)

        # 글로벌 인덱스 초기화 (각 환경의 5개 액터: franka, table, stand, cubeA, cubeB)
        self._global_indices = torch.arange(self.num_envs * 5, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        """상태 딕셔너리를 현재 시뮬레이션 상태로 업데이트"""
        self.states.update({
            # Franka 관련 상태
            "q": self._q[:, :],  # 모든 관절 위치
            "q_gripper": self._q[:, -2:],  # 그리퍼 위치 (마지막 2개 DOF)
            "eef_pos": self._eef_state[:, :3],  # 엔드 이펙터 위치
            "eef_quat": self._eef_state[:, 3:7],  # 엔드 이펙터 방향 (쿼터니언)
            "eef_vel": self._eef_state[:, 7:],  # 엔드 이펙터 속도 (선속도 + 각속도)
            "eef_lf_pos": self._eef_lf_state[:, :3],  # 왼쪽 손가락 위치
            "eef_lf_quat": self._eef_lf_state[:, 3:7],  # 왼쪽 손가락 방향
            "eef_rf_pos": self._eef_rf_state[:, :3],  # 오른쪽 손가락 위치
            "eef_rf_quat": self._eef_rf_state[:, 3:7],  # 오른쪽 손가락 방향

            # 큐브 관련 상태
            "cubeA_quat": self._cubeA_state[:, 3:7],  # cubeA 방향
            "cubeA_pos": self._cubeA_state[:, :3],  # cubeA 위치
            "cubeA_pos_relative": self._cubeA_state[:, :3] - self._eef_state[:, :3],  # cubeA의 상대 위치 (EEF 기준)
            "cubeB_quat": self._cubeB_state[:, 3:7],  # cubeB 방향
            "cubeB_pos": self._cubeB_state[:, :3],  # cubeB 위치
            "cubeA_to_cubeB_pos": self._cubeB_state[:, :3] - self._cubeA_state[:, :3],  # cubeA에서 cubeB로의 벡터
        })

    def _refresh(self):
        """시뮬레이션 텐서들을 새로고침"""
        self.gym.refresh_actor_root_state_tensor(self.sim)  # 액터 루트 상태 새로고침
        self.gym.refresh_dof_state_tensor(self.sim)  # DOF 상태 새로고침
        self.gym.refresh_rigid_body_state_tensor(self.sim)  # 강체 상태 새로고침
        self.gym.refresh_jacobian_tensors(self.sim)  # 자코비안 새로고침
        self.gym.refresh_mass_matrix_tensors(self.sim)  # 질량 행렬 새로고침

        # 상태 업데이트
        self._update_states()

    def compute_reward(self, actions):
        """보상 계산"""
        # JIT 컴파일된 보상 함수 호출
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states,
            self.reward_settings, self.max_episode_length
        )

    def compute_observations(self):
        """관측값 계산"""
        self._refresh()  # 상태 새로고침

        # 관측값에 포함될 상태 리스트
        obs = ["cubeA_quat", "cubeA_pos", "cubeA_to_cubeB_pos", "eef_pos", "eef_quat"]
        # 제어 타입에 따라 그리퍼 위치 또는 전체 관절 위치 추가
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]

        # 관측값 버퍼에 상태들을 연결하여 저장
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        # 각 관측값의 최대값 계산 (디버깅/모니터링용)
        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        return self.obs_buf

    def reset_idx(self, env_ids):
        """특정 환경들을 리셋하는 함수"""
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # 큐브 리셋 (cubeB 먼저, 그 다음 cubeA 샘플링 - 충돌 방지를 위해)
        self._reset_init_cube_state(cube='B', env_ids=env_ids, check_valid=False)
        self._reset_init_cube_state(cube='A', env_ids=env_ids, check_valid=True)

        # 새로운 초기 상태를 시뮬레이션 상태에 쓰기
        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
        self._cubeB_state[env_ids] = self._init_cubeB_state[env_ids]

        # 로봇 리셋
        # 랜덤 노이즈 생성
        reset_noise = torch.rand((len(env_ids), 9), device=self.device)
        # 기본 위치에 노이즈를 추가하고 제한값으로 클램핑
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)

        # 그리퍼 초기 위치 덮어쓰기 (노이즈 없음 - 항상 위치 제어됨)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]

        # 내부 관측값 리셋
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # 위치 제어를 현재 위치로, 속도/토크 제어를 0으로 설정
        # 주의: Task가 SimActions API를 사용하여 실제로 제어를 전파함
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # 업데이트 배포
        # Franka만 업데이트 (첫 번째 액터)
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        # 위치 목표 설정
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        # 토크 설정
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        # DOF 상태 설정
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # 큐브 상태 업데이트 (마지막 2개 액터: cubeA, cubeB)
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -2:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        # 진행 버퍼와 리셋 버퍼 초기화
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_init_cube_state(self, cube, env_ids, check_valid=True):
        """
        큐브의 위치를 샘플링하는 간단한 메서드

        startPositionNoise와 startRotationNoise를 기반으로 @cube의 위치를 샘플링하고,
        자동으로 내부 포즈를 리셋함. 적절한 self._init_cubeX_state를 채움

        @check_valid가 True이면, 샘플링된 위치가 다른 큐브와 접촉하지 않는지 확인함

        Args:
            cube(str): 샘플링할 큐브. 'A' 또는 'B'
            env_ids (tensor or None): 리셋할 특정 환경들
            check_valid (bool): 샘플링된 위치가 다른 큐브와 충돌하지 않는지 확인할지 여부
        """
        # env_ids가 None이면, 모든 환경을 리셋
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # 샘플링된 값을 저장할 버퍼 초기화
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)

        # 선택된 큐브에 따라 올바른 참조 가져오기
        if cube.lower() == 'a':
            this_cube_state_all = self._init_cubeA_state
            other_cube_state = self._init_cubeB_state[env_ids, :]
            cube_heights = self.states["cubeA_size"]
        elif cube.lower() == 'b':
            this_cube_state_all = self._init_cubeB_state
            other_cube_state = self._init_cubeA_state[env_ids, :]
            cube_heights = self.states["cubeA_size"]
        else:
            raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {cube}")

        # 충돌 없는 샘플링을 보장하기 위한 최소 큐브 거리는 각 큐브의 유효 반지름의 합
        min_dists = (self.states["cubeA_size"] + self.states["cubeB_size"])[env_ids] * np.sqrt(2) / 2.0

        # 큐브들이 너무 가까이 있지 않도록 최소 거리를 2배로 스케일링
        min_dists = min_dists * 2.0

        # 샘플링은 테이블 중앙을 "중심"으로 함
        centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)

        # z 값 설정 (고정 높이)
        sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights.squeeze(-1)[env_ids] / 2

        # 회전 초기화 (회전 없음, 쿼터니언의 w = 1)
        sampled_cube_state[:, 6] = 1.0

        # 유효한 샘플링을 확인하는 경우, 충돌이 없는지 확인하고 필요시 재샘플링
        # 큐브의 반지름을 기반으로 충돌 여부를 확인하는 간단한 휴리스틱 사용
        if check_valid:
            success = False
            # 아직 활발히 샘플링 중인 환경에 해당하는 인덱스
            active_idx = torch.arange(num_resets, device=self.device)
            num_active_idx = len(active_idx)
            # 최대 100번 시도
            for i in range(100):
                # x, y 값 샘플링
                sampled_cube_state[active_idx, :2] = centered_cube_xy_state + \
                                                     2.0 * self.start_position_noise * (
                                                             torch.rand_like(sampled_cube_state[active_idx, :2]) - 0.5)
                # 샘플링된 값이 유효한지 확인
                cube_dist = torch.linalg.norm(sampled_cube_state[:, :2] - other_cube_state[:, :2], dim=-1)
                # 최소 거리보다 가까운 큐브들의 인덱스
                active_idx = torch.nonzero(cube_dist < min_dists, as_tuple=True)[0]
                num_active_idx = len(active_idx)
                # active_idx가 비어있으면 모든 샘플링이 유효함
                if num_active_idx == 0:
                    success = True
                    break
            # 샘플링 성공 확인
            assert success, "Sampling cube locations was unsuccessful! ):"
        else:
            # 직접 샘플링 (유효성 검사 없음)
            sampled_cube_state[:, :2] = centered_cube_xy_state.unsqueeze(0) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 2, device=self.device) - 0.5)

        # 회전 값 샘플링
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            # z축 주변 회전만 (테이블 위에서)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            # 축-각도를 쿼터니언으로 변환하고 기존 쿼터니언과 곱함
            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])

        # 마지막으로, 샘플링된 값을 새로운 초기 상태로 설정
        this_cube_state_all[env_ids, :] = sampled_cube_state

    def _compute_osc_torques(self, dpose):
        """
        Operational Space Control (작업 공간 제어) 토크 계산

        참고 논문: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        유용한 자료: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        """
        # 현재 관절 위치와 속도 (팔만, 7개 DOF)
        q, qd = self._q[:, :7], self._qd[:, :7]

        # 질량 행렬의 역행렬
        mm_inv = torch.inverse(self._mm)

        # 엔드 이펙터 질량 행렬의 역행렬 계산
        # m_eef^-1 = J * M^-1 * J^T
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        # 엔드 이펙터 질량 행렬
        m_eef = torch.inverse(m_eef_inv)

        # 데카르트 액션 `dpose`를 관절 토크 `u`로 변환
        # u = J^T * m_eef * (kp * dpose - kd * vel)
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # 영공간 제어 토크 `u_null`은 관절 구성의 큰 변화를 방지함
        # OSC의 영공간에 추가되어 엔드 이펙터 방향이 일정하게 유지됨
        # 참고: roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        # 영공간 토크: 기본 자세로 되돌아가려는 복원력
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 7:] *= 0  # 7번째 이후는 0 (7개 DOF만 사용)
        u_null = self._mm @ u_null.unsqueeze(-1)
        # 영공간 투영 행렬을 사용하여 주 제어에 영공간 제어 추가
        # u_total = u + (I - J^T * J_inv) * u_null
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # 값을 유효한 토크 범위 내로 클램핑
        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))

        return u

    def pre_physics_step(self, actions):
        """물리 스텝 전에 호출되어 액션을 처리"""
        self.actions = actions.clone().to(self.device)

        # 팔과 그리퍼 명령 분리
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # 팔 제어 (먼저 값 스케일링)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        # OSC 모드인 경우 토크로 변환
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        # 팔 제어 버퍼에 설정
        self._arm_control[:, :] = u_arm

        # 그리퍼 제어
        u_fingers = torch.zeros_like(self._gripper_control)
        # 그리퍼 액션이 양수면 열기(상한값), 음수면 닫기(하한값)
        u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
                                      self.franka_dof_lower_limits[-2].item())
        u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
                                      self.franka_dof_lower_limits[-1].item())
        # 그리퍼 제어를 적절한 텐서 버퍼에 쓰기
        self._gripper_control[:, :] = u_fingers

        # 액션 배포 (시뮬레이션에 적용)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        """물리 스텝 후에 호출됨"""
        # 진행 버퍼 증가
        self.progress_buf += 1

        # 리셋이 필요한 환경 찾기
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # 관측값 계산
        self.compute_observations()
        # 보상 계산
        self.compute_reward(self.actions)

        # 디버그 시각화
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # 시각화할 관련 상태 가져오기
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            eef_lf_pos = self.states["eef_lf_pos"]
            eef_lf_rot = self.states["eef_lf_quat"]
            eef_rf_pos = self.states["eef_rf_pos"]
            eef_rf_rot = self.states["eef_rf_quat"]
            cubeA_pos = self.states["cubeA_pos"]
            cubeA_rot = self.states["cubeA_quat"]
            cubeB_pos = self.states["cubeB_pos"]
            cubeB_rot = self.states["cubeB_quat"]

            pos_list = [eef_pos, eef_lf_pos, eef_rf_pos]
            rot_list = [eef_rot, eef_lf_rot, eef_rf_rot]

            # 시각화 플롯 (각 환경에 대해)
            for i in range(self.num_envs):
                for pos, rot in zip(pos_list, rot_list):
                    # x축 방향 (빨간색)
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    # y축 방향 (녹색)
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    # z축 방향 (파란색)
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    # x축 선 그리기 (빨간색)
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    # y축 선 그리기 (녹색)
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    # z축 선 그리기 (파란색)
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
):
    """
    Franka 큐브 쌓기 태스크의 보상 계산 함수 (JIT 컴파일됨)

    Args:
        reset_buf: 리셋 버퍼
        progress_buf: 진행 버퍼
        actions: 액션
        states: 상태 딕셔너리
        reward_settings: 보상 설정 딕셔너리
        max_episode_length: 최대 에피소드 길이

    Returns:
        rewards: 보상 텐서
        reset_buf: 업데이트된 리셋 버퍼
    """
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    # 환경별 물리 파라미터 계산
    target_height = states["cubeB_size"] + states["cubeA_size"] / 2.0  # 목표 높이 (cubeB 위에 cubeA 중심)
    cubeA_size = states["cubeA_size"]
    cubeB_size = states["cubeB_size"]

    # 손에서 cubeA까지의 거리
    d = torch.norm(states["cubeA_pos_relative"], dim=-1)  # 그립 사이트에서의 거리
    d_lf = torch.norm(states["cubeA_pos"] - states["eef_lf_pos"], dim=-1)  # 왼쪽 손가락에서의 거리
    d_rf = torch.norm(states["cubeA_pos"] - states["eef_rf_pos"], dim=-1)  # 오른쪽 손가락에서의 거리
    # 평균 거리를 tanh로 스케일링 (0~1 범위로 정규화)
    dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)

    # cubeA를 들어올리는 보상
    cubeA_height = states["cubeA_pos"][:, 2] - reward_settings["table_height"]  # 테이블 위 높이
    cubeA_lifted = (cubeA_height - cubeA_size) > 0.04  # 4cm 이상 들어올렸는지 (불리언)
    lift_reward = cubeA_lifted

    # cubeA가 cubeB에 얼마나 잘 정렬되어 있는지 (cubeA가 들어올려진 경우에만 제공)
    offset = torch.zeros_like(states["cubeA_to_cubeB_pos"])
    offset[:, 2] = (cubeA_size + cubeB_size) / 2  # z 오프셋 (cubeB 위에 cubeA가 있어야 함)
    # 이상적 위치와의 거리
    d_ab = torch.norm(states["cubeA_to_cubeB_pos"] + offset, dim=-1)
    # 정렬 보상 (들어올린 경우에만)
    align_reward = (1 - torch.tanh(10.0 * d_ab)) * cubeA_lifted

    # 거리 보상은 dist와 align 보상의 최대값
    dist_reward = torch.max(dist_reward, align_reward)

    # 성공적으로 쌓기 완료에 대한 최종 보상
    # (cubeA가 목표 높이와 위치에 가깝고, 그리퍼가 잡고 있지 않은 경우에만)
    cubeA_align_cubeB = (torch.norm(states["cubeA_to_cubeB_pos"][:, :2], dim=-1) < 0.02)  # xy 평면에서 2cm 이내
    cubeA_on_cubeB = torch.abs(cubeA_height - target_height) < 0.02  # 목표 높이의 2cm 이내
    gripper_away_from_cubeA = (d > 0.04)  # 그리퍼가 4cm 이상 떨어짐
    # 모든 조건을 만족해야 쌓기 성공
    stack_reward = cubeA_align_cubeB & cubeA_on_cubeB & gripper_away_from_cubeA

    # 보상 구성
    # 쌓기 성공 시: 쌓기 보상
    # 그렇지 않으면: 거리 + 들어올리기 + 정렬 보상의 가중 합
    rewards = torch.where(
        stack_reward,
        reward_settings["r_stack_scale"] * stack_reward,  # 쌓기 보상
        reward_settings["r_dist_scale"] * dist_reward +
        reward_settings["r_lift_scale"] * lift_reward +
        reward_settings["r_align_scale"] * align_reward,  # 단계별 보상
    )

    # 리셋 계산
    # 최대 에피소드 길이에 도달했거나 쌓기에 성공한 경우 리셋
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (stack_reward > 0),
                           torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf
