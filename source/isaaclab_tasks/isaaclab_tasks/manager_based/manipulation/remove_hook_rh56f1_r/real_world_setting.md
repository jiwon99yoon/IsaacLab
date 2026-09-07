# Remove Hook Task 실제 환경 사양 

# 사용하는 로봇
1. HDR35_20 + ATI + RH56F1 (R) + Tashan sensor 
2. HDR35_20 + ATI + DG5F (L) + aidin sensor

# vision : 로봇과 샤시모듈 외부에 zivid camera설치

### 설비사양
# H/W               |       사양	                                                        |  수량	|   상세
소형 산업로봇	        |   현대로보틱스 HDR35-20	                |   2	|
로봇용 토크 센서	        |   ATI NET Axia80-M50	                |   2	|
그리퍼	            | 1) Tesollo DG-5F (왼손)	            |   1	|   자유도 20
	                | 2) Inspire robotics RH56F1 (오른손)	|   1	|   자유도 6
그리퍼용 토크/촉각 센서	| 1) Aidin 초소형 토크 센서	                                |   1	|   1번 그리퍼 팁 부착
	                |  2) Tashan 정전용량식 촉각 센서	                |   1	|   2번 그리퍼 팁 및 손바닥 부착
데이터 습득용 장비      |
(데이터 글러브)	    |   Manus Quantum Metagloves	        |    1  | 	-
고성능 물체 인식용 카메라	 |   Mechmind Pro M 또는Zivid2 M70	    |   2  	|    Isaac sim 연동 가능한 사양으로 선정 예정
저성능 동작학습용 카메라	|   Intel realsensor 455	            |   2	| -


# 실제 실험 세팅
샤시모듈은 고정되어있음 (strut와 spring 및 샤시모듈 전체는 고정되어있고, wire는 걸려있는 상황 - 로봇이 contact하거나 grasp하지 않는 이상 wire는 고정된 상태)
wire는 deformable object이나 시뮬레이션 상에선 일단 rigid body로 설정되어있음
rh56f1_r이 달린 모델은 left_ring을 타겟하도록, dg5f_l이 달린 모델은 right_ring을 타겟하도록


