
![robot <-> mujoco](../assets/IMG_1887.gif)



### 1. teaching.py
- **기능**: 실물 로봇 Direct Teaching (토크 OFF)
- **포트**: 12345 (MuJoCo로 전송)
- **사용법**: 로봇을 손으로 움직이면 MuJoCo가 실시간 동기화

### 2. recorder.py
- **기능**: VR 포즈 + MuJoCo 조인트 동시 기록
- **포트**: 12346(VR), 12347(MuJoCo)
- **출력**: JSON 형식 매핑 데이터

### 3. vr_pose.py
- **기능**: Docker 내 VR 컨트롤러 데이터 수집
- **실행**: ROS1 환경 (Docker 컨테이너)
- **전송**: Host로 포즈 데이터 스트리밍

### 4. mujoco_data.py
- **기능**: 실물로봇과 동기화된 시뮬레이션 환경에서 joint확인 가능
- **용도**: 데이터 검증 및 시각화

## 실행 순서

```bash
# 1. MuJoCo 시뮬레이터 실행
python3 mujoco_data.py

# 2. 실물 로봇 동기화
python3 teaching.py

# 3. VR 수집기 실행 (Docker)
docker exec -it quest2ros bash
python3 vr_pose.py

# 4. 데이터 레코더 실행
python3 recorder.py

```

## 수집된 데이터
- `mujoco_vr_mapping_*.json`: VR-로봇 매핑 데이터

## 데이터 형식
```json
{
  "timestamp": "2025-01-12 10:00:00",
  "vr_pose": [x, y, z, qx, qy, qz, qw],
  "robot_joints": [j1, j2, j3, j4],
  "gripper": 0.0
}
```
