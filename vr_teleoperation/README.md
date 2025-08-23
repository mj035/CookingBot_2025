# VR Teleoperation for Dual Arm Robots 🤖🤖

## 🎉 양팔 VR 텔레오퍼레이션 성공!

Meta Quest 2를 이용한 OpenManipulator-X 양팔 제어 시스템 구현 완료

## 📋 시스템 구성

### 아키텍처
```
Meta Quest 2 Controllers (Left & Right)
         ↓
    Docker (ROS1 + quest2ros)
         ↓
    Host (ROS2 Bridge)
         ↓
    dual_arm_bridge.py
         ↓ (Socket 12345)
    mujoco_mirror.py (시뮬레이션)
    또는
    mirror_dual_robot.py (실물 로봇)
```

## 🚀 실행 방법

### 1. MuJoCo 시뮬레이션

#### 자동 실행:
```bash
cd ~/CookingBot_2025/vr_teleoperation
./start_dual_arm.sh
```

#### 수동 실행:
```bash
# 터미널 1 - ROS2 브릿지
python3 dual_arm_bridge.py

# 터미널 2 - MuJoCo 시뮬레이터
python3 mujoco_mirror.py
```

### 2. 실물 로봇 (개발 예정)

Namespace 방식을 사용한 양팔 제어 구현 예정

## 📁 주요 파일

### 핵심 스크립트
- `dual_arm_bridge.py`: 양팔 VR 데이터 처리 및 전송
- `mujoco_mirror.py`: MuJoCo 양팔 시뮬레이터
- `start_dual_arm.sh`: 자동 실행 스크립트

### XML 파일
- `scene_dual.xml`: 양팔 로봇 씬 정의
- `omx.xml`: 왼팔 로봇 정의
- `omx_r.xml`: 오른팔 로봇 정의 (수정됨)

## 🔧 해결한 주요 문제들

### 1. 오른팔 Joint1 문제
- **문제**: 오른팔 Joint1(좌우 회전)이 작동하지 않음
- **원인**: `omx_r.xml`의 불필요한 `arm_base_r` body 계층
- **해결**: body 구조를 왼팔과 동일하게 수정

### 2. Y축 반전 문제
- **문제**: 오른팔이 좌우 반대로 움직임
- **원인**: 양팔이 같은 방향을 바라보는데 같은 Y축 사용
- **해결**: `mujoco_mirror.py`에서 오른팔 Joint1 값 반전

### 3. 단일 소켓 통신
- **문제**: 두 개의 브릿지 사용 시 지연 발생
- **해결**: 하나의 브릿지로 양팔 데이터 통합 전송

## 🎮 조작 방법

- **왼쪽 VR 컨트롤러** → 왼쪽 로봇팔
- **오른쪽 VR 컨트롤러** → 오른쪽 로봇팔
- **트리거** → 그리퍼 제어
- **A+B 버튼** → 재캘리브레이션

## 📊 기술 스택

- **VR**: Meta Quest 2 + quest2ros
- **시뮬레이션**: MuJoCo
- **로봇**: OpenManipulator-X (듀얼)
- **통신**: ROS2 + Socket
- **제어**: KD-Tree 기반 조인트 매핑

## 🔮 향후 계획

### 1. 실물 로봇 연결
- Namespace 방식으로 양팔 독립 제어
- OpenCR 하나로 ID 11-15 (왼팔), 21-25 (오른팔) 제어

### 2. Launch 파일 구성
```python
# dual_arm_hardware.launch.py
- 하드웨어 인터페이스 (하나)
- 왼팔 컨트롤러 (namespace: left_arm)
- 오른팔 컨트롤러 (namespace: right_arm)
```

### 3. mirror_dual_robot.py 개발
- mirror1.py 기반 양팔 버전
- 각 팔별 독립적인 토픽 발행

## 📝 참고사항

- Joint4 직관적 제어 구현 (손목 회전)
- 안전 범위 제한 적용
- 실시간 스무딩 처리

---
*2024.08.23 - 양팔 VR 텔레오퍼레이션 MuJoCo 시뮬레이션 성공*