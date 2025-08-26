# 🤖 Dual Arm VR Teleoperation System

양팔 OpenManipulator-X 로봇을 Meta Quest 2 VR 컨트롤러로 제어하는 시스템입니다.

## 📋 시스템 요구사항

### 하드웨어
- OpenManipulator-X 로봇 2대
- OpenCR 보드 1개
- Dynamixel 모터 (왼팔: ID 11-15, 오른팔: ID 21-25)
- Meta Quest 2 VR 헤드셋
- USB 케이블 (OpenCR 연결용)

### 소프트웨어
- Ubuntu 22.04
- ROS2 Humble
- Docker (ROS1 Noetic 이미지)
- Python 3.10+

## 🚀 빠른 실행 가이드

### 1. 의존성 설치

```bash
# ROS2 패키지 설치
sudo apt update
sudo apt install ros-humble-ros2-control \
                 ros-humble-ros2-controllers \
                 ros-humble-controller-manager \
                 ros-humble-xacro \
                 ros-humble-joint-trajectory-controller \
                 ros-humble-position-controllers

# Python 패키지 설치
pip install mujoco numpy scipy
```

### 2. 워크스페이스 빌드

```bash
# colcon 워크스페이스 빌드
cd ~/colcon_ws
colcon build --packages-select open_manipulator_x_description \
                              open_manipulator_x_bringup
source install/setup.bash
```

### 3. 실행 순서

#### Step 1: Docker에서 VR 브릿지 실행 (터미널 1)
```bash
# Docker 컨테이너 실행
docker run -it --network host --privileged \
           -v /dev:/dev \
           -e DISPLAY=$DISPLAY \
           -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
           ros:noetic-ros-base

# 컨테이너 내부에서
cd /workspace
python3 test3_dual.py
```

#### Step 2: 하드웨어 런치 (터미널 2)
```bash
cd ~/colcon_ws
source install/setup.bash

# OpenCR 연결 확인
ls -l /dev/ttyACM*

# 런치 파일 실행
ros2 launch open_manipulator_x_bringup dual_arm_hardware.launch.py \
            port_name:=/dev/ttyACM0
```

#### Step 3: MuJoCo 시뮬레이션 (터미널 3)
```bash
cd ~/teleop_ws/CookingBot/Mujoco/vr_teleoperation
python3 mujoco_mirror.py
```

#### Step 4: 실제 로봇 미러링 (터미널 4)
```bash
cd ~/teleop_ws/CookingBot/Mujoco/vr_teleoperation
python3 mirror_dual.py
```

## 📁 파일 구조

```
프로젝트 구조:
├── colcon_ws/src/open_manipulator/
│   ├── open_manipulator_x_bringup/
│   │   ├── launch/
│   │   │   └── dual_arm_hardware.launch.py
│   │   ├── config/
│   │   │   └── dual_arm_controller_manager.yaml
│   │   └── rviz/
│   │       └── open_manipulator_x.rviz
│   └── open_manipulator_x_description/
│       ├── urdf/
│       │   └── dual_arm_simple.urdf.xacro
│       └── ros2_control/
│           └── dual_arm_system.ros2_control.xacro
│
└── teleop_ws/CookingBot/Mujoco/vr_teleoperation/
    ├── mujoco_mirror.py      # MuJoCo 시뮬레이션
    ├── test3_dual.py         # VR 브릿지 (Docker/ROS1)
    ├── mirror_dual.py        # 로봇 제어 (Host/ROS2)
    ├── scene_dual.xml        # MuJoCo 씬
    ├── omx.xml              # 왼팔 모델
    └── omx_r.xml            # 오른팔 모델
```

## 🎮 사용 방법

### VR 캘리브레이션
1. VR 헤드셋을 착용하고 양쪽 컨트롤러를 들기
2. 'A' 버튼: 왼팔 캘리브레이션
3. 'X' 버튼: 오른팔 캘리브레이션
4. 트리거: 그리퍼 제어

### 안전 기능
- 조인트 리미트 자동 체크
- 과도한 움직임 제한
- Emergency Stop: 'B' 버튼

## 🔧 문제 해결

### OpenCR 연결 실패
```bash
# 포트 권한 설정
sudo chmod 666 /dev/ttyACM0

# 또는 사용자를 dialout 그룹에 추가
sudo usermod -a -G dialout $USER
# 로그아웃 후 재로그인 필요
```

### 모터 ID 확인/변경
```bash
# Dynamixel Wizard 사용
sudo apt install dynamixel-wizard
```

### Docker 네트워크 문제
```bash
# host 네트워크 모드 사용
docker run --network host ...
```

## 📊 모터 ID 매핑

| 로봇 | 조인트 | 모터 ID |
|------|--------|---------|
| 왼팔 | joint1 | 11 |
|      | joint2 | 12 |
|      | joint3 | 13 |
|      | joint4 | 14 |
|      | gripper | 15 |
| 오른팔 | joint1 | 21 |
|       | joint2 | 22 |
|       | joint3 | 23 |
|       | joint4 | 24 |
|       | gripper | 25 |

## 🌐 네트워크 포트

- VR Bridge → MuJoCo: localhost:12345
- MuJoCo → Robot Bridge: localhost:12346

## 📝 라이센스

MIT License

## 👥 기여자

- CookingBot Team

## 📞 문의

문제가 있으시면 Issue를 생성해주세요.