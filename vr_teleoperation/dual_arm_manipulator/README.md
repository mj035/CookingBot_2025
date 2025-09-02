# Dual Arm OpenManipulator-X ROS2 Package

듀얼 암 OpenManipulator-X를 ROS2 Humble에서 제어하기 위한 패키지입니다.

## 시스템 요구사항

- Ubuntu 22.04
- ROS2 Humble
- OpenCR 보드 1개
- OpenManipulator-X 2세트
- Dynamixel XM430-W350-T 모터 10개

## 모터 ID 설정

### 왼팔 (Left Arm)
- Joint 1: Motor ID 11
- Joint 2: Motor ID 12
- Joint 3: Motor ID 13
- Joint 4: Motor ID 14
- Gripper: Motor ID 15

### 오른팔 (Right Arm)
- Joint 1: Motor ID 21
- Joint 2: Motor ID 22
- Joint 3: Motor ID 23
- Joint 4: Motor ID 24
- Gripper: Motor ID 25

## 설치 방법

### 1. 필수 패키지 설치

```bash
# ROS2 Control 관련 패키지
sudo apt update
sudo apt install -y \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-controller-manager \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-trajectory-controller \
  ros-humble-position-controllers \
  ros-humble-gripper-controllers \
  ros-humble-hardware-interface \
  ros-humble-xacro \
  ros-humble-robot-state-publisher \
  ros-humble-rviz2

# Gazebo 시뮬레이션 (선택사항)
sudo apt install -y \
  ros-humble-gazebo-ros2-control \
  ros-humble-moveit

# Fake hardware (테스트용)
sudo apt install -y \
  ros-humble-fake-components
```

### 2. Dynamixel SDK 및 관련 패키지 빌드

```bash
# 워크스페이스 생성
mkdir -p ~/colcon_ws/src
cd ~/colcon_ws/src

# 필수 저장소 클론
git clone -b humble https://github.com/ROBOTIS-GIT/DynamixelSDK.git
git clone -b humble https://github.com/ROBOTIS-GIT/dynamixel_hardware_interface.git
git clone -b humble https://github.com/ROBOTIS-GIT/dynamixel_interfaces.git

# 이 패키지 클론 (또는 복사)
# git clone [your-repo-url] dual_arm_manipulator
# 또는
cp -r /path/to/dual_arm_manipulator ~/colcon_ws/src/

# 빌드
cd ~/colcon_ws
colcon build --symlink-install
source install/setup.bash
```

### 3. USB 포트 권한 설정

```bash
# 사용자를 dialout 그룹에 추가
sudo usermod -a -G dialout $USER

# 재로그인 또는 재부팅 필요
# 또는 임시로 권한 부여
sudo chmod 666 /dev/ttyACM0
```

## 실행 방법

### 1. 환경 설정

```bash
source /opt/ros/humble/setup.bash
source ~/colcon_ws/install/setup.bash
```

### 2. 하드웨어 실행

#### 실제 로봇 연결
```bash
# 기본 포트 (/dev/ttyACM0) 사용
ros2 launch dual_arm_manipulator dual_arm_hardware.launch.py

# 다른 포트 사용 시
ros2 launch dual_arm_manipulator dual_arm_hardware.launch.py port_name:=/dev/ttyUSB0

# RViz 함께 실행
ros2 launch dual_arm_manipulator dual_arm_hardware.launch.py start_rviz:=true
```

#### Fake Hardware (테스트용)
```bash
ros2 launch dual_arm_manipulator dual_arm_hardware.launch.py use_fake_hardware:=true start_rviz:=true
```

#### Gazebo 시뮬레이션
```bash
ros2 launch dual_arm_manipulator dual_arm_hardware.launch.py use_sim:=true
```

### 3. 컨트롤러 상태 확인

```bash
# 컨트롤러 목록 확인
ros2 control list_controllers

# 하드웨어 인터페이스 상태 확인
ros2 control list_hardware_interfaces
```

### 4. 토픽 확인

```bash
# 왼팔 제어 토픽
ros2 topic list | grep arm_controller

# 오른팔 제어 토픽
ros2 topic list | grep right_arm_controller

# Joint States
ros2 topic echo /joint_states
```

## 문제 해결

### 1. 포트를 찾을 수 없을 때

```bash
# 연결된 USB 장치 확인
ls /dev/ttyACM* /dev/ttyUSB*

# dmesg로 연결 로그 확인
dmesg | grep tty
```

### 2. 모터가 인식되지 않을 때

1. Dynamixel Wizard 2.0으로 모터 ID와 Baud Rate 확인
2. Baud Rate가 1Mbps (1000000)로 설정되어 있는지 확인
3. 모든 모터의 Protocol이 2.0으로 설정되어 있는지 확인

### 3. 컨트롤러가 로드되지 않을 때

```bash
# 컨트롤러 수동 로드
ros2 run controller_manager spawner joint_state_broadcaster
ros2 run controller_manager spawner arm_controller
ros2 run controller_manager spawner right_arm_controller
```

### 4. 권한 문제

```bash
# 영구적 해결
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6014", MODE="0666"' | sudo tee /etc/udev/rules.d/99-opencr.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## 패키지 구조

```
dual_arm_manipulator/
├── package.xml              # 패키지 메타데이터
├── CMakeLists.txt           # 빌드 설정
├── launch/
│   └── dual_arm_hardware.launch.py  # 메인 런치 파일
├── config/
│   └── dual_arm_controller_manager.yaml  # 컨트롤러 설정
├── urdf/
│   ├── dual_arm_simple.urdf.xacro  # 로봇 모델 정의
│   └── ros2_control/
│       └── dual_arm_system.ros2_control.xacro  # 하드웨어 인터페이스 정의
└── rviz/
    └── dual_arm.rviz        # RViz 설정 파일
```

## 라이센스

Apache-2.0

## 문의사항

이슈나 문의사항은 GitHub Issues를 통해 등록해주세요.