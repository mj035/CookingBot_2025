# OpenManipulator-X ROS2 Humble Setup Guide

이 가이드는 OpenManipulator-X를 ROS2 Humble 환경에서 사용하기 위한 설정 방법을 설명합니다.

## 1. ROS2 Control 관련 패키지 설치

```bash
sudo apt install \
  ros-humble-ros2-control \
  ros-humble-moveit* \
  ros-humble-gazebo-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-controller-manager \
  ros-humble-position-controllers \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-trajectory-controller \
  ros-humble-gripper-controllers \
  ros-humble-hardware-interface \
  ros-humble-xacro
```

## 2. OpenManipulator-X 패키지 소스 빌드

### 2.1 워크스페이스 생성 및 소스 다운로드
```bash
mkdir -p ~/colcon_ws/src
cd ~/colcon_ws/src/

git clone -b humble https://github.com/ROBOTIS-GIT/DynamixelSDK.git
git clone -b humble https://github.com/ROBOTIS-GIT/open_manipulator.git
git clone -b humble https://github.com/ROBOTIS-GIT/dynamixel_hardware_interface.git
git clone -b humble https://github.com/ROBOTIS-GIT/dynamixel_interfaces.git
```

### 2.2 패키지 빌드
```bash
cd ~/colcon_ws
colcon build --symlink-install
```

## 3. 환경 설정

```bash
echo 'source /usr/share/gazebo/setup.sh' >> ~/.bashrc
echo 'source ~/colcon_ws/install/local_setup.bash' >> ~/.bashrc
source ~/.bashrc
```

## 4. Dynamixel 모터 설정 (Wizard 2.0 사용)

### 4.1 모터 ID 설정
- Joint 1 ↔ Motor ID: 11
- Joint 2 ↔ Motor ID: 12  
- Joint 3 ↔ Motor ID: 13
- Joint 4 ↔ Motor ID: 14
- Gripper ↔ Motor ID: 15

### 4.2 통신 설정
- **Baud Rate**: 1Mbps (1,000,000 bps)
- **Protocol**: 2.0

### 4.3 Wizard 2.0 설정 절차
1. Dynamixel Wizard 2.0 실행
2. 포트 선택 (일반적으로 `/dev/ttyUSB0` 또는 `/dev/ttyACM0`)
3. Scan을 통해 연결된 모터 확인
4. 각 모터별로 ID 설정 (11~15)
5. Baud Rate를 1Mbps로 설정
6. 설정 저장

## 5. ROS2에서 OpenManipulator-X 실행

### 5.1 환경 설정 확인
```bash
source /opt/ros/humble/setup.bash
source ~/colcon_ws/install/local_setup.bash
```

### 5.2 하드웨어 실행
```bash
ros2 launch open_manipulator_x_bringup hardware.launch.py port_name:=/dev/ttyACM0
```

### 5.3 포트 확인
USB 포트 확인:
```bash
ls /dev/ttyACM*
ls /dev/ttyUSB*
```

## 6. 문제 해결

### 6.1 권한 문제
```bash
sudo usermod -a -G dialout $USER
sudo chmod 666 /dev/ttyACM0
```

### 6.2 포트 연결 확인
```bash
dmesg | grep tty
```

### 6.3 모터 연결 테스트
```bash
# MoveIt을 사용한 GUI 제어
ros2 launch open_manipulator_x_moveit_config demo.launch.py

# 키보드 텔레오퍼레이션
ros2 run open_manipulator_x_teleop open_manipulator_x_teleop_keyboard
```

## 7. 주의사항

- 모터 ID와 Baud Rate 설정이 정확해야 ROS2에서 인식 가능
- USB 케이블과 전원 연결 상태 확인
- 처음 연결 시 모터 캘리브레이션이 필요할 수 있음
- 안전을 위해 로봇 주변에 장애물이 없는지 확인

## 8. VR 텔레오퍼레이션 연동

모든 설정이 완료되면 다음 순서로 VR 제어 시스템 실행:

1. **Docker 컨테이너에서 VR 브릿지 실행**:
   ```bash
   docker exec -it quest2ros_fresh bash
   source /opt/ros/noetic/setup.bash
   source ~/catkin_ws/devel/setup.bash
   roscore &
   sleep 5
   rosrun quest2ros ros2quest.py
   ```

2. **호스트에서 OpenManipulator 실행**:
   ```bash
   source /opt/ros/humble/setup.bash
   source ~/colcon_ws/install/local_setup.bash
   ros2 launch open_manipulator_x_bringup hardware.launch.py port_name:=/dev/ttyACM0
   ```

3. **MuJoCo 시뮬레이션 실행** (선택사항):
   ```bash
   cd ~/CookingBot_2025/vr_teleoperation
   python3 mujoco_single_robot_v2.py
   ```