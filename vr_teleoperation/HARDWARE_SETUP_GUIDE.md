# 🚀 듀얼 암 하드웨어 실행 가이드 (초보자용)

## 📋 사전 준비사항
- Ubuntu 22.04
- ROS2 Humble 설치 완료
- OpenCR 보드 + USB 케이블
- OpenManipulator-X 로봇 2대

---

## 1️⃣ GitHub에서 코드 다운로드
```bash
# 홈 디렉토리로 이동
cd ~

# Git 설치 (없다면)
sudo apt update
sudo apt install git -y

# 코드 다운로드
git clone https://github.com/mj035/CookingBot_2025.git
```

## 2️⃣ ROS2 워크스페이스 생성
```bash
# colcon 워크스페이스 생성
mkdir -p ~/colcon_ws/src
cd ~/colcon_ws/src

# 필요한 파일들 복사
cp -r ~/CookingBot_2025/Mujoco/vr_teleoperation/dual_arm_manipulator ./
```

## 3️⃣ 의존성 패키지 설치
```bash
# ROS2 필수 패키지 설치
sudo apt update
sudo apt install -y \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-controller-manager \
    ros-humble-xacro \
    ros-humble-joint-trajectory-controller \
    ros-humble-position-controllers \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-broadcaster

# Dynamixel 하드웨어 인터페이스 설치
sudo apt install -y ros-humble-dynamixel-hardware-interface
```

## 4️⃣ 패키지 구조 생성
```bash
cd ~/colcon_ws/src

# open_manipulator_x_description 패키지 생성
mkdir -p open_manipulator_x_description/urdf
mkdir -p open_manipulator_x_description/ros2_control

# open_manipulator_x_bringup 패키지 생성  
mkdir -p open_manipulator_x_bringup/launch
mkdir -p open_manipulator_x_bringup/config
mkdir -p open_manipulator_x_bringup/rviz

# 파일 복사
cp ~/CookingBot_2025/Mujoco/vr_teleoperation/dual_arm_simple.urdf.xacro \
   open_manipulator_x_description/urdf/

cp ~/CookingBot_2025/Mujoco/vr_teleoperation/dual_arm_system.ros2_control.xacro \
   open_manipulator_x_description/ros2_control/

cp ~/CookingBot_2025/Mujoco/vr_teleoperation/dual_arm_hardware.launch.py \
   open_manipulator_x_bringup/launch/

cp ~/CookingBot_2025/Mujoco/vr_teleoperation/dual_arm_controller_manager.yaml \
   open_manipulator_x_bringup/config/
```

## 5️⃣ package.xml 생성

### open_manipulator_x_description/package.xml
```bash
cat > ~/colcon_ws/src/open_manipulator_x_description/package.xml << 'EOF'
<?xml version="1.0"?>
<package format="3">
  <name>open_manipulator_x_description</name>
  <version>1.0.0</version>
  <description>Dual arm robot description</description>
  <maintainer email="you@example.com">Your Name</maintainer>
  <license>MIT</license>
  
  <buildtool_depend>ament_cmake</buildtool_depend>
  <exec_depend>xacro</exec_depend>
  <exec_depend>robot_state_publisher</exec_depend>
  
  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
EOF
```

### open_manipulator_x_bringup/package.xml
```bash
cat > ~/colcon_ws/src/open_manipulator_x_bringup/package.xml << 'EOF'
<?xml version="1.0"?>
<package format="3">
  <name>open_manipulator_x_bringup</name>
  <version>1.0.0</version>
  <description>Dual arm robot bringup</description>
  <maintainer email="you@example.com">Your Name</maintainer>
  <license>MIT</license>
  
  <buildtool_depend>ament_cmake</buildtool_depend>
  <exec_depend>controller_manager</exec_depend>
  <exec_depend>ros2_control</exec_depend>
  <exec_depend>ros2_controllers</exec_depend>
  
  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
EOF
```

## 6️⃣ CMakeLists.txt 생성

### open_manipulator_x_description/CMakeLists.txt
```bash
cat > ~/colcon_ws/src/open_manipulator_x_description/CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.5)
project(open_manipulator_x_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf ros2_control
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
EOF
```

### open_manipulator_x_bringup/CMakeLists.txt
```bash
cat > ~/colcon_ws/src/open_manipulator_x_bringup/CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.5)
project(open_manipulator_x_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  launch/dual_arm_hardware.launch.py
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
EOF
```

## 7️⃣ 빌드
```bash
cd ~/colcon_ws
colcon build
source install/setup.bash
```

## 8️⃣ OpenCR 권한 설정
```bash
# USB 권한 설정
sudo chmod 666 /dev/ttyACM0

# 또는 영구적으로 권한 부여
sudo usermod -a -G dialout $USER
# (로그아웃 후 다시 로그인 필요)
```

## 9️⃣ 하드웨어 실행
```bash
# 터미널 1: 하드웨어 런치
cd ~/colcon_ws
source install/setup.bash

# OpenCR 연결 확인
ls -l /dev/ttyACM*

# 런치 파일 실행
ros2 launch open_manipulator_x_bringup dual_arm_hardware.launch.py port_name:=/dev/ttyACM0
```

## 🔟 동작 확인
```bash
# 터미널 2: 토픽 확인
source /opt/ros/humble/setup.bash
ros2 topic list

# 조인트 상태 확인
ros2 topic echo /joint_states
```

---

## ⚠️ 문제 해결

### 1. "Package not found" 에러
```bash
cd ~/colcon_ws
rm -rf build install log
colcon build
source install/setup.bash
```

### 2. "Permission denied /dev/ttyACM0" 에러
```bash
sudo chmod 666 /dev/ttyACM0
```

### 3. "Controller manager not found" 에러
```bash
sudo apt install ros-humble-controller-manager
```

### 4. 모터가 움직이지 않을 때
- OpenCR 전원 확인 (12V 어댑터)
- 모터 ID 확인 (왼팔: 11-15, 오른팔: 21-25)
- USB 케이블 교체

---

## 📝 요약 실행 명령어

### 한 번만 실행 (설치)
```bash
cd ~
git clone https://github.com/mj035/CookingBot_2025.git
# 위의 1~8번 단계 실행
```

### 매번 실행
```bash
cd ~/colcon_ws
source install/setup.bash
ros2 launch open_manipulator_x_bringup dual_arm_hardware.launch.py
```

---

## 🤖 모터 ID 설정
- **왼쪽 팔 (Left Arm)**
  - Joint1: ID 11
  - Joint2: ID 12
  - Joint3: ID 13
  - Joint4: ID 14
  - Gripper: ID 15

- **오른쪽 팔 (Right Arm)**
  - Joint1: ID 21
  - Joint2: ID 22
  - Joint3: ID 23
  - Joint4: ID 24
  - Gripper: ID 25

---

## 📞 추가 도움
문제가 발생하면 GitHub Issues에 문의해주세요:
https://github.com/mj035/CookingBot_2025/issues