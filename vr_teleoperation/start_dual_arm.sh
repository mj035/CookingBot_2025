#!/bin/bash
# 양팔 VR 텔레오퍼레이션 시작 스크립트

echo "🚀 양팔 VR 텔레오퍼레이션 시작"
echo "================================"

# 터미널 1: ROS2 브릿지
echo "1. ROS2 브릿지 실행 (dual_arm_bridge.py)..."
gnome-terminal --title="Dual Arm Bridge" -- bash -c "cd ~/CookingBot_2025/vr_teleoperation && python3 dual_arm_bridge.py; exec bash"

sleep 2

# 터미널 2: MuJoCo 시뮬레이터
echo "2. MuJoCo 시뮬레이터 실행 (mujoco_mirror.py)..."
gnome-terminal --title="MuJoCo Dual Arm" -- bash -c "cd ~/CookingBot_2025/vr_teleoperation && python3 mujoco_mirror.py; exec bash"

echo ""
echo "✅ 모든 프로세스가 시작되었습니다!"
echo ""
echo "🎮 사용법:"
echo "  - 왼쪽 VR 컨트롤러 → 왼쪽 로봇팔"
echo "  - 오른쪽 VR 컨트롤러 → 오른쪽 로봇팔"
echo "  - 트리거 → 그리퍼 제어"
echo "  - A+B 버튼 → 재캘리브레이션"
echo ""
echo "종료하려면 각 터미널에서 Ctrl+C를 누르세요."