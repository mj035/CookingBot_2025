#!/usr/bin/env python3
"""
안전한 시작 스크립트 - 하드웨어와 MuJoCo 초기 자세 맞추기
실물 로봇의 현재 위치를 읽어서 MuJoCo와 동기화
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time
import numpy as np

class SafeStartup(Node):
    def __init__(self):
        super().__init__('safe_startup')

        print("🛡️ === 안전 시작 프로그램 ===")
        print("📍 실물 로봇의 현재 위치를 확인합니다\n")

        # 오프셋 값
        self.LEFT_OFFSETS = [0.0, -0.43, 1.94, -0.42]
        self.RIGHT_OFFSETS = [0.66, -1.03, 0.96, -2.07]

        # MuJoCo 목표 자세
        self.target_pose = [0.0, -0.3, 0.8, 0.0]

        # JointState 구독
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        self.current_joints = None
        self.check_timer = self.create_timer(0.5, self.check_position)

        print("⏳ 하드웨어 런치파일이 실행 중인지 확인 중...")

    def joint_callback(self, msg):
        """현재 조인트 상태 수신"""
        if len(msg.position) >= 10:  # 양팔 데이터
            self.current_joints = {
                'left': list(msg.position[:5]),
                'right': list(msg.position[5:10])
            }

    def check_position(self):
        """현재 위치 확인 및 안전성 체크"""
        if self.current_joints is None:
            print("❌ 하드웨어 데이터 수신 안됨. 런치파일을 확인하세요")
            return

        print("\n📊 === 현재 하드웨어 위치 ===")

        # 왼팔 위치 (오프셋 제거하여 MuJoCo 좌표로 변환)
        left_mujoco = []
        for i in range(4):
            raw = self.current_joints['left'][i]
            mujoco_val = raw - self.LEFT_OFFSETS[i]
            left_mujoco.append(mujoco_val)

        # 오른팔 위치
        right_mujoco = []
        for i in range(4):
            raw = self.current_joints['right'][i]
            mujoco_val = raw - self.RIGHT_OFFSETS[i]
            right_mujoco.append(mujoco_val)

        print(f"왼팔 (MuJoCo 좌표):")
        print(f"  현재: J1={left_mujoco[0]:.2f}, J2={left_mujoco[1]:.2f}, "
              f"J3={left_mujoco[2]:.2f}, J4={left_mujoco[3]:.2f}")
        print(f"  목표: J1={self.target_pose[0]:.2f}, J2={self.target_pose[1]:.2f}, "
              f"J3={self.target_pose[2]:.2f}, J4={self.target_pose[3]:.2f}")

        print(f"\n오른팔 (MuJoCo 좌표):")
        print(f"  현재: J1={right_mujoco[0]:.2f}, J2={right_mujoco[1]:.2f}, "
              f"J3={right_mujoco[2]:.2f}, J4={right_mujoco[3]:.2f}")
        print(f"  목표: J1={self.target_pose[0]:.2f}, J2={self.target_pose[1]:.2f}, "
              f"J3={self.target_pose[2]:.2f}, J4={self.target_pose[3]:.2f}")

        # 차이 계산
        left_diff = [abs(left_mujoco[i] - self.target_pose[i]) for i in range(4)]
        right_diff = [abs(right_mujoco[i] - self.target_pose[i]) for i in range(4)]

        max_left_diff = max(left_diff)
        max_right_diff = max(right_diff)

        print(f"\n📏 최대 차이:")
        print(f"  왼팔: {max_left_diff:.3f} rad")
        print(f"  오른팔: {max_right_diff:.3f} rad")

        # 안전 판정
        SAFE_THRESHOLD = 0.5  # 0.5 라디안 (약 28도) 이내면 안전

        if max_left_diff < SAFE_THRESHOLD and max_right_diff < SAFE_THRESHOLD:
            print("\n✅ === 안전 확인 완료 ===")
            print("실물 로봇이 목표 초기 자세와 비슷합니다.")
            print("MuJoCo 테스트를 시작해도 안전합니다!")

        else:
            print("\n⚠️  === 주의 필요 ===")
            print("실물 로봇과 목표 자세 차이가 큽니다!")
            print("\n🔧 다음 자세로 수동 조정 후 다시 시도하세요:")
            print("  1. 양팔을 정면으로")
            print("  2. 어깨를 약간 아래로 (30도)")
            print("  3. 팔꿈치를 굽혀서")
            print("  4. 손목은 중립으로")

            # 어느 조인트가 문제인지 표시
            print("\n특히 조정이 필요한 부분:")
            for i, diff in enumerate(left_diff):
                if diff > SAFE_THRESHOLD:
                    print(f"  왼팔 Joint{i+1}: {diff:.2f} rad 차이")
            for i, diff in enumerate(right_diff):
                if diff > SAFE_THRESHOLD:
                    print(f"  오른팔 Joint{i+1}: {diff:.2f} rad 차이")

        print("\n" + "="*50)

def main():
    rclpy.init()

    try:
        node = SafeStartup()

        print("\n🚀 안전 체크 시작...")
        print("Ctrl+C로 종료\n")

        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\n종료됨")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()