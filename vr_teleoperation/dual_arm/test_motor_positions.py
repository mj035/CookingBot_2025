#!/usr/bin/env python3
"""
모터별 실제 위치 확인 테스트
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time

class MotorPositionTest(Node):
    def __init__(self):
        super().__init__('motor_position_test')

        print("\n🔍 === 모터 위치 테스트 ===")
        print("양팔을 똑같은 자세로 놓고 실행하세요\n")

        # Publishers
        self.left_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.right_pub = self.create_publisher(
            JointTrajectory, '/right_arm_controller/joint_trajectory', 10)

        # Subscriber
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)

        self.test_phase = 0
        self.test_positions = [0.0, 0.0, 0.0, 0.0]

        # Timer for test sequence
        self.timer = self.create_timer(3.0, self.run_test)

        self.left_joints = {}
        self.right_joints = {}

    def joint_callback(self, msg):
        """현재 조인트 값 저장"""
        # 왼팔
        for name in ['joint1', 'joint2', 'joint3', 'joint4']:
            for i, n in enumerate(msg.name):
                if n == name and i < len(msg.position):
                    self.left_joints[name] = msg.position[i]

        # 오른팔
        for name in ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4']:
            for i, n in enumerate(msg.name):
                if n == name and i < len(msg.position):
                    self.right_joints[name] = msg.position[i]

    def send_position(self, positions, arm='left'):
        """특정 위치로 이동 명령"""
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()

        if arm == 'left':
            traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
            pub = self.left_pub
        else:
            traj.joint_names = ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4']
            pub = self.right_pub

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=2, nanosec=0)

        traj.points = [point]
        pub.publish(traj)

    def run_test(self):
        """테스트 시퀀스"""
        if self.test_phase == 0:
            print("=" * 60)
            print("📊 현재 조인트 위치 비교")
            print("-" * 60)

            # 왼팔
            if self.left_joints:
                print("왼팔:")
                for name in ['joint1', 'joint2', 'joint3', 'joint4']:
                    if name in self.left_joints:
                        print(f"  {name}: {self.left_joints[name]:.4f}")

            # 오른팔
            if self.right_joints:
                print("\n오른팔:")
                for name in ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4']:
                    if name in self.right_joints:
                        short_name = name.replace('right_', '')
                        print(f"  {short_name}: {self.right_joints[name]:.4f}")

            # 차이 분석
            if 'joint3' in self.left_joints and 'right_joint3' in self.right_joints:
                diff = self.left_joints['joint3'] - self.right_joints['right_joint3']
                print(f"\n⚠️  Joint3 차이: {diff:.4f} rad ({diff*180/3.14159:.1f}°)")

                if abs(diff) > 0.5:
                    print("❌ Joint3 차이가 너무 큽니다! 모터 영점 문제일 가능성 높음")

            print("\n🔄 3초 후 Joint3만 0으로 이동 테스트...")
            self.test_phase = 1

        elif self.test_phase == 1:
            print("\n🎯 Joint3를 0 위치로 이동 중...")

            # 현재 위치 유지하되 joint3만 0으로
            left_pos = [
                self.left_joints.get('joint1', 0.0),
                self.left_joints.get('joint2', 0.0),
                0.0,  # joint3를 0으로
                self.left_joints.get('joint4', 0.0)
            ]

            right_pos = [
                self.right_joints.get('right_joint1', 0.0),
                self.right_joints.get('right_joint2', 0.0),
                0.0,  # joint3를 0으로
                self.right_joints.get('right_joint4', 0.0)
            ]

            self.send_position(left_pos, 'left')
            self.send_position(right_pos, 'right')

            print("왼팔 joint3: 현재 {:.3f} → 0.0".format(
                self.left_joints.get('joint3', 0.0)))
            print("오른팔 joint3: 현재 {:.3f} → 0.0".format(
                self.right_joints.get('right_joint3', 0.0)))

            self.test_phase = 2

        elif self.test_phase == 2:
            print("\n✅ 이동 완료. 현재 위치:")
            print(f"왼팔 joint3: {self.left_joints.get('joint3', 0.0):.4f}")
            print(f"오른팔 joint3: {self.right_joints.get('right_joint3', 0.0):.4f}")

            print("\n💡 두 팔의 물리적 자세를 비교해보세요:")
            print("   - 같은 자세면: 소프트웨어 영점 문제")
            print("   - 다른 자세면: 하드웨어 영점 문제")

            self.test_phase = 3
            self.timer.cancel()

def main():
    rclpy.init()

    try:
        test = MotorPositionTest()
        rclpy.spin(test)
    except KeyboardInterrupt:
        print("\n테스트 중단")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()