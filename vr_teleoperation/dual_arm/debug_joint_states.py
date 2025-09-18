#!/usr/bin/env python3
"""
Joint States 토픽 디버깅 - 실제 인덱스 매핑 확인
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time

class JointStateDebugger(Node):
    def __init__(self):
        super().__init__('joint_state_debugger')

        print("\n🔍 === Joint States 디버깅 시작 ===")
        print("양팔 하드웨어 연결 후 이 스크립트를 실행하세요\n")

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )

        self.first_msg = True
        self.msg_count = 0

    def joint_states_callback(self, msg):
        """Joint States 메시지 분석"""
        self.msg_count += 1

        if self.first_msg:
            print("=" * 80)
            print(f"📊 첫 번째 Joint States 메시지 분석")
            print("=" * 80)

            print(f"\n총 조인트 개수: {len(msg.name)}")
            print(f"Position 배열 크기: {len(msg.position)}")
            print(f"Velocity 배열 크기: {len(msg.velocity) if msg.velocity else 0}")
            print(f"Effort 배열 크기: {len(msg.effort) if msg.effort else 0}")

            print("\n🔧 전체 조인트 매핑 (인덱스 → 이름 → 위치값):")
            print("-" * 60)
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    print(f"  [{i:2d}] {name:20s} = {msg.position[i]:+.4f} rad")
                else:
                    print(f"  [{i:2d}] {name:20s} = (no position)")

            print("\n📌 왼팔 조인트 (joint1~4, gripper_joint):")
            print("-" * 60)
            left_joints = {}
            for i, name in enumerate(msg.name):
                if name in ['joint1', 'joint2', 'joint3', 'joint4', 'gripper_joint']:
                    left_joints[name] = i
                    if i < len(msg.position):
                        print(f"  {name:15s}: index={i:2d}, value={msg.position[i]:+.4f}")

            print("\n📌 오른팔 조인트 (right_joint1~4, right_gripper_joint):")
            print("-" * 60)
            right_joints = {}
            for i, name in enumerate(msg.name):
                if 'right_' in name:
                    right_joints[name] = i
                    if i < len(msg.position):
                        print(f"  {name:20s}: index={i:2d}, value={msg.position[i]:+.4f}")

            # 코드 생성 제안
            print("\n✨ mirror_dual.py에서 사용할 수정된 코드:")
            print("=" * 80)
            print("# 왼팔 초기값 저장 부분 (line 117-142 수정)")
            print("if self.robot_initial['left'] is None:")
            print("    self.robot_initial['left'] = []")
            print("    left_joint_indices = {}")
            print("    ")
            print("    # 실제 인덱스 매핑")
            for name in ['joint1', 'joint2', 'joint3', 'joint4']:
                if name in left_joints:
                    print(f"    left_joint_indices['{name}'] = {left_joints[name]}")
            print("    ")
            print("    # 올바른 순서로 저장")
            print("    for joint_name in ['joint1', 'joint2', 'joint3', 'joint4']:")
            print("        if joint_name in left_joint_indices:")
            print("            idx = left_joint_indices[joint_name]")
            print("            if idx < len(msg.position):")
            print("                self.robot_initial['left'].append(msg.position[idx])")
            print("=" * 80)

            self.first_msg = False

        # 주기적 업데이트 (5초마다)
        if self.msg_count % 50 == 0:
            print(f"\n⏱️  [{time.strftime('%H:%M:%S')}] 메시지 #{self.msg_count}")
            print("현재 조인트 위치:")

            # 왼팔
            print("  왼팔: ", end="")
            for name in ['joint1', 'joint2', 'joint3', 'joint4']:
                for i, n in enumerate(msg.name):
                    if name == n and i < len(msg.position):
                        print(f"{name}={msg.position[i]:+.3f} ", end="")
            print()

            # 오른팔
            print("  오른팔: ", end="")
            for name in ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4']:
                for i, n in enumerate(msg.name):
                    if name == n and i < len(msg.position):
                        short_name = name.replace('right_', '')
                        print(f"{short_name}={msg.position[i]:+.3f} ", end="")
            print()

            # 특이사항 체크
            print("\n⚠️  특이사항 체크:")

            # 왼팔 joint 체크
            unusual = []
            for i, name in enumerate(msg.name):
                if name in ['joint1', 'joint2', 'joint3', 'joint4'] and i < len(msg.position):
                    val = msg.position[i]
                    if abs(val) > 1.5:  # 비정상적으로 큰 값
                        unusual.append(f"{name}(idx={i})={val:.3f}")

            if unusual:
                print(f"  왼팔 비정상 값: {', '.join(unusual)}")
            else:
                print(f"  왼팔 정상 범위")

            # 오른팔 joint 체크
            unusual = []
            for i, name in enumerate(msg.name):
                if 'right_' in name and 'joint' in name and i < len(msg.position):
                    val = msg.position[i]
                    if abs(val) > 1.5:
                        unusual.append(f"{name}(idx={i})={val:.3f}")

            if unusual:
                print(f"  오른팔 비정상 값: {', '.join(unusual)}")
            else:
                print(f"  오른팔 정상 범위")

def main():
    rclpy.init()

    try:
        debugger = JointStateDebugger()

        print("\n🔍 Joint States 토픽 모니터링 중...")
        print("⚠️  Ctrl+C로 종료\n")

        rclpy.spin(debugger)

    except KeyboardInterrupt:
        print("\n\n🏁 디버깅 종료")
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()