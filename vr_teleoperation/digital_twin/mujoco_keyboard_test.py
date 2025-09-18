#!/usr/bin/env python3
"""
MuJoCo Keyboard Control Test - 키보드로 MuJoCo 제어 + ROS2 하드웨어 동기화 테스트
양팔을 키보드로 제어하여 실물 로봇과의 동기화 확인
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import threading

# MuJoCo 경로
XML_SCENE_PATH = '../dual_arm/scene_dual.xml'

# 오프셋 보정값 (teaching_1.py에서 가져옴)
LEFT_OFFSETS = [0.0, -0.43, 1.94, -0.42]
RIGHT_OFFSETS = [0.66, -1.03, 0.96, -2.07]

# 조인트 안전 범위
JOINT_LIMITS = {
    'j1': (-1.57, 1.57),  # 실제 하드웨어 한계
    'j2': (-1.5, 1.5),
    'j3': (-1.5, 1.4),
    'j4': (-1.7, 1.97)
}
GRIPPER_RANGE = (-0.01, 0.019)

# 키보드 제어 설정
JOINT_INCREMENT = 0.05  # 한 번 누를 때 조인트 변화량
GRIPPER_INCREMENT = 0.005

class MuJoCoKeyboardController(Node):
    """키보드로 MuJoCo 제어 + ROS2 발행"""

    def __init__(self):
        super().__init__('mujoco_keyboard_controller')

        print("🎮 MuJoCo Keyboard Control Test")
        print("📡 ROS2 하드웨어 동기화 테스트")
        print("📊 오프셋 보정값 적용:")
        print(f"   왼팔:  {LEFT_OFFSETS}")
        print(f"   오른팔: {RIGHT_OFFSETS}\n")

        # MuJoCo 모델 로드
        self.model = mujoco.MjModel.from_xml_path(XML_SCENE_PATH)
        self.data = mujoco.MjData(self.model)

        # 액추에이터 매핑
        self.left_map = self._map_actuators(side="L")
        self.right_map = self._map_actuators(side="R")

        # 현재 조인트 값
        self.left_joints = [0.0, 0.0, 0.0, 0.0]
        self.right_joints = [0.0, 0.0, 0.0, 0.0]
        self.left_gripper = -0.01
        self.right_gripper = -0.01

        # 선택된 팔과 조인트
        self.selected_arm = 'left'  # 'left' or 'right'
        self.selected_joint = 0  # 0-3 for joints, 4 for gripper

        # ROS2 퍼블리셔 설정
        self.setup_ros2_publishers()

        # 퍼블리시 타이머 (50Hz)
        self.timer = self.create_timer(0.02, self.publish_to_hardware)

        # 초기 자세 설정
        self._set_initial_pose()

        print("\n✅ 시스템 준비 완료!")
        self.print_controls()

    def setup_ros2_publishers(self):
        """ROS2 퍼블리셔 설정"""
        # JointTrajectory 퍼블리셔 (하드웨어 제어용)
        self.left_traj_pub = self.create_publisher(
            JointTrajectory,
            '/left_arm_controller/joint_trajectory',
            10
        )
        self.right_traj_pub = self.create_publisher(
            JointTrajectory,
            '/right_arm_controller/joint_trajectory',
            10
        )

        # JointState 퍼블리셔 (상태 모니터링용)
        self.joint_state_pub = self.create_publisher(
            JointState,
            '/mujoco_joint_states',
            10
        )

        print("✅ ROS2 퍼블리셔 설정 완료")

    def _map_actuators(self, side="L"):
        """MuJoCo 액추에이터 매핑"""
        if side == "L":
            names = {
                "j1": "actuator_joint1",
                "j2": "actuator_joint2",
                "j3": "actuator_joint3",
                "j4": "actuator_joint4",
                "g": "actuator_gripper_joint",
            }
        else:
            names = {
                "j1": "actuator_joint1_r",
                "j2": "actuator_joint2_r",
                "j3": "actuator_joint3_r",
                "j4": "actuator_joint4_r",
                "g": "actuator_gripper_joint_r",
            }

        out = {}
        for k, nm in names.items():
            try:
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
            except:
                aid = -1
            out[k] = aid
        return out

    def _set_initial_pose(self):
        """초기 자세 설정 - 안전한 시작을 위한 단계적 접근"""
        # 초기 자세: [0.0, -0.3, 0.8, 0.0]
        # Joint2=-0.3 (약간 아래), Joint3=0.8 (팔꿈치 굽힘)
        init = [0.0, -0.3, 0.8, 0.0]

        print("\n⚠️  === 안전 초기화 시퀀스 ===")
        print("📍 목표 초기 자세:")
        print(f"   Joint1: {init[0]:.2f} (중립)")
        print(f"   Joint2: {init[1]:.2f} (약간 아래)")
        print(f"   Joint3: {init[2]:.2f} (팔꿈치 굽힘)")
        print(f"   Joint4: {init[3]:.2f} (중립)")

        print("\n🔧 실물 로봇을 다음 자세로 수동 조정하세요:")
        print("   1. 양팔을 정면으로 향하게")
        print("   2. 어깨를 약간 아래로 (30도)")
        print("   3. 팔꿈치를 굽혀서 자연스럽게")
        print("   4. 손목은 중립 위치로")

        print("\n⏸️  준비되면 Enter를 누르세요...")
        input()

        self.left_joints = init.copy()
        self.right_joints = init.copy()

        # MuJoCo 적용
        self.update_mujoco()

        # 시뮬레이션 스텝
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

        print("✅ 초기 자세 설정 완료 - 천천히 동기화됩니다")

        # 초기화 완료 플래그
        self.initialized = True

    def update_mujoco(self):
        """MuJoCo 액추에이터 업데이트"""
        # 왼팔
        for i, k in enumerate(['j1', 'j2', 'j3', 'j4']):
            if self.left_map[k] >= 0:
                self.data.ctrl[self.left_map[k]] = self.left_joints[i]
        if self.left_map['g'] >= 0:
            self.data.ctrl[self.left_map['g']] = self.left_gripper

        # 오른팔
        for i, k in enumerate(['j1', 'j2', 'j3', 'j4']):
            if self.right_map[k] >= 0:
                self.data.ctrl[self.right_map[k]] = self.right_joints[i]
        if self.right_map['g'] >= 0:
            self.data.ctrl[self.right_map['g']] = self.right_gripper

    def publish_to_hardware(self):
        """ROS2로 하드웨어에 명령 전송"""
        # 초기화 완료 전에는 전송하지 않음 (안전)
        if not hasattr(self, 'initialized'):
            return

        current_time = self.get_clock().now()

        # JointState 메시지 (모니터링용)
        joint_state = JointState()
        joint_state.header.stamp = current_time.to_msg()
        joint_state.name = [
            'joint1', 'joint2', 'joint3', 'joint4', 'gripper',
            'joint1_r', 'joint2_r', 'joint3_r', 'joint4_r', 'gripper_r'
        ]
        joint_state.position = (
            self.left_joints + [self.left_gripper] +
            self.right_joints + [self.right_gripper]
        )
        self.joint_state_pub.publish(joint_state)

        # 왼팔 JointTrajectory (오프셋 적용)
        left_traj = JointTrajectory()
        left_traj.header.stamp = current_time.to_msg()
        left_traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'gripper']

        left_point = JointTrajectoryPoint()
        # 오프셋 적용하여 하드웨어로 전송
        left_point.positions = [
            self.left_joints[0] + LEFT_OFFSETS[0],
            self.left_joints[1] + LEFT_OFFSETS[1],
            self.left_joints[2] + LEFT_OFFSETS[2],
            self.left_joints[3] + LEFT_OFFSETS[3],
            self.left_gripper
        ]
        left_point.time_from_start = Duration(sec=0, nanosec=100000000)  # 0.1초
        left_traj.points = [left_point]
        self.left_traj_pub.publish(left_traj)

        # 오른팔 JointTrajectory (오프셋 적용)
        right_traj = JointTrajectory()
        right_traj.header.stamp = current_time.to_msg()
        right_traj.joint_names = ['joint1_r', 'joint2_r', 'joint3_r', 'joint4_r', 'gripper_r']

        right_point = JointTrajectoryPoint()
        # 오프셋 적용하여 하드웨어로 전송
        right_point.positions = [
            self.right_joints[0] + RIGHT_OFFSETS[0],
            self.right_joints[1] + RIGHT_OFFSETS[1],
            self.right_joints[2] + RIGHT_OFFSETS[2],
            self.right_joints[3] + RIGHT_OFFSETS[3],
            self.right_gripper
        ]
        right_point.time_from_start = Duration(sec=0, nanosec=100000000)
        right_traj.points = [right_point]
        self.right_traj_pub.publish(right_traj)

    def handle_key(self, key):
        """키보드 입력 처리"""
        # 팔 선택
        if key == 'q':
            self.selected_arm = 'left'
            print(f"🖐 왼팔 선택됨")
        elif key == 'e':
            self.selected_arm = 'right'
            print(f"🖐 오른팔 선택됨")

        # 조인트 선택
        elif key in '12345':
            self.selected_joint = int(key) - 1
            joint_name = ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Gripper'][self.selected_joint]
            print(f"🎯 {self.selected_arm.upper()} {joint_name} 선택됨")

        # 값 조절
        elif key in '+-':
            increment = JOINT_INCREMENT if key == '+' else -JOINT_INCREMENT

            if self.selected_arm == 'left':
                if self.selected_joint < 4:
                    # 조인트 조절
                    joint_key = ['j1', 'j2', 'j3', 'j4'][self.selected_joint]
                    lo, hi = JOINT_LIMITS[joint_key]
                    self.left_joints[self.selected_joint] += increment
                    self.left_joints[self.selected_joint] = np.clip(
                        self.left_joints[self.selected_joint], lo, hi
                    )
                else:
                    # 그리퍼 조절
                    gripper_inc = GRIPPER_INCREMENT if key == '+' else -GRIPPER_INCREMENT
                    self.left_gripper += gripper_inc
                    self.left_gripper = np.clip(self.left_gripper, *GRIPPER_RANGE)
            else:
                if self.selected_joint < 4:
                    # 조인트 조절
                    joint_key = ['j1', 'j2', 'j3', 'j4'][self.selected_joint]
                    lo, hi = JOINT_LIMITS[joint_key]
                    self.right_joints[self.selected_joint] += increment
                    self.right_joints[self.selected_joint] = np.clip(
                        self.right_joints[self.selected_joint], lo, hi
                    )
                else:
                    # 그리퍼 조절
                    gripper_inc = GRIPPER_INCREMENT if key == '+' else -GRIPPER_INCREMENT
                    self.right_gripper += gripper_inc
                    self.right_gripper = np.clip(self.right_gripper, *GRIPPER_RANGE)

            # MuJoCo 업데이트
            self.update_mujoco()
            self.print_status()

        # 리셋
        elif key == 'r':
            self._set_initial_pose()
            print("🔄 초기 자세로 리셋")

        # 도움말
        elif key == 'h':
            self.print_controls()

    def print_status(self):
        """현재 상태 출력"""
        print(f"\n📊 현재 상태:")
        print(f"왼팔:  J1={self.left_joints[0]:.2f}, J2={self.left_joints[1]:.2f}, "
              f"J3={self.left_joints[2]:.2f}, J4={self.left_joints[3]:.2f}, G={self.left_gripper:.3f}")
        print(f"오른팔: J1={self.right_joints[0]:.2f}, J2={self.right_joints[1]:.2f}, "
              f"J3={self.right_joints[2]:.2f}, J4={self.right_joints[3]:.2f}, G={self.right_gripper:.3f}")

        # 오프셋 적용된 값 (실제 하드웨어로 전송되는 값)
        print(f"\n📡 하드웨어 전송값 (오프셋 적용):")
        left_hw = [self.left_joints[i] + LEFT_OFFSETS[i] for i in range(4)]
        right_hw = [self.right_joints[i] + RIGHT_OFFSETS[i] for i in range(4)]
        print(f"왼팔:  J1={left_hw[0]:.2f}, J2={left_hw[1]:.2f}, "
              f"J3={left_hw[2]:.2f}, J4={left_hw[3]:.2f}")
        print(f"오른팔: J1={right_hw[0]:.2f}, J2={right_hw[1]:.2f}, "
              f"J3={right_hw[2]:.2f}, J4={right_hw[3]:.2f}")

    def print_controls(self):
        """조작법 출력"""
        print("\n🎮 === 키보드 조작법 ===")
        print("팔 선택:")
        print("  Q - 왼팔 선택")
        print("  E - 오른팔 선택")
        print("\n조인트 선택:")
        print("  1 - Joint1 (좌우 회전)")
        print("  2 - Joint2 (상하)")
        print("  3 - Joint3 (상하)")
        print("  4 - Joint4 (손목 회전)")
        print("  5 - Gripper")
        print("\n값 조절:")
        print("  + - 증가")
        print("  - - 감소")
        print("\n기타:")
        print("  R - 초기 자세로 리셋")
        print("  H - 도움말")
        print("  ESC - 종료")
        print("=" * 30)

    def run_viewer(self):
        """MuJoCo 뷰어 실행"""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 카메라 설정
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 180
            viewer.cam.lookat[:] = [0.0, 0.0, 0.3]

            print("\n✨ MuJoCo 뷰어 시작됨")
            print("키보드로 제어하세요!\n")

            last_time = time.time()

            while viewer.is_running():
                # 키보드 입력 처리 (viewer의 키 이벤트 사용)
                if hasattr(viewer, '_key_pressed'):
                    key = viewer._key_pressed
                    if key:
                        self.handle_key(key)
                        viewer._key_pressed = None

                # MuJoCo 스텝
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # FPS 제어
                current_time = time.time()
                dt = current_time - last_time
                if dt < 0.01:  # 100 FPS
                    time.sleep(0.01 - dt)
                last_time = current_time

            print("\n🏁 뷰어 종료")

def main():
    # ROS2 초기화
    rclpy.init()

    try:
        # 컨트롤러 생성
        controller = MuJoCoKeyboardController()

        # ROS2 스피닝을 별도 스레드에서
        ros_thread = threading.Thread(
            target=lambda: rclpy.spin(controller),
            daemon=True
        )
        ros_thread.start()

        print("\n🚀 === MuJoCo 키보드 제어 테스트 ===")
        print("📡 하드웨어와 동기화를 테스트합니다")
        print("⚠️  하드웨어 런치파일을 먼저 실행하세요:")
        print("    ros2 launch open_manipulator_x_bringup dual_arm_hardware.launch.py")
        print("\n키보드 입력은 터미널에서 직접 입력하세요")

        # 뷰어 실행 (블로킹)
        controller.run_viewer()

    except KeyboardInterrupt:
        print("\n⚠️ 사용자 중단")
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()
        print("🏁 프로그램 종료")

if __name__ == "__main__":
    # 간단한 키보드 입력을 위한 대체 메인
    import sys
    import select
    import termios
    import tty

    def get_key():
        """비블로킹 키보드 입력"""
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            return key
        return None

    # 터미널 설정 저장
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        # Raw 모드 설정
        tty.setraw(sys.stdin.fileno())

        # ROS2 초기화
        rclpy.init()

        # 컨트롤러 생성
        controller = MuJoCoKeyboardController()

        # ROS2 스피닝 스레드
        ros_thread = threading.Thread(
            target=lambda: rclpy.spin(controller),
            daemon=True
        )
        ros_thread.start()

        print("\n🚀 === MuJoCo 키보드 제어 테스트 ===")
        print("📡 하드웨어와 동기화를 테스트합니다")
        print("⚠️  하드웨어 런치파일을 먼저 실행하세요:")
        print("    ros2 launch open_manipulator_x_bringup dual_arm_hardware.launch.py\n")

        # MuJoCo 뷰어와 키보드 제어 루프
        with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
            # 카메라 설정
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 180
            viewer.cam.lookat[:] = [0.0, 0.0, 0.3]

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            controller.print_controls()
            tty.setraw(sys.stdin.fileno())

            while viewer.is_running():
                # 키보드 입력 체크
                key = get_key()
                if key:
                    if key == '\x1b':  # ESC
                        break
                    elif key in 'qe12345+-rhQE+-RH':
                        # 일시적으로 일반 모드로 전환
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        controller.handle_key(key.lower())
                        tty.setraw(sys.stdin.fileno())

                # MuJoCo 스텝
                mujoco.mj_step(controller.model, controller.data)
                viewer.sync()
                time.sleep(0.01)

    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 터미널 설정 복원
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        rclpy.shutdown()
        print("\n🏁 프로그램 종료")