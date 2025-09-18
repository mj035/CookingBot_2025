#!/usr/bin/env python3
"""
양팔 Direct Teaching - 실물 양팔을 동시에 MuJoCo와 동기화
토크 OFF 상태로 양팔 데이터 수집
왼팔 오프셋 재보정 버전
"""

import socket
import json
import time
import threading
from dynamixel_sdk import *
from datetime import datetime

class DualArmTeaching:
    def __init__(self):
        print("\n🎓 Dual Arm Direct Teaching Mode (왼팔 오프셋 재보정)")
        print("📍 양팔을 손으로 움직이면 MuJoCo가 실시간으로 따라옵니다")
        print("💾 Space: 현재 자세 저장 | Q: 종료\n")

        # Dynamixel 설정
        self.PROTOCOL_VERSION = 2.0
        self.BAUDRATE = 1000000
        self.DEVICENAME = '/dev/ttyACM0'

        # Control table addresses
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_PRESENT_POSITION = 132

        # 양팔 모터 ID
        self.LEFT_ARM_IDS = [11, 12, 13, 14]
        self.LEFT_GRIPPER_ID = 15
        self.RIGHT_ARM_IDS = [21, 22, 23, 24]
        self.RIGHT_GRIPPER_ID = 25

        # 현재 위치
        self.left_joints = [0.0, 0.0, 0.0, 0.0]
        self.left_gripper = 0.019
        self.right_joints = [0.0, 0.0, 0.0, 0.0]
        self.right_gripper = 0.019

        # 오프셋 보정값 (왼팔 재보정됨!)
        self.left_offsets = [0.10, -0.43, 1.87, -1.68]  # 왼팔 오프셋 수정
        self.right_offsets = [0.66, -1.03, 0.96, -2.07]  # 오른팔 유지

        print("📊 적용된 오프셋:")
        print(f"   왼팔:  {self.left_offsets} (재보정됨)")
        print(f"   오른팔: {self.right_offsets}\n")

        # 첫 전송 지연 플래그
        self.first_read_done = False

        # MuJoCo 소켓
        self.mujoco_socket = None
        self.running = True

        # Dynamixel 초기화
        self.setup_dynamixel()

        # MuJoCo 연결
        self.connect_mujoco()

        # 읽기 스레드 시작
        self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.read_thread.start()

        print("\n✅ 시스템 준비 완료!")
        print("🖐 양팔을 천천히 움직여보세요\n")

    def setup_dynamixel(self):
        """Dynamixel 초기화 및 토크 해제"""
        try:
            self.port_handler = PortHandler(self.DEVICENAME)
            self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)

            if not self.port_handler.openPort():
                print(f"❌ 포트 열기 실패: {self.DEVICENAME}")
                self.DEVICENAME = '/dev/ttyUSB0'
                self.port_handler = PortHandler(self.DEVICENAME)
                if not self.port_handler.openPort():
                    exit()

            if not self.port_handler.setBaudRate(self.BAUDRATE):
                print("❌ Baudrate 설정 실패")
                exit()

            print(f"✅ Dynamixel 연결 성공: {self.DEVICENAME}")

            # 양팔 모터 토크 해제
            print("🔓 양팔 모터 토크 해제 중...")
            all_motors = (self.LEFT_ARM_IDS + [self.LEFT_GRIPPER_ID] +
                         self.RIGHT_ARM_IDS + [self.RIGHT_GRIPPER_ID])

            for motor_id in all_motors:
                result, error = self.packet_handler.write1ByteTxRx(
                    self.port_handler, motor_id, self.ADDR_TORQUE_ENABLE, 0
                )
                if result == COMM_SUCCESS:
                    print(f"   모터 {motor_id}: 토크 OFF ✓")

            print("✋ 양팔을 손으로 움직일 수 있습니다!")

        except Exception as e:
            print(f"❌ Dynamixel 초기화 오류: {e}")
            exit()

    def connect_mujoco(self):
        """MuJoCo 소켓 연결"""
        print("🔌 MuJoCo 연결 시도 중...")
        self.mujoco_socket = None

        for port in [12345, 12346]:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect(('localhost', port))
                print(f"🔗 MuJoCo 연결 성공 (포트 {port})")

                # 연결 테스트
                test_data = {'test': 'connection'}
                sock.sendall((json.dumps(test_data) + '\n').encode())
                print("✅ 연결 테스트 완료")

                self.mujoco_socket = sock
                break

            except Exception as e:
                print(f"⚠️ 포트 {port} 연결 실패: {e}")
                if sock:
                    sock.close()

        if not self.mujoco_socket:
            print("💡 MuJoCo가 먼저 실행 중인지 확인하세요")

    def value_to_radian(self, value):
        """Dynamixel 값을 라디안으로 변환"""
        return (value - 2048) * 0.00153398078

    def read_loop(self):
        """실물 로봇 위치 읽기 루프"""
        while self.running:
            try:
                # 왼팔 조인트 읽기
                for i, motor_id in enumerate(self.LEFT_ARM_IDS):
                    present_position, result, error = self.packet_handler.read4ByteTxRx(
                        self.port_handler, motor_id, self.ADDR_PRESENT_POSITION
                    )
                    if result == COMM_SUCCESS:
                        raw_value = self.value_to_radian(present_position)
                        self.left_joints[i] = raw_value - self.left_offsets[i]

                # 왼팔 그리퍼 읽기
                gripper_pos, result, error = self.packet_handler.read4ByteTxRx(
                    self.port_handler, self.LEFT_GRIPPER_ID, self.ADDR_PRESENT_POSITION
                )
                if result == COMM_SUCCESS:
                    self.left_gripper = self.value_to_radian(gripper_pos)

                # 오른팔 조인트 읽기
                for i, motor_id in enumerate(self.RIGHT_ARM_IDS):
                    present_position, result, error = self.packet_handler.read4ByteTxRx(
                        self.port_handler, motor_id, self.ADDR_PRESENT_POSITION
                    )
                    if result == COMM_SUCCESS:
                        raw_value = self.value_to_radian(present_position)
                        self.right_joints[i] = raw_value - self.right_offsets[i]

                # 오른팔 그리퍼 읽기
                gripper_pos, result, error = self.packet_handler.read4ByteTxRx(
                    self.port_handler, self.RIGHT_GRIPPER_ID, self.ADDR_PRESENT_POSITION
                )
                if result == COMM_SUCCESS:
                    self.right_gripper = self.value_to_radian(gripper_pos)

                # 첫 읽기 완료 표시
                if not self.first_read_done:
                    print("📊 양팔 오프셋 보정 적용됨")
                    print(f"   왼팔 보정 후:  {[f'{j:.2f}' for j in self.left_joints]}")
                    print(f"   오른팔 보정 후: {[f'{j:.2f}' for j in self.right_joints]}")
                    self.first_read_done = True

                # MuJoCo로 전송
                self.send_to_mujoco()

                time.sleep(0.01)  # 100Hz

            except Exception as e:
                if self.running:
                    print(f"⚠️ 읽기 오류: {e}")

    def send_to_mujoco(self):
        """MuJoCo로 양팔 조인트 값 전송"""
        if self.mujoco_socket:
            try:
                data = {
                    'left_arm': {
                        'joint_angles': self.left_joints,
                        'gripper': self.left_gripper,
                        'calibrated': True
                    },
                    'right_arm': {
                        'joint_angles': self.right_joints,
                        'gripper': self.right_gripper,
                        'calibrated': True
                    },
                    'timestamp': time.time()
                }
                json_data = json.dumps(data) + '\n'
                self.mujoco_socket.sendall(json_data.encode())
            except:
                pass

    def print_status(self):
        """현재 상태 출력"""
        left_str = ', '.join([f'{j:.2f}' for j in self.left_joints])
        right_str = ', '.join([f'{j:.2f}' for j in self.right_joints])
        print(f"\r왼팔: [{left_str}] | 오른팔: [{right_str}]", end='')

    def cleanup(self):
        """종료 처리"""
        self.running = False
        time.sleep(0.1)

        if hasattr(self, 'port_handler'):
            self.port_handler.closePort()
            print("\n✅ Dynamixel 포트 닫음")

        if self.mujoco_socket:
            self.mujoco_socket.close()
            print("✅ MuJoCo 연결 종료")

def main():
    teaching = None

    try:
        teaching = DualArmTeaching()

        print("\n조작법:")
        print("  Q - 종료\n")

        while True:
            teaching.print_status()

            try:
                import sys, tty, termios
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    key = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

                if key == 'q' or key == 'Q':
                    break

            except:
                cmd = input("\n명령 (q): ").strip().lower()
                if cmd == 'q':
                    break

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n🛑 중단됨")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
    finally:
        if teaching:
            teaching.cleanup()
        print("🏁 프로그램 종료")

if __name__ == '__main__':
    main()