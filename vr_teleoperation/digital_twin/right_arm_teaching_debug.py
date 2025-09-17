#!/usr/bin/env python3
"""
오른팔 Direct Teaching - 디버그 버전
오프셋 적용 전후 값을 모두 출력
"""

import socket
import json
import time
import threading
from dynamixel_sdk import *
from datetime import datetime

class RightArmTeachingDebug:
    def __init__(self):
        print("\n🎓 Right Arm Direct Teaching - DEBUG MODE")
        print("📍 오프셋 적용 전후 값을 모두 확인합니다\n")

        # Dynamixel 설정
        self.PROTOCOL_VERSION = 2.0
        self.BAUDRATE = 1000000
        self.DEVICENAME = '/dev/ttyACM0'

        # Control table addresses
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_PRESENT_POSITION = 132

        # 오른팔 모터 ID
        self.RIGHT_ARM_IDS = [21, 22, 23, 24]
        self.RIGHT_GRIPPER_ID = 25

        # 현재 위치
        self.raw_joints = [0.0, 0.0, 0.0, 0.0]  # 보정 전 값
        self.current_joints = [0.0, 0.0, 0.0, 0.0]  # 보정 후 값
        self.current_gripper = 0.019

        # 오프셋 보정값
        self.joint_offsets = [0.61, -0.36, 1.84, -0.46]

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
        print(f"📊 오프셋: {self.joint_offsets}\n")

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

            # 오른팔 모터만 토크 해제
            print("🔓 오른팔 모터 토크 해제 중...")
            for motor_id in self.RIGHT_ARM_IDS + [self.RIGHT_GRIPPER_ID]:
                result, error = self.packet_handler.write1ByteTxRx(
                    self.port_handler, motor_id, self.ADDR_TORQUE_ENABLE, 0
                )
                if result == COMM_SUCCESS:
                    print(f"   모터 {motor_id}: 토크 OFF ✓")
                else:
                    print(f"   모터 {motor_id}: 토크 OFF 실패")

        except Exception as e:
            print(f"❌ Dynamixel 초기화 오류: {e}")
            exit()

    def connect_mujoco(self):
        """MuJoCo 소켓 연결"""
        print("🔌 MuJoCo 연결 시도 중...")
        for port in [12345, 12346]:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect(('localhost', port))
                print(f"🔗 MuJoCo 연결 성공 (포트 {port})")
                self.mujoco_socket = sock
                break
            except Exception as e:
                print(f"⚠️ 포트 {port} 연결 실패: {e}")
                if sock:
                    sock.close()

    def value_to_radian(self, value):
        """Dynamixel 값을 라디안으로 변환"""
        return (value - 2048) * 0.00153398078

    def read_loop(self):
        """실물 로봇 위치 읽기 루프"""
        first_print = True
        while self.running:
            try:
                # 오른팔 조인트 읽기
                for i, motor_id in enumerate(self.RIGHT_ARM_IDS):
                    present_position, result, error = self.packet_handler.read4ByteTxRx(
                        self.port_handler, motor_id, self.ADDR_PRESENT_POSITION
                    )
                    if result == COMM_SUCCESS:
                        self.raw_joints[i] = self.value_to_radian(present_position)
                        # 오프셋 보정 적용
                        self.current_joints[i] = self.raw_joints[i] - self.joint_offsets[i]

                # 그리퍼 읽기
                gripper_pos, result, error = self.packet_handler.read4ByteTxRx(
                    self.port_handler, self.RIGHT_GRIPPER_ID, self.ADDR_PRESENT_POSITION
                )
                if result == COMM_SUCCESS:
                    self.current_gripper = self.value_to_radian(gripper_pos)

                # 첫 읽기 시 상세 정보 출력
                if first_print:
                    print("\n📊 === 오프셋 적용 디버그 ===")
                    print(f"Raw 값 (실물):     {[f'{v:.3f}' for v in self.raw_joints]}")
                    print(f"오프셋:           {[f'{v:.3f}' for v in self.joint_offsets]}")
                    print(f"보정 후 (MuJoCo): {[f'{v:.3f}' for v in self.current_joints]}")
                    print("\n계산식: MuJoCo = Raw - Offset")
                    for i in range(4):
                        print(f"  Joint{i+1}: {self.current_joints[i]:.3f} = {self.raw_joints[i]:.3f} - {self.joint_offsets[i]:.3f}")
                    print("\n")
                    first_print = False

                # MuJoCo로 전송
                self.send_to_mujoco()
                time.sleep(0.01)  # 100Hz

            except Exception as e:
                if self.running:
                    print(f"⚠️ 읽기 오류: {e}")

    def send_to_mujoco(self):
        """MuJoCo로 현재 조인트 값 전송"""
        if self.mujoco_socket:
            try:
                data = {
                    'right_arm': {
                        'joint_angles': self.current_joints,
                        'gripper': self.current_gripper,
                        'calibrated': True
                    },
                    'timestamp': time.time()
                }
                json_data = json.dumps(data) + '\n'
                self.mujoco_socket.sendall(json_data.encode())
            except:
                pass

    def print_status(self):
        """현재 상태 출력 (디버그 정보 포함)"""
        print(f"\r[Raw] {[f'{j:.2f}' for j in self.raw_joints]} → [MuJoCo] {[f'{j:.2f}' for j in self.current_joints]}", end='')

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
        teaching = RightArmTeachingDebug()

        print("조작법:")
        print("  Q - 종료")
        print("  D - 현재 디버그 정보 출력\n")

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

                if key == 'd' or key == 'D':
                    print(f"\n\n=== 현재 디버그 정보 ===")
                    print(f"Raw 값:   {[f'{v:.3f}' for v in teaching.raw_joints]}")
                    print(f"오프셋:   {[f'{v:.3f}' for v in teaching.joint_offsets]}")
                    print(f"MuJoCo:   {[f'{v:.3f}' for v in teaching.current_joints]}")
                    print()

                if key == 'q' or key == 'Q':
                    break

            except:
                cmd = input("\n명령 (d/q): ").strip().lower()
                if cmd == 'd':
                    print(f"\n=== 현재 디버그 정보 ===")
                    print(f"Raw 값:   {[f'{v:.3f}' for v in teaching.raw_joints]}")
                    print(f"오프셋:   {[f'{v:.3f}' for v in teaching.joint_offsets]}")
                    print(f"MuJoCo:   {[f'{v:.3f}' for v in teaching.current_joints]}")
                elif cmd == 'q':
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