#!/usr/bin/env python3
"""
왼팔 Direct Teaching - 실물 로봇을 손으로 움직여서 MuJoCo 동기화
토크 OFF 상태로 안전하게 데이터 수집
"""

import socket
import json
import time
import threading
from dynamixel_sdk import *
from datetime import datetime

class LeftArmTeaching:
    def __init__(self):
        print("\n🎓 Left Arm Direct Teaching Mode")
        print("📍 왼팔을 손으로 움직이면 MuJoCo가 실시간으로 따라옵니다")
        print("💾 Space: 현재 자세 저장 | Q: 종료\n")
        
        # Dynamixel 설정
        self.PROTOCOL_VERSION = 2.0
        self.BAUDRATE = 1000000
        self.DEVICENAME = '/dev/ttyACM0'  # 필요시 수정
        
        # Control table addresses
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_PRESENT_POSITION = 132
        
        # 왼팔 모터 ID (11-15)
        self.LEFT_ARM_IDS = [11, 12, 13, 14]
        self.LEFT_GRIPPER_ID = 15
        
        # 현재 위치
        self.current_joints = [0.0, 0.0, 0.0, 0.0]
        self.current_gripper = 0.019
        
        # 오프셋 보정값 (실물 로봇 영점과 MuJoCo 영점의 차이)
        # 실물이 [0, -0.43, 1.94, -0.42]일 때 MuJoCo는 [0, 0, 0, 0]이어야 함
        self.joint_offsets = [0.0, -0.43, 1.94, -0.42]
        
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
        print("🖐 왼팔을 천천히 움직여보세요\n")
    
    def setup_dynamixel(self):
        """Dynamixel 초기화 및 토크 해제"""
        try:
            self.port_handler = PortHandler(self.DEVICENAME)
            self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)
            
            if not self.port_handler.openPort():
                print(f"❌ 포트 열기 실패: {self.DEVICENAME}")
                print("💡 다른 포트 시도: /dev/ttyUSB0")
                self.DEVICENAME = '/dev/ttyUSB0'
                self.port_handler = PortHandler(self.DEVICENAME)
                if not self.port_handler.openPort():
                    exit()
            
            if not self.port_handler.setBaudRate(self.BAUDRATE):
                print("❌ Baudrate 설정 실패")
                exit()
            
            print(f"✅ Dynamixel 연결 성공: {self.DEVICENAME}")
            
            # 왼팔 모터만 토크 해제
            print("🔓 왼팔 모터 토크 해제 중...")
            for motor_id in self.LEFT_ARM_IDS + [self.LEFT_GRIPPER_ID]:
                result, error = self.packet_handler.write1ByteTxRx(
                    self.port_handler, motor_id, self.ADDR_TORQUE_ENABLE, 0
                )
                if result == COMM_SUCCESS:
                    print(f"   모터 {motor_id}: 토크 OFF ✓")
                else:
                    print(f"   모터 {motor_id}: 토크 OFF 실패 (모터가 연결되지 않았을 수 있음)")
            
            print("✋ 왼팔을 손으로 움직일 수 있습니다!")
            
        except Exception as e:
            print(f"❌ Dynamixel 초기화 오류: {e}")
            exit()
    
    def connect_mujoco(self):
        """MuJoCo 소켓 연결"""
        print("🔌 MuJoCo 연결 시도 중...")
        self.mujoco_socket = None
        
        # 포트 12345, 12346 순서로 시도
        for port in [12345, 12346]:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)  # 2초 타임아웃
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
            print("💡 또는 모든 프로세스 종료 후 재시작하세요")
    
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
                        # 오프셋 보정 적용 (실물 값 - 오프셋 = MuJoCo 값)
                        self.current_joints[i] = raw_value - self.joint_offsets[i]
                
                # 그리퍼 읽기
                gripper_pos, result, error = self.packet_handler.read4ByteTxRx(
                    self.port_handler, self.LEFT_GRIPPER_ID, self.ADDR_PRESENT_POSITION
                )
                if result == COMM_SUCCESS:
                    self.current_gripper = self.value_to_radian(gripper_pos)
                
                # 첫 읽기 완료 표시
                if not self.first_read_done:
                    print("📊 오프셋 보정 적용됨")
                    print(f"   보정 후: {[f'{j:.2f}' for j in self.current_joints]}")
                    self.first_read_done = True
                
                # MuJoCo로 전송
                self.send_to_mujoco()
                
                time.sleep(0.01)  # 100Hz
                
            except Exception as e:
                if self.running:
                    print(f"⚠️ 읽기 오류: {e}")
    
    def send_to_mujoco(self):
        """MuJoCo로 현재 조인트 값 전송 (오프셋 보정됨)"""
        if self.mujoco_socket:
            try:
                data = {
                    'left_arm': {
                        'joint_angles': self.current_joints,
                        'gripper': self.current_gripper,
                        'calibrated': True
                    },
                    'timestamp': time.time()
                }
                json_data = json.dumps(data) + '\n'
                self.mujoco_socket.sendall(json_data.encode())
            except:
                # 연결 끊김 무시
                pass
    
    def get_current_state(self):
        """현재 상태 반환"""
        return {
            'joints': self.current_joints.copy(),
            'gripper': self.current_gripper,
            'timestamp': datetime.now().isoformat()
        }
    
    def print_status(self):
        """현재 상태 출력"""
        joints_str = ', '.join([f'{j:.2f}' for j in self.current_joints])
        print(f"\r조인트: [{joints_str}] | 그리퍼: {self.current_gripper:.2f}", end='')
    
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
    collected_data = []
    
    try:
        teaching = LeftArmTeaching()
        
        print("\n조작법:")
        print("  Space - 현재 자세 저장")
        print("  S - 데이터 파일로 저장")
        print("  Q - 종료\n")
        
        space_pressed = False
        
        while True:
            # 상태 출력
            teaching.print_status()
            
            # 키 입력 처리 (간단한 방식)
            try:
                import sys, tty, termios
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    key = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                
                if key == ' ' and not space_pressed:
                    # 현재 상태 저장
                    state = teaching.get_current_state()
                    collected_data.append(state)
                    print(f"\n💾 샘플 #{len(collected_data)} 저장됨")
                    space_pressed = True
                elif key != ' ':
                    space_pressed = False
                
                if key == 's' or key == 'S':
                    # 데이터 파일로 저장
                    if collected_data:
                        filename = f"left_arm_teaching_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(filename, 'w') as f:
                            json.dump(collected_data, f, indent=2)
                        print(f"\n✅ {len(collected_data)}개 샘플을 {filename}에 저장")
                
                if key == 'q' or key == 'Q':
                    break
                    
            except:
                # 키 입력 방식이 안 되면 input 사용
                cmd = input("\n명령 (space/s/q): ").strip().lower()
                if cmd == 'space' or cmd == ' ':
                    state = teaching.get_current_state()
                    collected_data.append(state)
                    print(f"💾 샘플 #{len(collected_data)} 저장됨")
                elif cmd == 's':
                    if collected_data:
                        filename = f"left_arm_teaching_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(filename, 'w') as f:
                            json.dump(collected_data, f, indent=2)
                        print(f"✅ {len(collected_data)}개 샘플을 {filename}에 저장")
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
        
        # 수집한 데이터 자동 저장
        if collected_data:
            filename = f"left_arm_teaching_autosave_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(collected_data, f, indent=2)
            print(f"💾 자동 저장: {filename}")
        
        print("🏁 프로그램 종료")

if __name__ == '__main__':
    main()