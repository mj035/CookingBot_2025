#!/usr/bin/env python3
"""
실물 로봇을 표준 초기 자세로 이동
토크 ON 상태로 초기 자세 설정 후 토크 OFF
"""

from dynamixel_sdk import *
import time

def set_initial_pose():
    """실물 로봇 초기 자세 설정"""
    
    # Dynamixel 설정
    PROTOCOL_VERSION = 2.0
    BAUDRATE = 1000000
    DEVICENAME = '/dev/ttyACM0'
    
    # Control table addresses
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132
    
    # 왼팔 모터 ID
    MOTOR_IDS = [11, 12, 13, 14, 15]
    
    # 표준 초기 자세 (실물 로봇 실측값)
    INITIAL_POSE = [0.0, -0.43, 1.94, -0.42, 2.5]  # Joint1~4 + Gripper
    
    # 라디안을 Dynamixel 값으로 변환
    def radian_to_value(radian):
        return int(radian / 0.00153398078 + 2048)
    
    print("🤖 실물 로봇 초기 자세 설정")
    print(f"📍 목표 자세: {INITIAL_POSE[:4]}")
    
    try:
        # 포트 연결
        port_handler = PortHandler(DEVICENAME)
        packet_handler = PacketHandler(PROTOCOL_VERSION)
        
        if not port_handler.openPort():
            print(f"❌ 포트 열기 실패: {DEVICENAME}")
            return
        
        if not port_handler.setBaudRate(BAUDRATE):
            print("❌ Baudrate 설정 실패")
            return
        
        print("✅ Dynamixel 연결 성공")
        
        # 1. 토크 ON
        print("\n🔒 토크 ON...")
        for motor_id in MOTOR_IDS:
            result, error = packet_handler.write1ByteTxRx(
                port_handler, motor_id, ADDR_TORQUE_ENABLE, 1
            )
            if result == COMM_SUCCESS:
                print(f"   모터 {motor_id}: 토크 ON")
        
        time.sleep(0.5)
        
        # 2. 초기 자세로 이동
        print("\n🎯 초기 자세로 이동 중...")
        for i, motor_id in enumerate(MOTOR_IDS):
            target_value = radian_to_value(INITIAL_POSE[i])
            packet_handler.write4ByteTxRx(
                port_handler, motor_id, ADDR_GOAL_POSITION, target_value
            )
            print(f"   모터 {motor_id}: {INITIAL_POSE[i]:.2f} rad")
        
        # 3. 이동 대기
        print("\n⏳ 이동 완료 대기 (3초)...")
        time.sleep(3)
        
        # 4. 현재 위치 확인
        print("\n📊 현재 위치 확인:")
        for motor_id in MOTOR_IDS[:4]:  # Joint1~4만
            present_pos, _, _ = packet_handler.read4ByteTxRx(
                port_handler, motor_id, ADDR_PRESENT_POSITION
            )
            radian = (present_pos - 2048) * 0.00153398078
            print(f"   모터 {motor_id}: {radian:.2f} rad")
        
        # 5. 토크 OFF
        print("\n🔓 토크 OFF...")
        for motor_id in MOTOR_IDS:
            packet_handler.write1ByteTxRx(
                port_handler, motor_id, ADDR_TORQUE_ENABLE, 0
            )
            print(f"   모터 {motor_id}: 토크 OFF")
        
        port_handler.closePort()
        print("\n✅ 초기 자세 설정 완료!")
        print("👋 이제 left_arm_teaching.py를 실행하세요")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == '__main__':
    set_initial_pose()