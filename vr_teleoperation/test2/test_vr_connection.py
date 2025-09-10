#!/usr/bin/env python3
"""
VR 연결 테스트 - Docker와 Host 간 통신 확인
"""

import socket
import json
import time

def test_send_vr_data():
    """VR 데이터를 sync_data_recorder로 직접 전송 테스트"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 12346))
        print("✅ 포트 12346 연결 성공!")
        
        # 테스트 데이터
        test_data = {
            'vr_data': {
                'position': [0.1, 0.2, 0.3],
                'orientation': [0, 0, 0, 1],
                'trigger': 0.5,
                'calibrated': True
            },
            'timestamp': time.time()
        }
        
        # 5번 전송
        for i in range(5):
            json_data = json.dumps(test_data) + '\n'
            sock.sendall(json_data.encode())
            print(f"📤 테스트 데이터 전송 {i+1}/5")
            time.sleep(1)
        
        sock.close()
        print("✅ 테스트 완료 - sync_data_recorder에서 VR 데이터를 확인하세요")
        
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        print("\n💡 확인사항:")
        print("1. sync_data_recorder.py가 실행 중인가?")
        print("2. 포트 12346이 열려 있는가?")
        print("   netstat -an | grep 12346")

if __name__ == '__main__':
    test_send_vr_data()