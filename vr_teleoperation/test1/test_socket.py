#!/usr/bin/env python3
"""
소켓 연결 테스트
"""

import socket
import json
import time

def test_as_client():
    """클라이언트로 테스트"""
    print("🔍 클라이언트 테스트...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 12345))
        print("✅ 서버 연결 성공!")
        
        # 테스트 데이터 전송
        test_data = {
            'left_arm': {
                'joint_angles': [0.1, 0.2, 0.3, 0.4],
                'gripper': -0.01
            }
        }
        sock.sendall((json.dumps(test_data) + '\n').encode())
        print("📤 테스트 데이터 전송 완료")
        
        time.sleep(1)
        sock.close()
        
    except Exception as e:
        print(f"❌ 연결 실패: {e}")

def test_as_server():
    """서버로 테스트"""
    print("🔍 서버 테스트...")
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('localhost', 12345))
        server.listen(1)
        print("📡 포트 12345에서 대기 중...")
        
        client, addr = server.accept()
        print(f"✅ 클라이언트 연결됨: {addr}")
        
        data = client.recv(1024).decode()
        print(f"📥 받은 데이터: {data[:100]}...")
        
        client.close()
        server.close()
        
    except Exception as e:
        print(f"❌ 서버 오류: {e}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        test_as_server()
    else:
        test_as_client()