# 🤖 왼팔 역방향 데이터 수집 시스템

## 📌 목적
실물 로봇을 손으로 조작하여 양질의 VR-Joint 매핑 데이터를 수집합니다.

## 🔄 데이터 수집 흐름
```
실물 로봇(손 조작) → MuJoCo(시각화) → VR 컨트롤러(매칭) → 데이터 저장
     ↓                    ↓                 ↓              ↓
  Joint 값           실시간 동기      Pose 측정      mapping.json
```

## 📁 파일 구조
```
test1/
├── README.md                    # 이 가이드
├── left_arm_teaching.py         # 실물 로봇 티칭 (토크 OFF)
├── mujoco_left_visualizer.py    # MuJoCo 왼팔만 시각화
├── vr_pose_collector.py         # VR 컨트롤러 포즈 수집 (Docker)
└── sync_data_recorder.py        # 통합 데이터 기록
```

## 🚀 실행 방법

### 사전 준비
1. **하드웨어 체크**
   ```bash
   # USB 포트 확인
   ls /dev/ttyUSB* /dev/ttyACM*
   
   # 권한 설정
   sudo chmod 666 /dev/ttyACM0
   ```

2. **Meta Quest 2 준비**
   - 왼쪽 컨트롤러만 사용
   - quest2ros Docker 실행 중

### Step 1: 실물 로봇 티칭 모드
```bash
# Terminal 1 (Host)
cd ~/CookingBot_2025/vr_teleoperation/test1
python3 left_arm_teaching.py

# 출력 예시:
# ✅ Dynamixel 연결 성공
# 🔓 모터 11,12,13,14,15 토크 OFF
# ✋ 왼팔을 손으로 움직일 수 있습니다
```

### Step 2: MuJoCo 시각화
```bash
# Terminal 2 (Host)
python3 mujoco_left_visualizer.py

# 실물 로봇 움직이면 MuJoCo가 실시간 따라옴
```

### Step 3: VR 포즈 수집
```bash
# Terminal 3 (Docker)
docker exec -it quest2ros bash
cd /workspace/test1
python3 vr_pose_collector.py

# VR 왼쪽 컨트롤러 포즈 실시간 출력
```

### Step 4: 데이터 동기 기록
```bash
# Terminal 4 (Host)
python3 sync_data_recorder.py

# 조작법:
# Space: 현재 상태 저장
# S: 파일로 내보내기
# Q: 종료
```

## 📊 수집할 핵심 자세 (8개)

### 기본 극단 위치
1. **최대 전방** - 팔 최대한 앞으로
2. **최대 후방** - 팔 최대한 뒤로
3. **최대 상향** - 팔 위로
4. **최대 하향** - 팔 아래로
5. **최대 좌측** - 팔 왼쪽으로
6. **최대 우측** - 팔 오른쪽으로
7. **중립 위치** - 기본 자세
8. **작업 위치** - 테이블 높이

### 추가 권장 자세 (X축 강화)
9. **전방 낮게** - 테이블 위 물건 집기
10. **전방 높게** - 선반 물건 집기
11. **중간 전방** - 요리 자세
12. **당기기** - 물건 당기는 자세
13. **밀기** - 물건 미는 자세

## 💾 데이터 형식

### 저장되는 데이터 (mapping_YYYYMMDD_HHMMSS.json)
```json
{
  "timestamp": "2024-12-12T10:30:00",
  "sample_id": 1,
  "robot_joints": [0.0, 0.5, -0.4, 0.0],  // Joint1~4
  "robot_gripper": -0.01,
  "vr_position": [0.2, 0.0, 0.0],         // X, Y, Z
  "vr_orientation": [0.0, 0.0, 0.0, 1.0], // Quaternion
  "vr_trigger": 0.0
}
```

## ⚠️ 안전 주의사항

### 실행 전 체크리스트
- [ ] 로봇 주변 공간 확보 (반경 1m)
- [ ] 비상정지 버튼 위치 확인
- [ ] USB 케이블 연결 상태
- [ ] 전원 12V 확인
- [ ] Dynamixel Wizard 대기

### 비상시 대처
1. **토크가 켜진 경우**: 즉시 전원 OFF
2. **통신 끊김**: USB 재연결
3. **이상 동작**: 비상정지 → 재부팅

## 📈 데이터 품질 체크

### 좋은 데이터
- ✅ 실제 작업 자세
- ✅ 극단 위치 포함
- ✅ 부드러운 전환
- ✅ 명확한 X축 변화

### 피해야 할 데이터
- ❌ 불가능한 자세
- ❌ 떨리는 값
- ❌ 중복된 위치
- ❌ 충돌 위험 자세

## 🔍 트러블슈팅

### 문제: 토크가 안 꺼짐
```bash
# Dynamixel Wizard로 수동 제어
# 또는 모터 ID 확인 (11-15 맞는지)
```

### 문제: MuJoCo 동기 안됨
```bash
# 포트 12345 확인
netstat -an | grep 12345
```

### 문제: VR 데이터 안 옴
```bash
# Docker 내부에서
rostopic echo /q2r_left_hand_pose
```

## 📝 수집 로그

### 세션 1 (예시)
- 날짜: 
- 수집 개수: 0/50
- 진행 상황:
  - [ ] 극단 8개
  - [ ] X축 강화 5개
  - [ ] 작업 자세 10개