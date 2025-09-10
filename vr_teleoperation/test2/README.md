# 🚀 데이터 수집 실행 가이드

## 📊 시스템 구조
```
실물 로봇 → teaching.py → MuJoCo → visualizer_fixed.py → recorder
                                                          ↑
                           Docker → vr_pose_collector.py ↑
```

## 🔌 포트 구성
- **12345**: teaching.py → mujoco_left_visualizer.py (기존 유지)
- **12346**: vr_pose_collector.py → sync_data_recorder.py (기존 유지) 
- **12347**: mujoco_left_visualizer_fixed.py → sync_data_recorder_port_fixed.py (새로 추가)

## ✅ 실행 순서

### Terminal 1: MuJoCo 시각화 (수정된 버전)
```bash
cd ~/CookingBot_2025/vr_teleoperation/test1
python3 mujoco_left_visualizer_fixed.py
```
- 포트 12345에서 teaching 데이터 수신
- 포트 12347로 MuJoCo 조인트값 전송

### Terminal 2: 데이터 레코더 (수정된 버전)
```bash
cd ~/CookingBot_2025/vr_teleoperation/test1
python3 sync_data_recorder_port_fixed.py
```
- 포트 12347에서 MuJoCo 조인트값 수신
- 포트 12346에서 VR 포즈값 수신

### Terminal 3: 실물 로봇 티칭 (기존 파일 그대로)
```bash
cd ~/CookingBot_2025/vr_teleoperation/test1
python3 left_arm_teaching.py
```
- 포트 12345로 MuJoCo에 전송

### Terminal 4: VR 포즈 수집 (Docker에서 실행, 기존 파일 그대로)
```bash
docker exec -it quest2ros bash
cd /workspace/test1
python3 vr_pose_collector.py
```
- 포트 12346으로 Host에 전송

## 📝 데이터 수집 프로세스

1. **모든 프로그램 실행 확인**
   - MuJoCo 화면에 로봇 표시
   - Recorder에 "MuJoCo✓ VR✓" 표시

2. **실물 로봇 조작**
   - 손으로 원하는 자세로 이동
   - MuJoCo에서 실시간 확인

3. **VR 컨트롤러 매칭**
   - MuJoCo와 동일한 자세로 VR 컨트롤러 위치

4. **데이터 저장**
   - Space 키로 현재 상태 저장
   - S 키로 파일 내보내기

## 🔍 확인 사항

### 정상 동작 확인
- visualizer_fixed.py: "Recorder✓" 표시
- recorder_port_fixed.py: "[M✓ V✓]" 표시
- 저장된 데이터에 MuJoCo 조인트값과 VR 포즈값 포함

### 트러블슈팅
```bash
# 포트 사용 확인
lsof -i :12345  # teaching → mujoco
lsof -i :12346  # vr_pose → recorder  
lsof -i :12347  # mujoco → recorder

# 프로세스 종료
killall python3
```

## 💾 출력 파일
- `mujoco_vr_mapping_YYYYMMDD_HHMMSS.json`
- MuJoCo 조인트값 + VR 포즈값 매핑 데이터