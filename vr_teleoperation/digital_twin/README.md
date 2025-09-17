# 🤖 Digital Twin - 양팔 로봇 VR 텔레오퍼레이션

MuJoCo 시뮬레이션과 실물 로봇을 완벽하게 동기화하여 안전하고 정교한 텔레오퍼레이션을 구현합니다.

## 🎯 핵심 개념

**Digital Twin**: MuJoCo가 실물 로봇의 디지털 쌍둥이 역할
- VR 입력 → MuJoCo 검증 → 실물 로봇 제어
- 충돌 감지, 한계 체크를 MuJoCo에서 수행
- 안전이 검증된 명령만 실물로 전송

## 📁 파일 구조

### Phase 1: 오프셋 측정
- `right_arm_teaching.py` - 오른팔 오프셋 측정 (모터 ID 21-25)
- `mujoco_right_visualizer.py` - 오른팔 MuJoCo 시각화

### Phase 2: 양팔 동기화
- `dual_arm_teaching.py` - 양팔 동시 teaching
- `mujoco_dual_visualizer.py` - 양팔 MuJoCo 시각화

### Phase 3: 통합 시스템
- `mujoco_dual_sync.py` - VR 입력 + 실물 제어 통합
- `offsets.json` - 측정된 오프셋 값 저장

## 🚀 사용 방법

### 1단계: 오른팔 오프셋 측정

```bash
# Terminal 1: Teaching 실행
cd digital_twin
python3 right_arm_teaching.py

# Terminal 2: MuJoCo 시각화
python3 mujoco_right_visualizer.py

# 오른팔을 홈 포지션으로 이동 후 'C' 키로 캘리브레이션
```

### 2단계: 양팔 동기화 확인

```bash
# 하드웨어 연결
ros2 launch your_package dual_arm_hardware.launch.py

# Teaching 실행
python3 dual_arm_teaching.py

# MuJoCo 시각화
python3 mujoco_dual_visualizer.py
```

### 3단계: VR 텔레오퍼레이션

```bash
# 1. 하드웨어 연결 (호스트)
ros2 launch your_package dual_arm_hardware.launch.py

# 2. Digital Twin MuJoCo (호스트)
python3 mujoco_dual_sync.py

# 3. VR Bridge (도커)
docker exec -it vr_container python3 dual_arm_bridge_improved.py

# 4. Meta Quest 2 켜기
```

## ⚙️ 오프셋 값

측정된 오프셋 값 (예시):
```json
{
  "left_arm": [0.0, -0.43, 1.94, -0.42],
  "right_arm": [측정 필요]
}
```

## 🛡️ 안전 기능

- **충돌 감지**: MuJoCo에서 충돌 시 실물 정지
- **속도 제한**: 급격한 움직임 방지
- **조인트 한계**: 안전 범위 내에서만 동작
- **비상 정지**: Ctrl+C로 즉시 정지

## 📊 시스템 아키텍처

```
VR Input → Docker Bridge → MuJoCo (Host) → Hardware
              ↓                ↑
         Socket(12345)    Sync & Offset
                          Calibration
```

## 🔧 필요 사항

- Dynamixel SDK
- MuJoCo
- ROS2 Humble
- Meta Quest 2 + quest2ros

## 📝 주의사항

1. 왼팔 모터 ID: 11-15
2. 오른팔 모터 ID: 21-25
3. 포트: `/dev/ttyACM0` 또는 `/dev/ttyUSB0`
4. 오프셋은 로봇별로 다를 수 있음

## 🎓 개발자

- CookingBot 2025 프로젝트
- Digital Twin 기반 안전한 텔레오퍼레이션