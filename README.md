# 2025 한이음 공모전 CookingBot

## Meta Quest 2 VR → OpenManipulator-X Teleoperation

![VR Teleoperation Demo](assets/IMG_1756.gif)

*Meta Quest 2 VR 컨트롤러로 OpenManipulator-X 로봇을 실시간 제어하는 모습*

---

## 프로젝트 개요

본 프로젝트는 Meta Quest 2 VR 헤드셋을 활용하여 OpenManipulator-X 로봇을 직관적으로 제어하는 텔레오퍼레이션 시스템을 개발했습니다. 

사용자가 VR 공간에서 자연스럽게 손을 움직이면, 실제 로봇이 동일한 동작을 수행합니다. 

기존의 복잡한 역기구학(IK) 해법 대신 혁신적인 **Offset-based Control** 방식을 도입하여, 더 안전하고 직관적인 로봇 제어를 구현했습니다.

궁극적인 목표는 듀얼암 로봇 시스템으로 확장하여 샌드위치 제작과 같은 협업 요리 작업을 수행하는 것입니다.

## 시스템 아키텍처

```
Meta Quest 2 (VR) → Docker (ROS1 + quest2ros) → Host (ROS2) → Physical Robot
                                  ↓
                             MuJoCo Simulation (verification)
```

---

## Dual Arm 제어 성공

![Dual Arm Demo](assets/IMG_1862.gif)

양팔 로봇 동시 제어 구현을 성공했고 wifi 도시락을 사용해서 장소 제한 없이 구현 가능하도록 개선했습니다.

---

## 시작하기

### 필수 환경
- Ubuntu 22.04
- ROS2 Humble
- MuJoCo 2.3+
- Docker (quest2ros)
- Meta Quest 2 + quest2ros 앱

### 프로젝트 구조

```
CookingBot_2025/
├── vr_teleoperation/      # VR 텔레오퍼레이션 시스템
│   ├── single_arm/        # 싱글암 제어 코드
│   └── dual_arm/          # 듀얼암 제어 코드
├── open_manipulator/      # ROS2 하드웨어 설정
├── data/                  # 데이터 수집 도구
├── docker/                # Docker 설정
└── examples/              # 예제 코드
```

### 빠른 시작

1. **하드웨어 설정**
   - OpenManipulator-X 연결 및 설정은 [하드웨어 가이드](open_manipulator/HARDWARE_SETUP_GUIDE.md) 참조

2. **VR 시스템 실행**
   - Single Arm: [싱글암 가이드](vr_teleoperation/single_arm) 참조
   - Dual Arm: [듀얼암 가이드](vr_teleoperation/dual_arm) 참조
   - 자세한 사용법은 [VR 시스템 문서](vr_teleoperation/README.md)

3. **데이터 수집**
   - VR-로봇 매핑 데이터 수집: [데이터 수집 가이드](data/README.md) 참조

