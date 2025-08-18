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
