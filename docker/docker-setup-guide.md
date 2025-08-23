# Quest2ROS Docker Setup Guide
metaquest2 quest2ros앱이 ubuntu 20.04 ros1 noetic에서만 호환되므로
host 환경이 ubuntu 22.04 ros2 humble이어서 도커 컨테이너로 컨트롤러 연결 및 pose값 받아옴.
이 컨테이너에서 metaquest2 연결 및 브릿지.py 실행함.

### 1. 사전 준비
```bash
# Docker와 docker-compose 설치 확인
docker --version
docker-compose --version

# X11 권한 설정 (GUI 프로그램 실행용)
xhost +local:docker
```

### 2. 컨테이너 실행 방법

#### Option A: docker-compose 사용 (추천)
```bash
# 프로젝트 폴더에서
docker-compose up -d quest2ros

# 컨테이너 접속
docker-compose exec quest2ros bash
```

#### Option B: 직접 Docker 명령 사용
```bash
# 이미지 다운로드
docker pull mjo035/quest2ros_fresh:latest

# 컨테이너 실행
docker run -it --name quest2ros_fresh \
  --privileged \
  --network host \
  -e DISPLAY=$DISPLAY \
  -e ROS_MASTER_URI=http://localhost:11311 \
  -e ROS_HOSTNAME=localhost \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace \
  -v /dev:/dev \
  mjo035/quest2ros_fresh:latest
```

### 3. 컨테이너 내부에서 사용법
```bash
# ROS 환경 설정 확인
echo $ROS_MASTER_URI
source /opt/ros/noetic/setup.bash

# 프로젝트 파일들은 /workspace에 마운트됨
cd /workspace
ls -la

# quest2ros 패키지 사용
# (기존 사용법과 동일)
```

### 4. 컨테이너 중지/재시작
```bash
# 중지
docker-compose down

# 재시작
docker-compose up -d quest2ros

# 로그 확인
docker-compose logs quest2ros
```

### 5. 문제 해결
- GUI 프로그램이 실행되지 않는 경우: `xhost +local:docker` 실행
- 권한 문제: `--privileged` 플래그가 포함되어 있는지 확인
- 네트워크 문제: `--network host` 사용으로 호스트 네트워크 공유

### 6. 주의사항
- 이 컨테이너는 ROS Noetic이 설치된 Ubuntu 20.04 환경입니다
- VR 헤드셋 연결 시 USB 장치 권한이 필요할 수 있습니다
