<br>

<h1 align="center">
  2026 부산대학교 TECH WEEK:<br>
  Autonomous Mobile Robot의 Search & Rescue Mission
</h1>
<br>

---

<br><br>

![](부산대_TECHWEEK_Physical_AI.png)

## 구현된 해커톤 베이스라인

`worlds/apartment.wbt`는 이제 `controllers/tb3_rescue/tb3_rescue.py`를 실행합니다.
이 컨트롤러는 사전 지도·GPS·Supervisor·사과 좌표를 사용하지 않고 다음 과정을
하나의 자율 임무로 수행합니다.

- 엔코더 오도메트리와 LiDAR 스캔 정합 기반 위치 추정
- LiDAR log-odds occupancy grid mapping
- 실제 이동 거리·방문 이력·카메라 관측 범위를 고려한 frontier exploration 및 A* global planning
- 카메라/LiDAR 사람 추적, 이동 예측 및 실시간 장애물 회피
- 엔코더와 독립적인 LiDAR 이동량 비교로 카펫 걸림·헛바퀴 감지 및 제한된 후진·회전 복구
- YOLO11n `apple` 클래스 + 빨간색·형태 검증; 색상 후보는 재관측용으로만 사용
- 카메라 Display의 YOLO 박스 및 색상 후보 박스 표시
- 서로 다른 빨간 사과 2개의 근접 관측을 확인한 후 시작점 복귀 및 정지

실행 방법과 지도 색상, 결과 파일은
[`controllers/tb3_rescue/README.md`](controllers/tb3_rescue/README.md)를 참고하세요.
