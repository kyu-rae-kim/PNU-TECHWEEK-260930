<div align="center">

# PNU TECH WEEK 2026 — Physical AI

### Autonomous Mobile Robot Search & Rescue Mission

Webots 환경에서 TurtleBot3가 아파트를 스스로 탐색하고,<br>
**빨간 사과 2개를 찾아 접근한 뒤 출발점으로 복귀**하는 자율주행 프로젝트입니다.

![Webots 기반 Physical AI 프로젝트](./부산대_TECHWEEK_Physical_AI.png)

</div>

## 프로젝트 소개

부산대학교 TECH WEEK Physical AI 실습 및 해커톤을 위해 제작한 프로젝트입니다.
메인 컨트롤러는 사전 지도, GPS, Supervisor, 사과의 월드 좌표 없이 카메라,
2D LiDAR, 휠 엔코더만으로 임무를 수행합니다.

```text
센서 입력 → 위치 추정·지도 작성 → 목표 탐색·경로 계획 → 안전 제어
                ↑                                      ↓
                └──────────── 재계획·복구 ─────────────┘
```

### 주요 기능

- 엔코더 오도메트리와 LiDAR scan matching을 결합한 위치 추정
- log-odds occupancy grid mapping 및 frontier exploration
- A* 기반 전역 경로 계획과 실시간 장애물 회피
- YOLO11n과 색상·형태 검증을 이용한 빨간 사과 식별
- 카메라와 LiDAR를 결합한 보행자 추적 및 안전거리 유지
- 낮은 물체 감지, 카펫 걸림·헛바퀴 판단 및 제한적 자율 복구
- 서로 다른 사과 2개 확인 후 출발점 복귀 및 임무 결과 저장

## 동작 흐름

1. 시작 위치에서 주변을 관측하고 로컬 좌표계의 지도를 생성합니다.
2. 아직 확인하지 않은 영역을 선택해 안전한 경로를 계획합니다.
3. 빨간 사과 후보를 발견하면 접근하여 YOLO로 다시 확인합니다.
4. 서로 다른 사과 2개를 근거리에서 확인하면 출발점으로 복귀합니다.
5. 복귀 오차가 기준 이내이면 정지하고 `COMPLETE` 상태를 기록합니다.

## 기술 스택

| 구분 | 사용 기술 |
| --- | --- |
| Simulation | Webots R2025a |
| Robot | TurtleBot3 Burger |
| Language | Python |
| Perception | Ultralytics YOLO11n, RGB Camera, 2D LiDAR |
| Navigation | Occupancy Grid, Frontier Exploration, A* |
| Libraries | NumPy, Ultralytics |

## 프로젝트 구조

```text
.
├─ controllers/
│  ├─ tb3_rescue/          # 메인 자율 탐색·복귀 컨트롤러와 회귀 테스트
│  ├─ tb3_teleop*/         # 키보드 조작 및 센서·YOLO 예제
│  ├─ tb3_lidar/           # LiDAR 확인 예제
│  ├─ tb3_cam/             # 카메라 확인 예제
│  └─ tb3_segmentation/    # 영상 분할 예제
├─ models/YOLO/            # YOLO 모델 가중치
├─ protos/                 # 색상별 사과 PROTO
├─ worlds/                 # Webots 실습 및 미션 월드
├─ tests/                  # 실습용 테스트 리소스
├─ TECH-WEEK-26_Physical-AI.ipynb
└─ requirements.txt
```

## 시작하기

### 1. 준비물

- [Webots R2025a](https://cyberbotics.com/)
- Python 3.10 이상 권장
- [`uv`](https://docs.astral.sh/uv/) 또는 `pip`

### 2. 저장소와 Python 환경 준비

```powershell
git clone https://github.com/syeoniism/PNU-TECHWEEK-260930.git
cd PNU-TECHWEEK-260930

uv venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

`pip`을 사용하는 경우 다음과 같이 설치할 수 있습니다.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Webots Python 경로 설정

[`controllers/tb3_rescue/runtime.ini`](./controllers/tb3_rescue/runtime.ini)의
`COMMAND`를 현재 저장소의 가상환경 Python 절대 경로로 수정합니다.

```ini
[python]
COMMAND = D:/path/to/PNU-TECHWEEK-260930/.venv/Scripts/python.exe
OPTIONS = -u
```

### 4. 미션 실행

1. Webots에서 [`worlds/apartment.wbt`](./worlds/apartment.wbt)를 엽니다.
2. 시뮬레이션을 **Reset**합니다.
3. **Run**을 눌러 자율 임무를 시작합니다.

> 실행 중인 Webots 컨트롤러에는 Python 코드 변경이 자동 반영되지 않습니다.
> 코드를 수정한 뒤에는 월드를 다시 불러오거나 시뮬레이션을 재시작하세요.

## 테스트

Webots를 실행하지 않고도 SLAM, 사과 판별, 안전 제어, 걸림 복구 등의 핵심
로직을 회귀 테스트로 확인할 수 있습니다.

```powershell
.venv\Scripts\python.exe -m unittest discover -s controllers/tb3_rescue -v
```

## 결과 확인

실행 결과는 기본적으로 `controllers/tb3_rescue/mission_output/`에 저장됩니다.

| 파일 | 내용 |
| --- | --- |
| `mission_result.json` | 상태, 발견·방문 수, 추정 위치, 복귀 오차, 정지 사유 |
| `occupancy_map.pgm` | 주행 중 생성한 occupancy grid map |

정상 완료 여부는 `mission_result.json`의 다음 값을 함께 확인합니다.

```json
{
  "state": "COMPLETE",
  "success": true,
  "red_apples_visited": 2
}
```

상세 알고리즘, 안전 조건, 지도 표시 방식과 검증 기록은
[`controllers/tb3_rescue/README.md`](./controllers/tb3_rescue/README.md)에 정리되어 있습니다.

## 한계

- 일반 COCO 모델을 사용하므로 장면에 따라 사과를 놓치거나 유사 물체를 잘못
  인식할 수 있습니다.
- 전역 loop closure가 없어 장시간 주행 시 위치 오차가 누적될 수 있습니다.
- 2D LiDAR의 높이 밖에 있는 장애물이나 급격하게 움직이는 보행자에 대한 회피를
  완전히 보장하지 않습니다.
- 전체 임무 성공 여부는 Webots 실행 로그와 결과 JSON으로 별도 확인해야 합니다.

## 관련 자료

- [`TECH-WEEK-26_Physical-AI.ipynb`](./TECH-WEEK-26_Physical-AI.ipynb) — 실습 노트북
- [`부산대 TECH WEEK Physical AI.pdf`](./부산대%20TECH%20WEEK%20Physical%20AI.pdf) — 강의 자료
- [`controllers/tb3_rescue/README.md`](./controllers/tb3_rescue/README.md) — 메인 컨트롤러 상세 문서
