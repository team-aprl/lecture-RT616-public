# 포인트클라우드 CPU/GPU 실습

**Docker 없이 실행합니다.** NVIDIA GPU 노트북, 최신 NVIDIA 드라이버, **Python 3.12 64-bit**가 필요합니다. 설치 스크립트가 프로젝트 전용 가상환경에 Python 패키지와 CUDA 라이브러리를 설치합니다.

[Colab에서 바로 실행](https://colab.research.google.com/github/team-aprl/lecture-RT616-public/blob/main/practice/gpu/bunnys_projections/PointCloud_CPU_GPU.ipynb) · [상세 문서 / 문제 해결](README.md)

## Windows PowerShell

```powershell
git clone https://github.com/team-aprl/lecture-RT616-public.git
cd lecture-RT616-public/practice/gpu/bunnys_projections
nvidia-smi
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
```

설치가 끝나면 **http://127.0.0.1:8766/** 를 엽니다. 다음부터는 run.ps1만 실행합니다. 종료는 터미널에서 Ctrl+C.

## Linux

Python 3.12와 venv 지원을 준비한 뒤 같은 폴더에서:

```bash
bash setup.sh
bash run.sh
```

## 수업에서 해볼 것

1. Bunny, Grid 8로 시작해 경로 재생. 드래그로 각도, 휠로 거리 조절.
2. Dragon / Happy Buddha / Drill도 선택해 보기.
3. Grid 32 또는 64로 늘리기. 64는 모델 4,096개입니다.
4. **K=60, 비교 Both → K 프레임 렌더링**.
5. 프레임별 CPU/GPU ms와 마지막 총시간·평균·중앙값·p95·비율 비교.
6. 결과 JSON 저장. 배터리/전원 연결, 포인트 수, 카메라 경로를 바꾸어 비교.

**CPU는 Numba 단일 스레드, GPU는 CUDA**입니다. K 프레임은 이미지 전송 없이 서버 안에서 계산합니다. GPU 측정은 계산뿐 아니라 pose 업로드와 깊이 영상 CPU 회수까지 포함합니다. 화면 FPS와 혼동하지 마세요.

NVIDIA GPU가 없으면 setup.ps1/run.ps1에 **-Cpu**, Linux에서는 **--cpu**를 붙이세요. AMD·Intel·Apple GPU에서 CUDA를 사용하는 실습은 아닙니다.

GPU 진단: Windows에서는 **py -3.12 run.py doctor --require-gpu**. Linux GPU 및 hosted Colab 직접 실행 검증 범위는 [VALIDATION.md](VALIDATION.md)를 확인하세요.


Windows에서는 긴 경로 오류를 피하려고 %LOCALAPPDATA%/rt616-envs 아래에 가상환경을 만듭니다. Linux는 이 폴더의 .venv를 사용하며, 실행 스크립트가 자동으로 찾습니다.
