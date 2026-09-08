from pathlib import Path
import nbformat
out=Path(__file__).resolve().parent
n=nbformat.v4.new_notebook()
n.metadata={'accelerator':'GPU','kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'colab':{'name':'PointCloud_CPU_GPU.ipynb'}}
md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
n.cells=[md("""# Point Cloud · CPU vs GPU 실습
**런타임 → 런타임 유형 변경 → GPU** 선택 후 아래 **① 준비 → ② 뷰어** 두 셀을 실행하세요.

- Bunny / Dragon / Happy Buddha / Drill을 최대 **64×64**로 배열합니다.
- 공전·접근/후퇴·나선·내부 비행·호버링: 경로 선택 후 **경로 재생**.
- 화면 드래그=각도, 휠=거리.
- **K 프레임 렌더링**: K 입력, 비교 Both 선택. 같은 K개 카메라 자세를 CPU/GPU로 계산하고 프레임별 ms·총시간·평균·중앙값·p95를 출력합니다.
- 이미지 전송이 느려도 K 프레임 측정은 서버 안에서 진행합니다. GPU 측정은 pose 업로드·계산·depth CPU 회수를 포함하며, 이미지 변환·전송·표시는 제외합니다.
""")]
setup="""#@title ① 준비 — 패키지와 실습 파일
import importlib.util, subprocess, sys
packages={'numpy':'numpy','numba':'numba','PIL':'pillow','matplotlib':'matplotlib','plyfile':'plyfile','cupy':'cupy-cuda12x[ctk]'}
missing=[pkg for module,pkg in packages.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','-q','install',*missing])
from pathlib import Path
"""
for name in ['bunny_depth_lab.py','stanford_scene.py','serve_demo.py','colab_live.py','benchmark_lab.py']:
 setup+=f"Path({name!r}).write_text({(out/name).read_text(encoding='utf-8')!r},encoding='utf-8')\n"
setup+="print('준비 완료. ② 뷰어 셀을 실행하세요.')"
c=code(setup);c.metadata={'cellView':'form'};n.cells.append(c)
c=code("""#@title ② 라이브 뷰어 + K 프레임 비교
import colab_live, importlib
colab_live.stop()  # 이 노트북이 띄운 이전 서버만 종료
importlib.reload(colab_live)
live_process = colab_live.launch()
""");c.metadata={'tags':['interactive']};n.cells.append(c)
n.cells.append(md("""## 선택: 뷰어 없이 셀에 프레임별 시간 출력
아래 **③**은 같은 K 프레임 실험을 텍스트 로그로만 실행합니다. 뷰어의 자동 재생을 정지한 뒤 실행하세요.
K·GRID·MODEL·CAMERA_PATH를 바꿀 수 있습니다. 최초 준비·3회 warmup은 측정에서 제외됩니다.
"""))
n.cells.append(code("""#@title ③ 선택 — 콘솔에서 K 프레임 비교
from stanford_scene import Scene
from benchmark_lab import benchmark_frames
K, GRID, MODEL, CAMERA_PATH = 10, 8, 'Bunny', 'Hover'
scene = Scene(MODEL, GRID, 36000)
results = list(benchmark_frames(scene, k=K, path=CAMERA_PATH, mode='Both'))
for backend, stats in results[-1]['stats'].items():
    print(backend, {key:round(value,3) for key,value in stats.items()})
import json
Path('k_frames_result.json').write_text(json.dumps(results,indent=2))
print('Saved k_frames_result.json')
"""))
n.cells.append(md("""**해석:** CPU는 Numba 단일 스레드, GPU는 CUDA입니다. 모델당 샘플 상한과 실제 처리 점 수를 화면에서 확인하세요. 점과 복제 위치를 따로 저장하지만 모든 복제 점을 투영합니다.
실제 프레임 처리 시간은 장치·장면·카메라 위치에 따라 달라집니다.
화면 재표시: colab_live.display_live() / 서버 종료: colab_live.stop().

출처: [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/) · [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
"""))
nbformat.write(n,out/'PointCloud_CPU_GPU.ipynb')
print('Dedicated notebook: 6 cells, 3 code cells')

