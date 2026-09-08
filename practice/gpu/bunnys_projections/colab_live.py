"""Launch the actual CPU/CUDA depth server inside a Colab output frame."""
import os,sys,subprocess,time,urllib.request
from pathlib import Path
_process=None
def launch(port=8766,show=True):
    global _process
    if _process is not None and _process.poll() is None:
        if show:display_live(port)
        return _process
    # Never terminate a process that this launcher did not create.
    import socket
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',port))==0:
            raise RuntimeError(f'Port {port} is already in use. Use launch(port={port+1}).')
    folder=Path(__file__).resolve().parent
    env=os.environ.copy();env['RT616_DEMO_PORT']=str(port)
    with (folder/'live_server.log').open('w',encoding='utf-8') as log:
        _process=subprocess.Popen([sys.executable,'-u',str(folder/'serve_demo.py')],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
    for _ in range(240):
        if _process.poll() is not None:raise RuntimeError((folder/'live_server.log').read_text(encoding='utf-8'))
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/',timeout=1) as r:
                if r.status==200:break
        except Exception:time.sleep(.5)
    else:
        stop();raise RuntimeError('Server startup timed out; inspect live_server.log and rerun.')
    if show:display_live(port)
    return _process
def display_live(port=8766):
    try:
        from google.colab import output
    except ImportError:
        from IPython.display import IFrame,display
        display(IFrame(f'http://127.0.0.1:{port}/?transport=png',width='100%',height=1000))
    else:
        output.serve_kernel_port_as_iframe(port,path='/?transport=png',height=1000,cache_in_notebook=False)
    print('화면의 경로 재생을 누르세요. 드래그=각도, 휠=접근/후퇴. 정지 버튼으로 일시정지.')
def stop():
    global _process
    if _process is not None and _process.poll() is None:
        _process.terminate()
        try:_process.wait(timeout=10)
        except subprocess.TimeoutExpired:_process.kill();_process.wait()
    _process=None

