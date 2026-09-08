"""Create a folder-local environment. Run with Python 3.12; this is not a package installer."""
import argparse,os,subprocess,sys,venv
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--cpu',action='store_true',help='Install the CPU-only dependencies')
args=parser.parse_args()
if sys.version_info[:2] != (3,12):
    raise SystemExit('Use Python 3.12 for this tested environment (Windows: py -3.12 setup.py).')
root=Path(__file__).resolve().parent
from runtime_paths import environment_dir,environment_python
folder=environment_dir()
if not folder.exists():venv.EnvBuilder(with_pip=True).create(folder)
python=environment_python()
print('Environment:',folder,flush=True)
subprocess.run([str(python),'-m','pip','install','--upgrade','pip'],check=True)
subprocess.run([str(python),'-m','pip','install','-r',str(root/('requirements-cpu.txt' if args.cpu else 'requirements.txt'))],check=True)
env=os.environ.copy()
if args.cpu:env['RT616_CPU_ONLY']='1'
subprocess.run([str(python),str(root/'doctor.py'),*(['--cpu'] if args.cpu else ['--require-gpu'])],check=True,env=env)
print('Ready. Windows: .\\run.ps1 | Linux: bash run.sh')
