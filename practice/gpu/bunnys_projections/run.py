"""Portable launcher: python run.py [serve|benchmark|doctor] [options]."""
import os,subprocess,sys
from pathlib import Path
from runtime_paths import environment_python
root=Path(__file__).resolve().parent
args=sys.argv[1:];env=os.environ.copy()
if '--cpu' in args:
    env['RT616_CPU_ONLY']='1';args.remove('--cpu')
command=args.pop(0) if args and args[0] in ['serve','benchmark','doctor'] else 'serve'
target={'serve':'serve_demo.py','benchmark':'benchmark.py','doctor':'doctor.py'}[command]
python=environment_python()
if not python.exists():raise SystemExit('Run setup.py with Python 3.12 first.')
try:
    result=subprocess.run([str(python),str(root/target),*args],cwd=root,env=env)
    raise SystemExit(result.returncode)
except KeyboardInterrupt:
    raise SystemExit(130)
