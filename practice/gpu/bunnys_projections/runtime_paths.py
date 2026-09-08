"""Per-project environment; keep Windows CUDA header paths below MAX_PATH."""
import hashlib,os
from pathlib import Path
def environment_dir():
    root=Path(__file__).resolve().parent
    if os.name=='nt':
        key=hashlib.sha256(str(root).lower().encode()).hexdigest()[:10]
        return Path(os.environ['LOCALAPPDATA'])/'rt616-envs'/key
    return root/'.venv'
def environment_python():
    return environment_dir()/('Scripts/python.exe' if os.name=='nt' else 'bin/python')

