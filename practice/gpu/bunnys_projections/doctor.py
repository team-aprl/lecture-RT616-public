"""Device/library report plus a real, tiny JIT-compiled projection test."""
import argparse,json,os,sys,platform
from importlib.metadata import version
parser=argparse.ArgumentParser()
parser.add_argument('--cpu',action='store_true')
parser.add_argument('--require-gpu',action='store_true')
args=parser.parse_args()
if args.cpu:os.environ['RT616_CPU_ONLY']='1'
from bunny_depth_lab import GPU_AVAILABLE,GPU_ERROR,cp,np
from smoke_test import check_renderers
report={'python':platform.python_version(),'os':platform.system(),
        'packages':{x:version(x) for x in ['numpy','numba','pillow','matplotlib','plyfile']},'gpu_available':GPU_AVAILABLE}
if GPU_AVAILABLE:
    props=cp.cuda.runtime.getDeviceProperties(0)
    name=props['name'];report['gpu']=name.decode() if isinstance(name,bytes) else name
    report['cuda_runtime']=cp.cuda.runtime.runtimeGetVersion()
    report['cuda_driver']=cp.cuda.runtime.driverGetVersion()
    report['cupy']=cp.__version__
else:report['gpu_error']=GPU_ERROR
print(json.dumps(report,indent=2))
if args.require_gpu and not GPU_AVAILABLE:
    raise SystemExit('GPU unavailable. Check nvidia-smi and the NVIDIA driver, or use setup.py --cpu.')
check_renderers()
print('PASS: CPU projection'+(' and actual CUDA kernel / depth equality' if GPU_AVAILABLE else ' (CPU-only)'))
