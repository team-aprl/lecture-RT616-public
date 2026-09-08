"""RT616: Stanford Bunny camera-depth rasterization, CPU and CUDA.
Data: https://graphics.stanford.edu/data/3Dscanrep/ (bunny/reconstruction/bun_zipper.ply)
One point -> one pixel; optical-axis z in scene units; not ray tracing or LiDAR range.
CPU: compiled Numba single-thread fused loop. GPU: fused CuPy RawKernel.
"""
import os, sys, json, time, tarfile, urllib.request, platform, hashlib
from pathlib import Path
import numpy as np
from numba import njit

# Windows pip CUDA DLLs; no change to the system driver or global PATH.
_dll_handles=[]
if os.name == 'nt':
    for root in [Path(sys.prefix)/'Lib/site-packages/nvidia']:
        if root.exists():
            for p in root.glob('*/bin'):
                _dll_handles.append(os.add_dll_directory(str(p)))
GPU_ERROR=""
try:
    if os.environ.get("RT616_CPU_ONLY") == "1":
        raise RuntimeError("CPU-only mode requested")
    import cupy as cp
    GPU_AVAILABLE=cp.cuda.runtime.getDeviceCount()>0
except Exception as exc:
    cp=None; GPU_AVAILABLE=False; GPU_ERROR=str(exc)

CUDA_SOURCE=r'''
extern "C" __global__ void project_depth(
    const float* xyz, const float* R, const float* eye,
    unsigned int* depth, int n, int w, int h, float f) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float x=xyz[3*i]-eye[0], y=xyz[3*i+1]-eye[1], z=xyz[3*i+2]-eye[2];
    float X=R[0]*x+R[1]*y+R[2]*z;
    float Y=R[3]*x+R[4]*y+R[5]*z;
    float Z=R[6]*x+R[7]*y+R[8]*z;
    if(Z<=0.01f) return;
    int u=(int)floorf(f*X/Z+0.5f*w);
    int v=(int)floorf(0.5f*h-f*Y/Z);
    if(u>=0 && u<w && v>=0 && v<h)
        atomicMin(depth+v*w+u,__float_as_uint(Z));
}
'''

def load_bunny(cache='rt616_data'):
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    archive=cache/'bunny.tar.gz'
    if not archive.exists():
        bundled=(Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd())/'bunny.tar.gz'
        if bundled.exists():
            import shutil
            shutil.copyfile(bundled,archive)
        else:
            urllib.request.urlretrieve('https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz',archive)
    # Read one known member; never extract archive paths.
    with tarfile.open(archive,'r:gz') as tf:
        raw=tf.extractfile('bunny/reconstruction/bun_zipper.ply').read()
    header, body=raw.split(b'end_header\n',1)
    assert b'format ascii' in header
    n=int(next(x for x in header.decode().splitlines() if x.startswith('element vertex')).split()[-1])
    pts=np.asarray([[float(v) for v in line.split()[:3]] for line in body.decode().splitlines()[:n]],np.float32)
    pts-=np.array([(pts[:,0].min()+pts[:,0].max())/2,pts[:,1].min(),(pts[:,2].min()+pts[:,2].max())/2],np.float32)
    pts/=np.ptp(pts[:,1])
    return pts, hashlib.sha256(archive.read_bytes()).hexdigest()

def make_scene(bunny,side=8):
    offsets=np.array([[(i-(side-1)/2)*1.5,0,(j-(side-1)/2)*1.5] for i in range(side) for j in range(side)],np.float32)
    return np.ascontiguousarray((bunny[None,:,:]+offsets[:,None,:]).reshape(-1,3))

def camera(angle,side=8):
    radius=side*1.15
    eye=np.array([radius*np.sin(angle),side*.7,radius*np.cos(angle)],np.float32)
    forward=np.array([0,.3,0],np.float32)-eye; forward/=np.linalg.norm(forward)
    right=np.cross(forward,np.array([0,1,0],np.float32)); right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    return np.ascontiguousarray(np.stack([right,up,forward])),eye

@njit(cache=True,fastmath=False)
def render_cpu(points,R,eye,w=640,h=480,f=np.float32(500)):
    depth=np.full((h,w),np.float32(np.inf),np.float32)
    for i in range(len(points)):
        x=points[i,0]-eye[0]; y=points[i,1]-eye[1]; z=points[i,2]-eye[2]
        X=R[0,0]*x+R[0,1]*y+R[0,2]*z
        Y=R[1,0]*x+R[1,1]*y+R[1,2]*z
        Z=R[2,0]*x+R[2,1]*y+R[2,2]*z
        if Z<=np.float32(.01): continue
        u=int(np.floor(f*X/Z+np.float32(.5*w)))
        v=int(np.floor(np.float32(.5*h)-f*Y/Z))
        if 0<=u<w and 0<=v<h and Z<depth[v,u]: depth[v,u]=Z
    return depth

def render_numpy(points,R,eye,w=640,h=480,f=np.float32(500)):
    # Independent vectorized reference; not the optimized CPU timing baseline.
    p=(points-eye)@R.T
    valid=p[:,2]>np.float32(.01); p=p[valid]
    u=np.floor(f*p[:,0]/p[:,2]+np.float32(.5*w)).astype(np.int32)
    v=np.floor(np.float32(.5*h)-f*p[:,1]/p[:,2]).astype(np.int32)
    mask=(u>=0)&(u<w)&(v>=0)&(v<h)
    out=np.full(w*h,np.inf,np.float32)
    np.minimum.at(out,v[mask]*w+u[mask],p[mask,2])
    return out.reshape(h,w)

class GPURenderer:
    def __init__(self,points,w=640,h=480):
        if not GPU_AVAILABLE: raise RuntimeError('CUDA GPU unavailable; select a GPU runtime in Colab')
        self.points=cp.asarray(points); self.w=w; self.h=h
        self.depth=cp.empty((h,w),cp.float32)
        self.R=cp.empty((3,3),cp.float32); self.eye=cp.empty(3,cp.float32)
        self.kernel=cp.RawKernel(CUDA_SOURCE,'project_depth',options=('--fmad=false',))
        cp.cuda.get_current_stream().synchronize()
    def set_camera(self,R,eye): self.R.set(R); self.eye.set(eye)
    def compute(self):
        self.depth.fill(cp.inf)
        if len(self.points)==0: return self.depth
        self.kernel(((len(self.points)+255)//256,),(256,),
            (self.points,self.R,self.eye,self.depth,np.int32(len(self.points)),np.int32(self.w),np.int32(self.h),np.float32(500)))
        return self.depth
    def frame(self,R,eye,copy_points=None):
        if copy_points is not None: self.points.set(copy_points)
        self.set_camera(R,eye); self.compute()
        return cp.asnumpy(self.depth)

def stats(times):
    a=np.array(times)*1000
    return dict(median_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)),iqr_ms=float(np.percentile(a,75)-np.percentile(a,25)))

def benchmark(side=8,repeats=30,w=640,h=480,cache='rt616_data'):
    bunny,sha=load_bunny(cache); points=make_scene(bunny,side); R,eye=camera(.2,side)
    cpu=render_cpu(points,R,eye,w,h) # JIT cold time excluded
    info=dict(cpu=platform.processor(),platform=platform.platform(),python=sys.version.split()[0],
              numpy=np.__version__,side=side,bunnies=side*side,points=len(points),width=w,height=h,
              repeats=repeats,dataset_sha256=sha,cpu_baseline='Numba fused single-thread; fastmath=False',
              scope='warm compute; GPU resident frame includes camera upload and depth download; per-frame point-upload measured separately; browser display excluded')
    times=[]
    for _ in range(repeats):
        t=time.perf_counter(); render_cpu(points,R,eye,w,h); times.append(time.perf_counter()-t)
    info['cpu']=dict(name=platform.processor(),**stats(times))
    if not GPU_AVAILABLE:
        info['gpu_unavailable']=globals().get('GPU_ERROR','No device'); return info
    renderer=GPURenderer(points,w,h); gpu=renderer.frame(R,eye)
    props=cp.cuda.runtime.getDeviceProperties(0)
    info['device']=dict(name=props['name'].decode(),compute_capability=f"{props['major']}.{props['minor']}",
                        total_bytes=int(props['totalGlobalMem']),cupy=cp.__version__,runtime=cp.cuda.runtime.runtimeGetVersion(),driver=cp.cuda.runtime.driverGetVersion())
    common=np.isfinite(cpu)&np.isfinite(gpu); union=np.isfinite(cpu)|np.isfinite(gpu)
    info['accuracy']=dict(mask_iou=float(common.sum()/max(1,union.sum())),max_depth_error=float(np.max(np.abs(cpu[common]-gpu[common]))),
                          mean_depth_error=float(np.mean(np.abs(cpu[common]-gpu[common]))))
    assert info['accuracy']['mask_iou']>.999,info['accuracy']
    assert info['accuracy']['mean_depth_error']<1e-4,info['accuracy']
    times=[]
    for _ in range(repeats):
        start,end=cp.cuda.Event(),cp.cuda.Event(); start.record(); renderer.compute(); end.record(); end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start,end)/1000)
    info['gpu_compute']=stats(times)
    for label,upload in [('gpu_resident_frame',None),('gpu_upload_each_frame',points)]:
        times=[]
        for _ in range(repeats):
            t=time.perf_counter(); renderer.frame(R,eye,upload); times.append(time.perf_counter()-t)
        info[label]=stats(times)
    info['speedup_resident']=info['cpu']['median_ms']/info['gpu_resident_frame']['median_ms']
    return info

def depth_rgb(depth,near=0,far=24):
    # A visualization of measured optical-axis depth, with fixed limits across frames.
    from matplotlib import colormaps
    valid=np.isfinite(depth); a=np.zeros_like(depth)
    a[valid]=np.clip((depth[valid]-near)/(far-near),0,1)
    rgb=(colormaps['turbo'](a)[...,:3]*255).astype(np.uint8); rgb[~valid]=245
    return rgb

def interactive_view(side=8,backend='GPU'):
    import ipywidgets as widgets
    from IPython.display import display,clear_output
    from PIL import Image
    bunny,_=load_bunny(); points=make_scene(bunny,side)
    renderer=GPURenderer(points) if GPU_AVAILABLE else None
    render_cpu(points,*camera(0,side))
    angle=widgets.FloatSlider(min=-180,max=180,value=0,step=3,description='Yaw',continuous_update=False)
    choice=widgets.Dropdown(options=['CPU']+(['GPU'] if renderer else []),value=backend if renderer else 'CPU',description='Backend')
    out=widgets.Output()
    def draw(_=None):
        R,eye=camera(np.deg2rad(angle.value),side)
        t=time.perf_counter(); depth=renderer.frame(R,eye) if choice.value=='GPU' else render_cpu(points,R,eye)
        ms=(time.perf_counter()-t)*1000
        with out:
            clear_output(wait=True)
            print(f'{len(points):,} points | {choice.value} frame compute/copy {ms:.2f} ms | notebook display excluded')
            display(Image.fromarray(depth_rgb(depth,far=side*3)))
    angle.observe(draw,names='value'); choice.observe(draw,names='value'); display(widgets.VBox([angle,choice,out])); draw()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--out',default='results'); p.add_argument('--repeats',type=int,default=30)
    args=p.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    results=[benchmark(side=s,repeats=args.repeats,cache=out/'data') for s in [2,8,12]]
    (out/'benchmark.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(results,indent=2,ensure_ascii=False))
    from PIL import Image
    bunny,_=load_bunny(out/'data'); points=make_scene(bunny,8); renderer=GPURenderer(points) if GPU_AVAILABLE else None
    for j,a in enumerate([0,.55,1.1]):
        R,eye=camera(a,8); d=renderer.frame(R,eye) if renderer else render_cpu(points,R,eye)
        Image.fromarray(depth_rgb(d,far=24)).save(out/f'bunny-depth-{j}.png')
