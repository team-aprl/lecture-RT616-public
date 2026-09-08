"""Fixed-pose batches: rendering timings exclude image encoding and transport."""
from stanford_scene import *
def benchmark_frames(scene,k=60,path='Hover',phase=0,yaw=0,pitch=25,distance=1.2,mode='Both',upload=False):
    if not 1<=k<=300:raise ValueError('K must be between 1 and 300')
    if mode not in ['Both','CPU','GPU']:raise ValueError('Invalid backend')
    backends=['CPU','GPU'] if mode=='Both' else [mode]
    if 'GPU' in backends and not GPU_AVAILABLE:raise RuntimeError('GPU unavailable; select CPU or enable GPU runtime')
    phases=[(phase+i/k)%1 for i in range(k)]
    poses=[pose(t,scene.extent,path,yaw,pitch,distance) for t in phases]
    environment=dict(python=platform.python_version(),os=platform.system(),numpy=np.__version__,gpu_available=GPU_AVAILABLE)
    if GPU_AVAILABLE:
        props=cp.cuda.runtime.getDeviceProperties(0);name=props['name']
        environment.update(gpu=name.decode() if isinstance(name,bytes) else name,cupy=cp.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion())
    yield dict(type='start',environment=environment,resolution=[640,480],dtype='float32',k=k,scene=scene.info,path=path,backends=backends,upload=upload,
               measurement='render+GPU pose upload+depth readback; image encoding/transfer/display excluded')
    # Compile and stabilize before recording. Results must finish on the host.
    for backend in backends:
        for _ in range(3):scene.frame(*poses[0],backend,upload)
    yield dict(type='warmup',message='Warmup complete: 3 frames/backend, excluded.')
    totals={b:[] for b in backends};start_batch=time.perf_counter()
    for i,(R,e) in enumerate(poses):
        row=dict(type='frame',frame=i+1,k=k,phase=phases[i],ms={})
        results={}
        # Alternate execution order to reduce systematic first/second bias.
        order=backends if i%2==0 else list(reversed(backends))
        for backend in order:
            start=time.perf_counter();depth=scene.frame(R,e,backend,upload);ms=(time.perf_counter()-start)*1000
            row['ms'][backend]=ms;totals[backend].append(ms)
            if i in [0,k-1]:results[backend]=depth
        if len(results)==2:
            a,b=results['CPU'],results['GPU'];mask=np.isfinite(a)
            row['matching']=bool(np.array_equal(mask,np.isfinite(b)) and np.allclose(a[mask],b[mask],rtol=1e-5,atol=1e-5))
        print(f"[{i+1:03d}/{k}] "+' | '.join(f'{b}: {row["ms"][b]:.3f} ms' for b in backends),flush=True)
        yield row
    stats={b:dict(total_ms=float(np.sum(v)),mean_ms=float(np.mean(v)),median_ms=float(np.median(v)),
                  p95_ms=float(np.percentile(v,95))) for b,v in totals.items()}
    yield dict(type='done',stats=stats,batch_wall_ms=(time.perf_counter()-start_batch)*1000,
               speedup=stats['CPU']['total_ms']/stats['GPU']['total_ms'] if len(backends)==2 else None)
