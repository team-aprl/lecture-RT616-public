"""Instanced Stanford models, camera trajectories, local and Colab controls."""
from bunny_depth_lab import *
import io,gc
MODELS={
 'Bunny':('https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz','bun_zipper.ply'),
 'Dragon':('https://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz','dragon_vrip.ply'),
 'Happy Buddha':('https://graphics.stanford.edu/pub/3Dscanrep/happy/happy_recon.tar.gz','happy_vrip.ply'),
 'Drill':('https://graphics.stanford.edu/pub/3Dscanrep/drill.tar.gz','drill_shaft_vrip.ply')}
PATHS=['Orbit','Dolly','Helix','Fly-through','Hover']
def load_model(name='Bunny',cap=36000,cache='rt616_data'):
    from plyfile import PlyData
    url,suffix=MODELS[name];folder=Path(cache);folder.mkdir(parents=True,exist_ok=True)
    archive=folder/url.split('/')[-1]
    if not archive.exists():urllib.request.urlretrieve(url,archive)
    with tarfile.open(archive,'r:gz') as tf:
        candidates=[m for m in tf.getmembers() if m.name.endswith(suffix)]
        if not candidates:raise ValueError(f'{name}: missing {suffix}; PLY members: {[m.name for m in tf.getmembers() if m.name.endswith(".ply")]}')
        ply=PlyData.read(io.BytesIO(tf.extractfile(candidates[0]).read()))
    v=ply['vertex'];pts=np.column_stack([v[k] for k in ['x','y','z']]).astype(np.float32)
    count=len(pts);pts-=np.array([(pts[:,0].min()+pts[:,0].max())/2,pts[:,1].min(),(pts[:,2].min()+pts[:,2].max())/2],np.float32)
    pts/=np.ptp(pts[:,1])
    if cap and count>cap:
        pts=pts[np.sort(np.random.default_rng(616).choice(count,int(cap),replace=False))]
    return np.ascontiguousarray(pts),dict(model=name,source=url,original_points=count,points_per_instance=len(pts),sample_seed=616,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())

def pose(t,side=8,path='Orbit',yaw=0,pitch=25,distance=1.2):
    """t is normalized path phase; distance is relative to the grid extent."""
    a=2*np.pi*t+np.deg2rad(yaw);r=side*distance;target=np.array([0,.5,0],np.float32)
    elev=np.deg2rad(pitch)
    if path=='Dolly':r*=.2+.8*(.5+.5*np.cos(2*np.pi*t));a=np.deg2rad(yaw)
    elif path=='Helix':elev=np.deg2rad(np.clip(pitch+32.5*np.sin(2*np.pi*t),-75,85))
    elif path=='Hover':r*=1+.15*np.sin(6*np.pi*t);elev+=.15*np.sin(4*np.pi*t)
    if path=='Fly-through':
        x=.55*np.sin(2*np.pi*t);z=side*distance*np.cos(2*np.pi*t);y=np.deg2rad(yaw)
        eye=np.array([x*np.cos(y)+z*np.sin(y),1.1+.5*np.sin(4*np.pi*t)+(pitch-25)*.02,-x*np.sin(y)+z*np.cos(y)],np.float32)
        target=np.array([0,.5,0],np.float32)
    else:eye=target+np.array([r*np.cos(elev)*np.sin(a),r*np.sin(elev),r*np.cos(elev)*np.cos(a)],np.float32)
    f=target-eye;f/=np.linalg.norm(f);right=np.cross(f,np.array([0,1,0],np.float32));right/=np.linalg.norm(right)
    up=np.cross(right,f)
    return np.ascontiguousarray(np.stack([right,up,f]),np.float32),np.ascontiguousarray(eye,np.float32)

@njit(cache=True,fastmath=False)
def render_instances_cpu(base,offsets,R,eye,w=640,h=480,f=np.float32(500)):
    d=np.full((h,w),np.float32(np.inf),np.float32)
    for j in range(len(offsets)):
        for i in range(len(base)):
            x=(base[i,0]+offsets[j,0])-eye[0];y=(base[i,1]+offsets[j,1])-eye[1];z=(base[i,2]+offsets[j,2])-eye[2]
            X=R[0,0]*x+R[0,1]*y+R[0,2]*z;Y=R[1,0]*x+R[1,1]*y+R[1,2]*z;Z=R[2,0]*x+R[2,1]*y+R[2,2]*z
            if Z<=np.float32(.01):continue
            u=int(np.floor(f*X/Z+np.float32(w*.5)));v=int(np.floor(np.float32(h*.5)-f*Y/Z))
            if 0<=u<w and 0<=v<h and Z<d[v,u]:d[v,u]=Z
    return d

INSTANCE_KERNEL=CUDA_SOURCE.replace('const float* xyz,','const float* xyz, const float* offsets, int base_n,').replace(
 'float x=xyz[3*i]-eye[0], y=xyz[3*i+1]-eye[1], z=xyz[3*i+2]-eye[2];',
 'int b=i%base_n, j=i/base_n; float x=(xyz[3*b]+offsets[3*j])-eye[0], y=(xyz[3*b+1]+offsets[3*j+1])-eye[1], z=(xyz[3*b+2]+offsets[3*j+2])-eye[2];')
class Scene:
    def __init__(self,model='Bunny',side=8,cap=36000,cache='rt616_data'):
        if side not in [2,8,12,16,32,64]:raise ValueError('Unsupported grid')
        self.base,self.info=load_model(model,cap,cache);self.side=side
        spacing=max(1.5,float(np.ptp(self.base[:,0]))*1.15,float(np.ptp(self.base[:,2]))*1.15)
        self.offsets=np.array([[(i-(side-1)/2)*spacing,0,(j-(side-1)/2)*spacing] for i in range(side) for j in range(side)],np.float32)
        self.extent=side*spacing/1.5
        self.n=len(self.base)*len(self.offsets)
        if self.n>=2**31:raise ValueError('Too many points: reduce points/model')
        self.info.update(grid=side,instances=side*side,logical_points=self.n,stored_bytes=self.base.nbytes+self.offsets.nbytes,spacing=spacing)
        if GPU_AVAILABLE:
            self.bg=cp.asarray(self.base);self.og=cp.asarray(self.offsets);self.Rg=cp.empty((3,3),cp.float32);self.eg=cp.empty(3,cp.float32);self.dg=cp.empty((480,640),cp.float32)
            self.kernel=cp.RawKernel(INSTANCE_KERNEL,'project_depth',options=('--fmad=false',));self.frame(*pose(0,self.extent),'GPU')
        render_instances_cpu(self.base[:1],self.offsets[:1],*pose(0,self.extent))
    def frame(self,R,e,backend='GPU',upload=False):
        if backend=='CPU':return render_instances_cpu(self.base,self.offsets,R,e)
        if not GPU_AVAILABLE:raise RuntimeError('GPU unavailable; select a Colab GPU runtime or CPU')
        if upload:self.bg.set(self.base);self.og.set(self.offsets)
        self.Rg.set(R);self.eg.set(e);self.dg.fill(cp.inf)
        self.kernel(((self.n+255)//256,),(256,),(self.bg,self.og,np.int32(len(self.base)),self.Rg,self.eg,self.dg,np.int32(self.n),np.int32(640),np.int32(480),np.float32(500)))
        return cp.asnumpy(self.dg)

def trajectory(scene,path='Hover',frames=60,backend='GPU',yaw=0,pitch=25,distance=1.2,out='trajectory'):
    """Export a rendered GIF plus actual per-frame backend times and camera extrinsics."""
    from PIL import Image
    images=[];rows=[]
    for i in range(frames):
        t=i/frames;R,e=pose(t,scene.extent,path,yaw,pitch,distance)
        start=time.perf_counter();d=scene.frame(R,e,backend);ms=(time.perf_counter()-start)*1000
        images.append(Image.fromarray(depth_rgb(d,far=scene.extent*3)))
        rows.append(dict(frame=i,phase=t,ms=ms,eye=e.tolist(),R_world_to_camera=R.tolist()))
    images[0].save(str(out)+'.gif',save_all=True,append_images=images[1:],duration=50,loop=0)
    Path(str(out)+'.json').write_text(json.dumps(dict(scene=scene.info,path=path,backend=backend,frames=rows,playback='Fixed 20 fps preview, NOT measured throughput'),indent=2),encoding='utf-8')
    return rows

def colab_viewer():
    import ipywidgets as w
    from IPython.display import display,clear_output
    from PIL import Image
    model=w.Dropdown(options=list(MODELS),description='Model');side=w.Dropdown(options=[2,8,12,16,32,64],value=8,description='Grid')
    cap=w.Dropdown(options=[12000,36000,100000],value=36000,description='Points/model')
    path=w.Dropdown(options=PATHS,value='Hover',description='Path');backend=w.Dropdown(options=['GPU','CPU'] if GPU_AVAILABLE else ['CPU'],description='Backend')
    yaw=w.FloatSlider(min=-180,max=180,description='Yaw',continuous_update=False);pitch=w.FloatSlider(min=-60,max=85,value=25,description='Pitch',continuous_update=False)
    distance=w.FloatSlider(min=.1,max=2.5,value=1.2,step=.05,description='Distance/Z',continuous_update=False)
    phase=w.IntSlider(min=0,max=119,description='Path phase',continuous_update=False);play=w.Play(min=0,max=119,interval=100);w.jslink((play,'value'),(phase,'value'))
    build=w.Button(description='Load scene');export=w.Button(description='Render 60-frame GIF');output=w.Output();state={}
    def draw(_=None):
        if 'scene' not in state:return
        s=state['scene'];R,e=pose(phase.value/120,s.extent,path.value,yaw.value,pitch.value,distance.value)
        start=time.perf_counter();d=s.frame(R,e,backend.value);ms=(time.perf_counter()-start)*1000
        with output:
            clear_output(wait=True);print(s.info);print(f'{backend.value}: {ms:.2f} ms; widget/display excluded; eye={e}')
            display(Image.fromarray(depth_rgb(d,far=s.extent*3)))
    def load(_):
        play.value=0
        with output:clear_output(wait=True);print('Loading scene...')
        state.clear();gc.collect()
        if GPU_AVAILABLE:cp.get_default_memory_pool().free_all_blocks()
        state['scene']=Scene(model.value,side.value,cap.value);draw()
    def save(_):
        if 'scene' not in state:return
        with output:
            print('Rendering actual frames...')
            rows=trajectory(state['scene'],path.value,60,backend.value,yaw.value,pitch.value,distance.value)
            print('Saved trajectory.gif + trajectory.json; median backend ms:',np.median([x['ms'] for x in rows]))
            display(__import__('IPython').display.Image(filename='trajectory.gif'))
    build.on_click(load);export.on_click(save)
    for widget in [path,backend,yaw,pitch,distance,phase]:widget.observe(draw,names='value')
    display(w.VBox([w.HBox([model,side,cap]),build,w.HBox([path,backend]),yaw,pitch,distance,w.HBox([play,phase]),export,output]));load(None)
