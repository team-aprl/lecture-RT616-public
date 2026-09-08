from stanford_scene import *
from http.server import HTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
import socket
from benchmark_lab import benchmark_frames
active={}
def scene(model,side,cap):
    key=(model,side,cap)
    if active.get('key')!=key:
        active.clear();gc.collect()
        if GPU_AVAILABLE:cp.get_default_memory_pool().free_all_blocks()
        active['scene']=Scene(model,side,cap);active['key']=key
    return active['scene']
HTML=r"""<!doctype html><meta charset="utf-8"><title>RT616 · Stanford Trajectory Lab</title>
<style>body{font:16px Arial;margin:24px;background:#f7fafb;color:#183039}main{max-width:1150px}canvas{width:min(900px,100%);box-sizing:border-box;border:1px solid #ddd}button,select{font:inherit;margin:5px;padding:7px}label{display:inline-block;margin:5px}input{vertical-align:middle}small{display:block;line-height:1.6;max-width:950px}#status{font:14px monospace;margin:12px 0;min-height:36px}h1{margin-bottom:8px}</style>
<main><h1>Stanford Trajectory Lab</h1><p>모델과 격자를 선택하고, 카메라가 접근·후퇴하거나 장면 사이를 비행하는 깊이 영상을 보세요.</p>
<label>Model <select id="model"><option>Bunny</option><option>Dragon</option><option>Happy Buddha</option><option>Drill</option></select></label>
<label>Grid <select id="side"><option>2</option><option selected>8</option><option>12</option><option>16</option><option>32</option><option>64</option></select></label>
<label>Points/model <select id="cap"><option>12000</option><option selected>36000</option><option>100000</option></select></label>
<label>Backend <select id="backend"><option>GPU</option><option>CPU</option></select></label><br>
<label>Path <select id="path"><option>Orbit</option><option>Dolly</option><option>Helix</option><option>Fly-through</option><option selected>Hover</option></select></label>
<button id="play">경로 재생</button><button id="reset">시점 초기화</button>
<label><input id="upload" type="checkbox">GPU 원본+배치 재업로드</label><br>
<label>Yaw <input id="yaw" type="range" min="-180" max="180" value="0"></label>
<label>Pitch <input id="pitch" type="range" min="-60" max="85" value="25"></label>
<label>Distance/Z <input id="distance" type="range" min="0.1" max="2.5" step="0.02" value="1.2"></label><br>
<label>경로 위치 <input id="phase" type="range" min="0" max="1" step="0.002" value="0"></label>
<label>속도 <input id="speed" type="range" min="0.1" max="3" step="0.1" value="1"></label>
<section style="margin:14px 0;padding:12px;background:#e9f0f2">
<label>K <input id="batchK" type="number" min="1" max="300" value="60" style="width:65px"></label>
<label>비교 <select id="batchMode"><option>Both</option><option>CPU</option><option>GPU</option></select></label>
<button id="batchRun">K 프레임 렌더링</button><button id="batchStop" disabled>중단</button>
<small>현재 장면·경로를 한 바퀴 도는 동일한 K개 시점을 계산합니다. 이미지 전송 없이 서버 안에서 연속 측정합니다. Warmup 3회는 제외합니다.</small>
<div id="batchSummary" style="font-weight:bold;margin:8px 0" aria-live="polite"></div>
<pre id="batchLog" style="max-height:260px;overflow:auto;background:#122731;color:#bce6d1;padding:10px;font:13px monospace;white-space:pre-wrap">프레임별 CPU/GPU ms가 여기에 출력됩니다.</pre>
<button id="batchSave" disabled>측정 JSON 저장</button>
</section><div id="status">초기화 중… 처음 고르는 모델은 Stanford에서 다운로드합니다.</div><canvas id="view" width="640" height="480"></canvas>
<small>Orbit: 공전 · Dolly: 접근/후퇴 · Helix: 고도가 변하는 공전 · Fly-through: 격자 내부 통과 · Hover: 거리·높이 변화.<br>
원본 모델 + 배치 좌표만 저장하는 instancing을 CPU/GPU에 동일하게 적용합니다. 모든 표시 점을 처리하며 culling으로 점 수를 줄이지 않습니다. Points/model은 명시적인 결정적 샘플링 상한입니다. 재업로드는 이 압축 표현의 복사이며 이전 실습의 전체 점 배열 복사와 범위가 다릅니다.<br>
깊이는 camera Z, 단위는 모델 높이=1인 scene unit. 가까운 곳은 청색, 먼 곳은 황색/적색이며 빈 픽셀은 연회색입니다. CPU는 컴파일된 단일 스레드입니다. 화면 표시·색칠·통신은 backend 시간에서 제외합니다.<br>
<a href="https://colab.research.google.com/github/team-aprl/lecture-RT616-public/blob/main/practice/gpu/bunnys_projections/PointCloud_CPU_GPU.ipynb">Colab에서 같은 실험</a> · <a href="https://graphics.stanford.edu/data/3Dscanrep/">Stanford Computer Graphics Laboratory</a></small></main>
<script>const $=x=>document.getElementById(x),ctx=$('view').getContext('2d');if(!__GPU_AVAILABLE__){
 $('backend').value='CPU';$('batchMode').value='CPU';
 for(const id of ['backend','batchMode'])for(const option of $(id).options)if(option.value!=='CPU')option.disabled=true;
}
let running=false,busy=false,pending=false,batching=false,batchAbort=null,batchRows=[],last=performance.now();
async function frame(){if(batching)return;if(busy){pending=true;return;}busy=true;pending=false;let begin=performance.now();
const p=new URLSearchParams();['model','side','cap','backend','path','yaw','pitch','distance','phase'].forEach(k=>p.set(k,$(k).value));p.set('upload',$('upload').checked?'1':'0');p.set('format',new URLSearchParams(location.search).get('transport')==='png'?'png':'raw');
try{let r=await fetch('frame?'+p);if(!r.ok)throw Error(await r.text());if(p.get('format')==='png'){let bitmap=await createImageBitmap(await r.blob());ctx.drawImage(bitmap,0,0);bitmap.close();}else{let bytes=await r.arrayBuffer();ctx.putImageData(new ImageData(new Uint8ClampedArray(bytes),640,480),0,0);}
$('status').textContent=`${p.get('model')} | ${r.headers.get('X-Points')} points / ${p.get('side')}×${p.get('side')} instances | stored ${r.headers.get('X-Stored-MiB')} MiB | ${p.get('backend')} ${r.headers.get('X-Render-Ms')} ms | viewer ${(performance.now()-begin).toFixed(1)} ms | eye ${r.headers.get('X-Eye')}`;
}catch(e){$('status').textContent=e.message;running=false;$('play').textContent='경로 재생';}finally{busy=false;}
if(running){let now=performance.now(),dt=Math.min((now-last)/1000,.25);last=now;$('phase').value=(Number($('phase').value)+dt*Number($('speed').value)/20)%1;setTimeout(frame,Math.max(0,16.7-(performance.now()-begin)));}else if(pending)frame();}
$('play').onclick=()=>{running=!running;last=performance.now();$('play').textContent=running?'정지':'경로 재생';if(running)frame();};
$('reset').onclick=()=>{$('yaw').value=0;$('pitch').value=25;$('distance').value=1.2;$('phase').value=0;frame();};
['model','side','cap','backend','path','yaw','pitch','distance','phase','upload'].forEach(k=>$(k).onchange=frame);
let drag=null;$('view').onpointerdown=e=>{drag=[e.clientX,e.clientY];$('view').setPointerCapture(e.pointerId)};
$('view').onpointermove=e=>{if(!drag)return;$('yaw').value=Math.max(-180,Math.min(180,Number($('yaw').value)+(e.clientX-drag[0])*.3));$('pitch').value=Math.max(-60,Math.min(85,Number($('pitch').value)-(e.clientY-drag[1])*.2));drag=[e.clientX,e.clientY];frame()};$('view').onpointerup=()=>drag=null;
$('view').onwheel=e=>{e.preventDefault();$('distance').value=Math.max(.1,Math.min(2.5,Number($('distance').value)+e.deltaY*.001));frame()};
function batchLine(t){$('batchLog').textContent+=t+'\n';$('batchLog').scrollTop=$('batchLog').scrollHeight;}
function batchEvent(e){
 batchRows.push(e);
 if(e.type==='start')batchLine(e.scene.model+' | '+e.scene.grid+'×'+e.scene.grid+' | '+e.scene.logical_points.toLocaleString()+' points/frame | K='+e.k+' | '+e.path);
 if(e.type==='warmup')batchLine(e.message);
 if(e.type==='frame'){
  batchLine('['+String(e.frame).padStart(3,'0')+'/'+e.k+'] '+['CPU','GPU'].filter(b=>b in e.ms).map(b=>b+' '+e.ms[b].toFixed(3)+' ms').join(' | ')+(e.matching===undefined?'':' | depth '+(e.matching?'MATCH':'MISMATCH')));
  $('batchSummary').textContent=e.frame+' / '+e.k+' 프레임 완료';
 }
 if(e.type==='done'){
  let lines=Object.entries(e.stats).map(([b,v])=>b+': 총 '+v.total_ms.toFixed(2)+' ms | 평균 '+v.mean_ms.toFixed(3)+' | 중앙값 '+v.median_ms.toFixed(3)+' | p95 '+v.p95_ms.toFixed(3)+' ms');
  if(e.speedup)lines.push('CPU/GPU 총 계산시간 비율: '+e.speedup.toFixed(2)+'×');
  $('batchSummary').textContent=lines.join(' / ');lines.forEach(batchLine);
  batchLine('서버 경과시간 '+e.batch_wall_ms.toFixed(1)+' ms (로그·검증 포함). 위 총시간은 프레임 렌더링 시간의 합.');
 }
 if(e.type==='error')throw Error(e.message);
}
$('batchRun').onclick=async()=>{
 const k=Number($('batchK').value);if(!Number.isInteger(k)||k<1||k>300){$('batchSummary').textContent='K는 1~300의 정수입니다.';return;}
 running=false;$('play').textContent='경로 재생';batching=true;
 while(busy)await new Promise(r=>setTimeout(r,20));pending=false;
 const p=new URLSearchParams();['model','side','cap','path','yaw','pitch','distance','phase'].forEach(x=>p.set(x,$(x).value));p.set('k',k);p.set('mode',$('batchMode').value);p.set('upload',$('upload').checked?'1':'0');
 const controls=[...document.querySelectorAll('input,select,button')].filter(e=>!['batchStop','batchSave'].includes(e.id));
 controls.forEach(e=>e.disabled=true);$('batchStop').disabled=false;$('batchSave').disabled=true;
 $('batchLog').textContent='';$('batchSummary').textContent='장면 준비·warmup 중…';batchRows=[];batchAbort=new AbortController();
 try{
  const r=await fetch('benchmark?'+p,{signal:batchAbort.signal});if(!r.ok)throw Error(await r.text());
  const reader=r.body.getReader(),decoder=new TextDecoder();let buf='';
  while(true){const {done,value}=await reader.read();buf+=decoder.decode(value||new Uint8Array(),{stream:!done});let lines=buf.split('\n');buf=lines.pop();for(const line of lines)if(line.trim())batchEvent(JSON.parse(line));if(done){if(buf.trim())batchEvent(JSON.parse(buf));break;}}
 }catch(e){batchLine(e.name==='AbortError'?'사용자가 중단했습니다.':'오류: '+e.message);$('batchSummary').textContent=e.name==='AbortError'?'중단됨':'측정 오류';}
 finally{batching=false;controls.forEach(e=>e.disabled=false);$('batchStop').disabled=true;$('batchSave').disabled=!batchRows.length;batchAbort=null;}
};
$('batchStop').onclick=()=>batchAbort?.abort();
$('batchSave').onclick=()=>{let url=URL.createObjectURL(new Blob([JSON.stringify(batchRows,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=url;a.download='rt616-k-frames.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
frame();</script>"""
class Handler(BaseHTTPRequestHandler):
    def setup(self):super().setup();self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    def do_GET(self):
        p=urlparse(self.path)
        if p.path=='/':
            b=HTML.replace('__GPU_AVAILABLE__','true' if GPU_AVAILABLE else 'false').encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.end_headers();self.wfile.write(b);return
        if p.path=='/benchmark':self.benchmark(p);return
        if p.path!='/frame':self.send_error(404);return
        try:
            q=parse_qs(p.query);get=lambda k,d:q.get(k,[d])[0]
            model=get('model','Bunny');side=int(get('side','8'));cap=int(get('cap','36000'));path=get('path','Hover')
            if model not in MODELS or cap not in [12000,36000,100000] or path not in PATHS:raise ValueError('Invalid selection')
            s=scene(model,side,cap);R,e=pose(float(get('phase','0')),s.extent,path,float(get('yaw','0')),float(get('pitch','25')),float(get('distance','1.2')))
            start=time.perf_counter();d=s.frame(R,e,get('backend','GPU'),get('upload','0')=='1');ms=(time.perf_counter()-start)*1000
            rgb=depth_rgb(d,far=s.extent*3);rgba=np.full((480,640,4),255,np.uint8);rgba[:,:,:3]=rgb;b=rgba.tobytes()
            ctype='application/octet-stream'
            if get('format','raw')=='png':
                from PIL import Image
                stream=io.BytesIO();Image.fromarray(rgb).save(stream,format='PNG',compress_level=1);b=stream.getvalue();ctype='image/png'
            self.send_response(200);self.send_header('Content-Type',ctype);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(b)))
            for k,v in {'X-Render-Ms':f'{ms:.3f}','X-Points':str(s.n),'X-Stored-MiB':f'{s.info["stored_bytes"]/2**20:.2f}','X-Eye':','.join(f'{x:.2f}' for x in e)}.items():self.send_header(k,v)
            self.end_headers();self.wfile.write(b)
        except Exception as ex:self.send_error(400,str(ex))

    def benchmark(self,p):
        started=False
        try:
            q=parse_qs(p.query);get=lambda k,d:q.get(k,[d])[0]
            model=get('model','Bunny');cap=int(get('cap','36000'));path=get('path','Hover')
            if model not in MODELS or cap not in [12000,36000,100000] or path not in PATHS:raise ValueError('Invalid selection')
            s=scene(model,int(get('side','8')),cap)
            events=benchmark_frames(s,int(get('k','60')),path,float(get('phase','0')),float(get('yaw','0')),float(get('pitch','25')),float(get('distance','1.2')),get('mode','Both'),get('upload','0')=='1')
            first=next(events)
            self.send_response(200);self.send_header('Content-Type','application/x-ndjson; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('X-Accel-Buffering','no');self.end_headers();started=True
            def send(event):self.wfile.write((json.dumps(event)+'\n').encode());self.wfile.flush()
            send(first)
            for event in events:send(event)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
        except Exception as ex:
            if not started:self.send_error(400,str(ex))
            else:
                try:self.wfile.write((json.dumps(dict(type='error',message=str(ex)))+'\n').encode());self.wfile.flush()
                except OSError:pass

    def log_message(self,*args):pass
if __name__=='__main__':
    import os
    import argparse
    parser=argparse.ArgumentParser(description='Stanford point-cloud CPU/CUDA viewer')
    parser.add_argument('--port',type=int,default=int(os.environ.get('RT616_DEMO_PORT','8766')))
    args=parser.parse_args();port=args.port
    print('Preparing Bunny (first run downloads the Stanford archive)...',flush=True)
    scene('Bunny',8,36000);print(f'Open http://127.0.0.1:{port}',flush=True);HTTPServer(('127.0.0.1',port),Handler).serve_forever()
