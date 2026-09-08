"""Console entry point: same K poses, per-frame stdout, JSON measurements."""
import argparse,json,os
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--frames',type=int,default=60)
parser.add_argument('--grid',type=int,choices=[2,8,12,16,32,64],default=8)
parser.add_argument('--model',choices=['Bunny','Dragon','Happy Buddha','Drill'],default='Bunny')
parser.add_argument('--path',choices=['Orbit','Dolly','Helix','Fly-through','Hover'],default='Hover')
parser.add_argument('--points',type=int,choices=[12000,36000,100000],default=36000)
parser.add_argument('--backend',choices=['Both','CPU','GPU'],default='Both')
parser.add_argument('--upload',action='store_true')
parser.add_argument('--out',default='results/benchmark.json')
args=parser.parse_args()
if args.backend=='CPU':os.environ['RT616_CPU_ONLY']='1'
from stanford_scene import Scene
from benchmark_lab import benchmark_frames
scene=Scene(args.model,args.grid,args.points)
events=list(benchmark_frames(scene,args.frames,args.path,mode=args.backend,upload=args.upload))
dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
dest.write_text(json.dumps(events,indent=2),encoding='utf-8')
print(json.dumps(events[-1],indent=2));print('Saved',dest)
