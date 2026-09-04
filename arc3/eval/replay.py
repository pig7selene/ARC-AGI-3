"""Small self-contained HTML replay viewer for JSONL traces."""
from __future__ import annotations
import argparse,html,json
from pathlib import Path

TEMPLATE='''<!doctype html><meta charset="utf-8"><title>ARC-AGI-3 replay</title><style>body{{font:14px system-ui;background:#111;color:#eee}}section{{display:flex;gap:20px;align-items:flex-start}}canvas{{image-rendering:pixelated;border:1px solid #888;width:512px;height:512px}}pre{{white-space:pre-wrap;max-width:600px}}</style><h1>ARC-AGI-3 replay</h1><section><canvas id="c" width="64" height="64"></canvas><pre id="m"></pre></section><button id="p">Play</button> <input id="s" type="range" min="0" max="{max}" value="0" style="width:400px"><script>const t={data};let i=0, c=document.querySelector('canvas'),x=c.getContext('2d'),s=document.querySelector('#s'),m=document.querySelector('#m');function draw(){{let e=t[i]||{{}},g=e.grid||[];c.width=g[0]?.length||64;c.height=g.length||64;let im=x.createImageData(c.width,c.height),pal=['#000','#e22','#2c6','#28f','#fc2','#f2f','#2dd','#fff','#888','#f80','#0ff','#adf','#fa8','#d8f','#afa','#f88'];g.flat().forEach((v,j)=>{{let q=pal[v]||'#fff',n=parseInt(q.slice(1),16),k=j*4;im.data[k]=n>>16;im.data[k+1]=n>>8&255;im.data[k+2]=n&255;im.data[k+3]=255}});x.putImageData(im,0,0);m.textContent=JSON.stringify({{step:e.step,action:e.chosen_action||e.action,phase:e.phase,diff:e.actual_frame_diff,expected:e.expected_effect,hypotheses:e.current_hypotheses}},null,2);s.value=i}}s.oninput=()=>{{i=+s.value;draw()}};document.querySelector('#p').onclick=()=>{{let z=setInterval(()=>{{if(i>=t.length-1)return clearInterval(z);i++;draw()}},400)}};draw();</script>'''

def write_replay(source:Path,dest:Path):
    rows=[json.loads(x) for x in source.read_text(encoding="utf8").splitlines() if x.strip()]; dest.write_text(TEMPLATE.format(max=max(0,len(rows)-1),data=json.dumps(rows,default=str)),encoding="utf8"); return dest
def main():
    p=argparse.ArgumentParser(); p.add_argument("trace",type=Path); p.add_argument("-o","--output",type=Path); a=p.parse_args(); out=a.output or a.trace.with_suffix(".html"); print(write_replay(a.trace,out))
if __name__=="__main__": main()
