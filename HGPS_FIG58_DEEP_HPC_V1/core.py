"""HGPS Figure 5-8 controlled research utilities (source provenance preserved)."""
from __future__ import annotations
import hashlib, json, os, time, traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

REF_SHA='a846fd6ba268dc1d9105b0ee3b48e58bb54b69d20b94e03bedd907b4eb0ffa2e'
V22_KNOWN_SHA='5ae6ffef4fbf777984e3b1052362770da8fcfb83f8400a438789094874677a6c'
WSS_KNOWN_SHA='a50905a79ca64c5b56444cd69f318eb06d83a6de1b1df7d2074c93dfa04d273f'

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def jd(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(obj),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def clean(obj):
    if isinstance(obj,dict): return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean(x) for x in obj]
    if isinstance(obj,np.ndarray):return clean(obj.tolist())
    if isinstance(obj,np.generic):return clean(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj): return None
    if isinstance(obj,Path):return str(obj)
    return obj

def stamp():return datetime.now(timezone.utc).isoformat()

def emit(*x):print(stamp(),*x,flush=True)

def hold(out,stage,why,extra=None):
    d={'stage':stage,'status':'HOLD','reason':why,'time_utc':stamp()}
    if extra:d.update(extra)
    jd(Path(out)/f'{stage}_GATE.json',d)
    emit(stage,'HOLD',why)
    return d

def output_report(out,stage,status,**kwargs):
    d=dict(stage=stage,status=status,time_utc=stamp(),**kwargs)
    jd(Path(out)/f'{stage}_GATE.json',d)
    emit(stage,status)
    return d

def wquantile(values,weights,qs):
    o=np.argsort(values)
    v=np.asarray(values)[o];w=np.asarray(weights)[o]
    cw=np.cumsum(w,dtype=float)
    return np.interp(qs,cw/cw[-1],v).tolist()
