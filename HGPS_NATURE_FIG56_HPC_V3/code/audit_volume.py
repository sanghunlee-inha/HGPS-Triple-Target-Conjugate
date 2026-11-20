#!/usr/bin/env python3
"""Numeric candidate tetra integrity audit; no mutation of geometry."""
import json, sys, warnings
from pathlib import Path
import numpy as np
import meshio

root=Path(sys.argv[1])
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    mesh=meshio.read(root/'mmg_repaired.mesh')
tri=np.concatenate([np.asarray(b.data,dtype=np.int64) for b in mesh.cells if b.type=='triangle'])
tets=np.concatenate([np.asarray(b.data,dtype=np.int64) for b in mesh.cells if b.type=='tetra'])
p=np.asarray(mesh.points[:,:3],dtype=np.float64)
if not np.isfinite(p).all(): raise RuntimeError('Non-finite vertices')
neg=zero=pos=0; tot=0.; minabs=float('inf')
for a in range(0,len(tets),120000):
    q=p[tets[a:a+120000]]
    v=np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0])/6.
    neg+=int((v<0).sum());zero+=int((v==0).sum());pos+=int((v>0).sum())
    tot+=float(np.sum(np.abs(v)));minabs=min(minabs,float(np.min(np.abs(v))))
vs={'positive':pos,'negative':neg,'zero':zero,'absolute_volume':tot,'minimum_absolute_tetra_volume':minabs}
audit=json.loads((root/'BOUNDARY_ORIENTATION_AUDIT.json').read_text(encoding='utf-8'))
if audit.get('preconversion_surface_gate')!='PASS' or neg or zero: raise RuntimeError('HOLD: surface or tetra signed-volume gate failed')
record={'optimized_volume_stats':vs,'optimized_tetra':len(tets),'optimized_triangle':len(tri),'gate':'PASS_SIGNED_VOLUMES_WITH_SURFACE_GATE'}
(root/'STRICT_GEOMETRY_AUDIT.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
