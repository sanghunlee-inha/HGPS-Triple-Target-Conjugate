#!/usr/bin/env python3
"""Optional real-source GPU regression sensitivity, NOT causal or clinical PK/PD validation.
Required CSV: seed,PRNP_modulation,TBK1_modulation,SRSF1_modulation,vascular_PD_endpoint
Runs leave-one-seed-out evaluation; no synthetic samples or forced GPU load.
"""
import argparse, csv, json, hashlib
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--threads',type=int,default=24)
a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
req=['seed','PRNP_modulation','TBK1_modulation','SRSF1_modulation','vascular_PD_endpoint']
with a.input.open('r',encoding='utf-8-sig',newline='') as h:
    rd=csv.DictReader(h)
    if any(k not in (rd.fieldnames or []) for k in req):raise SystemExit('HOLD: Missing source fields '+str(req))
    rows=list(rd)
if not rows:raise SystemExit('HOLD: Empty mechanistic input')
x=np.asarray([[float(row[k]) for k in req[1:]] for row in rows],dtype=np.float64)
seeds=np.asarray([row['seed'] for row in rows]); unique=np.unique(seeds)
if len(unique)<3:raise SystemExit('HOLD: at least three independently identified seed groups required')
if not np.all(np.isfinite(x)):raise SystemExit('HOLD: nonfinite mechanistic data')
if np.min(np.std(x[:,:3],axis=0))==0:raise SystemExit('HOLD: mechanistic feature has zero global variance')
import torch
if not torch.cuda.is_available():raise SystemExit('HOLD: CUDA not available. CPU-only analysis is supported by fig56_analysis.py')
torch.set_num_threads(max(1,a.threads))
torch.backends.cuda.matmul.allow_tf32=False
device='cuda:0'
output=[]
for seed in unique:
    train=seeds!=seed;test=seeds==seed
    if train.sum()<8 or test.sum()<2:raise SystemExit('HOLD: insufficient independent train/test seed data')
    tr=x[train];te=x[test]
    mu=tr[:,:3].mean(0);sd=tr[:,:3].std(0)
    if np.any(sd<1e-12):raise SystemExit('HOLD: constant feature within train fold')
    tx=np.column_stack([np.ones(train.sum()),(tr[:,:3]-mu)/sd]).astype('float32')
    ty=tr[:,3].astype('float32')
    vx=np.column_stack([np.ones(test.sum()),(te[:,:3]-mu)/sd]).astype('float32')
    vx_gpu=torch.from_numpy(vx).to(device)
    X=torch.from_numpy(tx).to(device); y=torch.from_numpy(ty).to(device)
    # Regularized estimation of descriptive source relationships, no drug-to-wall inference.
    lam=1e-3
    A=X.T@X;A[1:,1:] += torch.eye(3,device=device)*lam
    B=X.T@y
    coefs=torch.linalg.solve(A,B)
    pred=(vx_gpu@coefs).cpu().numpy().astype('float64')
    yy=te[:,3]
    rmse=float(np.sqrt(np.mean((yy-pred)**2)))
    denom=float(np.sum((yy-yy.mean())**2))
    r2=1-float(np.sum((yy-pred)**2))/denom if denom>0 else None
    output.append({'test_seed':str(seed),'n_train':int(train.sum()),'n_test':int(test.sum()),
                   'rmse':rmse,'r2':r2,'model_coefficients_standardized':coefs.cpu().numpy().astype('float64').tolist()})
import csv as csvmod
with (a.out/'GPU_SEED_HOLDOUT_SENSITIVITY.csv').open('w',encoding='utf-8-sig',newline='') as h:
    w=csvmod.DictWriter(h,fieldnames=['test_seed','n_train','n_test','rmse','r2','model_coefficients_standardized']);w.writeheader();w.writerows(output)
sha=hashlib.sha256(a.input.read_bytes()).hexdigest()
(a.out/'GPU_EVIDENCE_GATE.json').write_text(json.dumps({'gate':'PASS_NUMERIC_SOURCE_SENSITIVITY_NOT_CAUSAL',
    'source_sha256':sha,'n_samples':len(rows),'n_independent_seed_groups':len(unique),
    'gpu_name':torch.cuda.get_device_name(0),'gpu_enabled':True,
    'intervention_contribution_inferred':False,'drug_specific_clinical_efficacy_inferred':False,
    'source_results':output},indent=2,allow_nan=False),encoding='utf-8')
print('GPU source-based cross-seed sensitivity completed; no clinical efficacy inference.')
