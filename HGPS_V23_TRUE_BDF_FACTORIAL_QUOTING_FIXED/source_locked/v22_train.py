"""V22 real GPU learning of independent new BDF labels, with held-out QMC seed.

Five actual independently trained neural emulators. Each test seed is never
used for training, early stopping, feature normalization or model selection.
No emulated outcomes are substituted for paired BDF results or clinical data.
"""
from __future__ import annotations
import argparse, json, math, os, sqlite3, subprocess, time
from pathlib import Path
import numpy as np
from v22_core import (SEEDS,MODES,preflight,seed_design,unpack,load_json,write_atomic,
                      digest,log,Hold)

REQUIRED_HORIZONS=(182.625,365.25,730.5)

def make_dataset(base,v21,output,protocol):
 out=Path(output).resolve()
 gate=load_json(out/'V22_BDF_FINAL_GATE.json')
 if gate.get('status')!='PASS_5_INDEPENDENT_QMC_DESIGNS_BDF_NUMERICS_NOT_CLINICAL':
  raise Hold('HOLD_ONLY_FULL_24M_PHYSICS_CAN_TRAIN')
 core,paths,model,audit,source,candidates,sel=preflight(v21,base,protocol)
 n=int(gate['design_sizes'][0]);dim=len(model.REQUIRED)+2+3;ydim=len(core.OUTPUTS)*3
 if n<256:raise Hold('HOLD_MINIMUM_256_BIOLOGIES_PER_INDEPENDENT_QMC_SEED')
 X=np.empty((n*15,dim),dtype=np.float32);Y=np.empty((n*15,ydim),dtype=np.float32)
 split=np.empty(n*15,np.int16);group=np.empty(n*15,np.int32)
 rownum=0
 for seed in SEEDS:
  design=seed_design(core,model,audit,candidates,n,seed)
  db=sqlite3.connect(out/f'V22_SEED_{seed}.sqlite')
  sz=db.execute('SELECT COUNT(*) FROM scenario').fetchone()[0]
  if sz!=n:raise Hold('HOLD_SEED_DATA_INCOMPLETE')
  for idx,sha,blob in db.execute('SELECT idx,design_sha,blob FROM scenario ORDER BY idx'):
   d=design[idx]
   if sha!=d['scenario_sha256']:raise Hold('HOLD_DESIGN_RECONSTRUCTION_SHA_MISMATCH')
   xx=np.r_[np.log1p(np.array([d['params'][k] for k in model.REQUIRED],dtype=np.float64)),
             np.log(d['volume_ratio']),np.log(d['dose_scale'])].astype(np.float32)
   results=unpack(blob)
   if {r['mode'] for r in results}!=set(MODES):raise Hold('HOLD_MISSING_RELEASE_MODE')
   bymode={r['mode']:r for r in results}
   for mid,mode in enumerate(MODES):
    rec=bymode[mode]
    rows=sorted(rec['rows'],key=lambda r:float(r['horizon_days']))
    if len(rows)!=3 or tuple(float(x['horizon_days']) for x in rows)!=REQUIRED_HORIZONS:
     raise Hold('HOLD_6_12_24_MONTH_LABELS_REQUIRED')
    vals=np.array([float(row['delta_'+k]) for row in rows for k in core.OUTPUTS],dtype=np.float32)
    if not np.isfinite(vals).all():raise Hold('HOLD_NAN_IN_REAL_BDF_LABELS')
    X[rownum,:-3]=xx;X[rownum,-3:]=np.eye(3,dtype=np.float32)[mid]
    Y[rownum]=vals;split[rownum]=seed;group[rownum]=seed*n+idx
    rownum+=1
  db.close()
 if rownum!=n*15 or not np.isfinite(X).all() or not np.isfinite(Y).all():
  raise Hold('HOLD_DATASET_BUILD_INCOMPLETE')
 return X,Y,split,group,core.OUTPUTS

def partitions(seed,split,group):
 if seed not in SEEDS:raise Hold('HOLD_UNRECOGNIZED_TEST_SEED')
 val_seed=SEEDS[(SEEDS.index(seed)+1)%5]
 tr=np.where((split!=seed)&(split!=val_seed))[0]
 va=np.where(split==val_seed)[0]
 te=np.where(split==seed)[0]
 for x,y in ((tr,va),(tr,te),(va,te)):
  if np.intersect1d(np.unique(group[x]),np.unique(group[y])).size:
   raise Hold('HOLD_GROUP_DATA_LEAKAGE')
 return tr,va,te,val_seed

def monitor():
 try:
  import psutil
  x={'cpu_util_pct':psutil.cpu_percent(interval=None),'ram_pct':psutil.virtual_memory().percent}
 except Exception:x={'cpu_util_pct':None}
 try:
  p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu',
     '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2)
  if p.returncode==0:
   v=p.stdout.strip().splitlines()[0].split(',');x.update(gpu_util_pct=float(v[0]),gpu_vram_mib=float(v[1]),gpu_temp_c=float(v[2]))
 except Exception:x['gpu_monitor']='UNAVAILABLE'
 return x

def train(base,v21,out,protocol,*,device='cuda',max_epochs=100,steps_per_epoch=10000,
          patience=8,batch_size=1024,width=768,learning_rate=2e-4,ckpt_every=2000,
          test_only=False):
 try:import torch
 except ImportError as e:raise Hold('HOLD_PYTORCH_MISSING') from e
 if device=='cuda' and not torch.cuda.is_available():raise Hold('HOLD_CUDA_UNAVAILABLE')
 if not test_only and (max_epochs!=100 or steps_per_epoch!=10000):
  raise Hold('HOLD_EXPECTED_100_EPOCHS_10000_UPDATES_PER_FOLD')
 if min(patience,batch_size,width,ckpt_every)<=0:raise Hold('HOLD_INVALID_HYPERPARAMETER')
 out=Path(out).resolve();X,Y,split,groups,outcomes=make_dataset(base,v21,out,protocol)
 trainout=out/'GPU_V22';trainout.mkdir(exist_ok=True)
 data_fingerprints={f'seed_{s}':digest(out/f'V22_SEED_{s}.sqlite') for s in SEEDS}
 lock=dict(data_fingerprints=data_fingerprints,bdf_gate_sha256=digest(out/'V22_BDF_FINAL_GATE.json'),
  source_sha256=digest(__file__),protocol_sha256=digest(protocol),test_seeds=list(SEEDS),
  epochs=max_epochs,steps_per_epoch=steps_per_epoch,patience=patience,
  batch_size=batch_size,width=width,device=device,learning_rate=learning_rate,
  ckpt_every=ckpt_every,holdout_policy='TEST_ONE_INDEPENDENT_SOBOL_SEED_VALIDATION_ANOTHER_TRAIN_THREE',
  real_BDF_only=True)
 lp=trainout/'TRAIN_LOCK.json'
 if lp.exists() and load_json(lp)!=lock:raise Hold('HOLD_TRAIN_INPUTS_CHANGED')
 if not lp.exists():write_atomic(lp,lock)
 torch.set_num_threads(min(6,os.cpu_count() or 6))
 torch.backends.cudnn.benchmark=False
 results=[];begin=time.monotonic()
 for fold,seed in enumerate(SEEDS):
  tr,va,te,val_seed=partitions(seed,split,groups)
  mu=X[tr].mean(axis=0);std=np.maximum(X[tr].std(axis=0),1e-3)
  xnorm=np.clip((X-mu)/std,-12.,12.).astype(np.float32)
  yraw=np.arcsinh(Y.astype(np.float64))
  ym=yraw[tr].mean(axis=0);ys=np.maximum(yraw[tr].std(axis=0),.02)
  ynorm=np.clip((yraw-ym)/ys,-40.,40.).astype(np.float32)
  xt=torch.from_numpy(xnorm).to(device);yt=torch.from_numpy(ynorm).to(device)
  train_ix=torch.tensor(tr,device=device,dtype=torch.long)
  torch.manual_seed(seed)
  if device=='cuda':torch.cuda.manual_seed_all(seed)
  nn=torch.nn.Sequential(torch.nn.Linear(X.shape[1],width),torch.nn.SiLU(),
       torch.nn.Linear(width,width),torch.nn.SiLU(),torch.nn.Linear(width,Y.shape[1])).to(device)
  opt=torch.optim.AdamW(nn.parameters(),lr=learning_rate,weight_decay=1e-4)
  rng=torch.Generator(device=device);rng.manual_seed(seed+90210)
  ckpt=trainout/f'FOLD_{seed}_CKPT.pt';bestp=trainout/f'FOLD_{seed}_BEST.pt'
  start_epoch=0;start_step=0;updates=0;bad=0;best=float('inf');best_epoch=0
  if ckpt.is_file():
   d=torch.load(ckpt,map_location=device,weights_only=True)
   if d['seed']!=seed:raise Hold('HOLD_BAD_CHECKPOINT_SEED')
   nn.load_state_dict(d['model']);opt.load_state_dict(d['opt'])
   rng.set_state(d['rng']);start_epoch=int(d['epoch']);start_step=int(d['next_step'])
   updates=int(d['updates']);best=float(d['best']);best_epoch=int(d['best_epoch']);bad=int(d['bad'])
   log('RESUME',test_seed=seed,epoch=start_epoch,next_step=start_step,genuine_updates=updates)
  def validation(indices):
   nn.eval();tot=0.;count=0
   with torch.inference_mode():
    for k in range(0,len(indices),4096):
     inds=torch.tensor(indices[k:k+4096],device=device)
     v=nn(xt[inds]);loss=(v-yt[inds]).square().sum()
     tot+=float(loss.cpu());count+=len(inds)*Y.shape[1]
   return tot/count
  def checkpoint(next_epoch,next_step):
   tmp=ckpt.with_suffix('.tmp')
   torch.save(dict(seed=seed,model=nn.state_dict(),opt=opt.state_dict(),
       rng=rng.get_state(),epoch=next_epoch,next_step=next_step,
       updates=updates,best=best,best_epoch=best_epoch,bad=bad),tmp)
   os.replace(tmp,ckpt)
  epoch=start_epoch
  for epoch in range(start_epoch,max_epochs):
   if bad>=patience:break
   nn.train();start=start_step if epoch==start_epoch else 0
   for step in range(start,steps_per_epoch):
    idx=torch.randint(0,len(tr),(batch_size,),generator=rng,device=device)
    xx=xt[train_ix[idx]];yy=yt[train_ix[idx]]
    pred=nn(xx);loss=(pred-yy).square().mean()
    if not torch.isfinite(loss):raise Hold('HOLD_NONFINITE_TRAIN_LOSS')
    opt.zero_grad(set_to_none=True);loss.backward()
    torch.nn.utils.clip_grad_norm_(nn.parameters(),5.);opt.step();updates+=1
    if (step+1)%ckpt_every==0:
     checkpoint(epoch,step+1)
     log('TRAIN_PROGRESS',fold_seed=seed,epoch=epoch+1,step=step+1,
       genuine_optimizer_updates=updates,elapsed_hours=round((time.monotonic()-begin)/3600,4),
       **monitor())
    # A separate PowerShell window can request a graceful stop EVEN IF
    # Ctrl+C fails to reach the CUDA worker. Save the complete AdamW state.
    if (step+1)%200==0 and (out/'REQUEST_STOP.flag').exists():
     checkpoint(epoch,step+1)
     log('GPU_PAUSED',fold_seed=seed,epoch=epoch+1,next_step=step+1,
         genuine_optimizer_updates=updates,all_optimizer_state_preserved=True)
     return {'status':'PAUSED','fold_seed':seed,'genuine_optimizer_updates':updates}
   start_step=0
   val=validation(va)
   if val<best-1e-5:
    best=val;best_epoch=epoch+1;bad=0
    tmp=bestp.with_suffix('.tmp')
    torch.save(nn.state_dict(),tmp);os.replace(tmp,bestp)
   else:bad+=1
   checkpoint(epoch+1,0)
   log('EPOCH',test_seed=seed,validation_seed=val_seed,epoch=epoch+1,validation_loss=val,
     best_validation_loss=best,patience_count=bad,genuine_updates=updates)
   if bad>=patience:
    log('EARLY_STOP',test_seed=seed,reason='NO_HELDOUT_VALIDATION_IMPROVEMENT',genuine_updates=updates)
    break
  if not bestp.is_file():raise Hold('HOLD_BEST_CKPT_MISSING')
  nn.load_state_dict(torch.load(bestp,map_location=device,weights_only=True))
  testnorm=validation(te)
  nn.eval();allpred=[]
  with torch.inference_mode():
   for k in range(0,len(te),4096):
    inds=torch.tensor(te[k:k+4096],device=device)
    allpred.append(nn(xt[inds]).float().cpu().numpy())
  zp=np.concatenate(allpred).astype(np.float64)
  raw_pred=np.sinh(zp*ys+ym)
  real=Y[te].astype(np.float64)
  if not np.isfinite(raw_pred).all():raise Hold('HOLD_NONFINITE_PREDICTIONS')
  train_baseline=np.mean(Y[tr],axis=0)
  baseline_mae=float(np.mean(np.abs(real-train_baseline)))
  test_mae=float(np.mean(np.abs(real-raw_pred)))
  per_outcome={k:float(np.mean(np.abs(real[:,j::len(outcomes)]-raw_pred[:,j::len(outcomes)])))
               for j,k in enumerate(outcomes)}
  r=dict(test_seed=seed,validation_seed=val_seed,train_seeds=[s for s in SEEDS if s not in (seed,val_seed)],
   independent_train_biologies=len(np.unique(groups[tr])),independent_validation_biologies=len(np.unique(groups[va])),
   independent_test_biologies=len(np.unique(groups[te])),true_optimizer_updates=updates,
   best_validation_epoch=best_epoch,test_mse_normalized_asinh=testnorm,
   test_MAE_original_dimensionless_states=test_mae,
   constant_train_mean_baseline_test_MAE=baseline_mae,mae_better_than_naive_baseline=test_mae<baseline_mae,
   test_per_endpoint_MAE=per_outcome,not_clinical_prediction=True)
  results.append(r)
  write_atomic(trainout/f'FOLD_{seed}_EVALUATION.json',r)
  log('FOLD_COMPLETE',**{k:r[k] for k in ('test_seed','true_optimizer_updates','test_MAE_original_dimensionless_states','constant_train_mean_baseline_test_MAE')})
 final=dict(status='PASS_FIVE_HELDOUT_QMC_TEST_FOLDS_NOT_CLINICAL',n_independent_test_folds=len(results),
   max_requested_optimizer_steps=5*max_epochs*steps_per_epoch,
   actual_optimizer_steps=sum(r['true_optimizer_updates'] for r in results),
   all_folds_beat_constant_baseline=all(r['mae_better_than_naive_baseline'] for r in results),
   test_result=results,patient_predictive_validity=False)
 write_atomic(trainout/'V22_GPU_FINAL_GATE.json',final)
 log('GPU_FINAL',folds=len(results),genuine_steps=final['actual_optimizer_steps'],
   all_folds_beat_constant_baseline=final['all_folds_beat_constant_baseline'])
 return final

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for key in ('base','v21','output','protocol'):p.add_argument('--'+key,required=True)
 p.add_argument('--device',default='cuda',choices=('cuda','cpu'))
 p.add_argument('--epochs',type=int,default=100);p.add_argument('--steps',type=int,default=10000)
 p.add_argument('--patience',type=int,default=8);p.add_argument('--batch-size',type=int,default=1024)
 p.add_argument('--width',type=int,default=768);p.add_argument('--ckpt-every',type=int,default=2000)
 p.add_argument('--test-only',action='store_true')
 a=p.parse_args();result=train(a.base,a.v21,a.output,a.protocol,device=a.device,
  max_epochs=a.epochs,steps_per_epoch=a.steps,patience=a.patience,
  batch_size=a.batch_size,width=a.width,ckpt_every=a.ckpt_every,test_only=a.test_only)
 if result.get('status')=='PAUSED':raise SystemExit(75)
