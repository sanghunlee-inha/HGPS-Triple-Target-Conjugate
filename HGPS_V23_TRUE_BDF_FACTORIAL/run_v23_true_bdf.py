#!/usr/bin/env python3
"""V23 actual original V21/V22 BDF 3-factor full-factorial deep study.

No alternate ODE is defined here. All integrals are performed by unmodified
V21 run_untreated/run_lonafarnib/run_conjugate functions.  Sources are strictly
sha256 locked.  The design is the original 65,536 per-seed V22 Sobol design,
subsampled by fixed indices without changing any input coordinate.
Synthetic hypothesis experiment, NOT clinical/patient-level causal inference.
"""
from __future__ import annotations
import argparse, concurrent.futures, gzip, hashlib, json, os, pathlib, platform, sqlite3, sys, time, traceback, zlib
from collections import defaultdict

HERE=pathlib.Path(__file__).resolve().parent
SOURCES=HERE/'source_locked'
SEEDS=(11,23,37,51,73)
MODES=('independent','sequential_p1_tbki_mir','sequential_mir_tbki_p1')
MASKS=tuple(range(8))
TOTAL_V22_PER_SEED=65536
HORIZONS=(182.625,365.25,730.5)
REQUIRED_SHA={'v21_core.py':'fe9b814ca3342b7448f4f1dde23437b3e2287ddeb8748c2cdb284b83117b8cf8',
              'v22_core.py':'6c55d4468b2f1e75a9b3fdace4e5698109f82836c5bd35c5a1155f6439ebd840'}
# Worker globals are initialized only once per process.  On Windows, spawned
# workers are independent; original model modules import after hash checks.
CTX={}

class Hold(RuntimeError):pass

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def read_json(path):return json.loads(pathlib.Path(path).read_text(encoding='utf-8-sig'))

def atomic_json(path,obj):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.writing')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,sort_keys=True),encoding='utf-8')
 os.replace(tmp,path)

def log(event,**kwargs):
 print('V23_'+event+'='+json.dumps(kwargs,ensure_ascii=False,sort_keys=True,allow_nan=False),flush=True)

def source_verify():
 hashes=read_json(SOURCES/'ORIGINAL_SOURCE_HASHES.json')
 for name,expect in hashes.items():
  if sha(SOURCES/name)!=expect:raise Hold('HOLD_BUNDLED_ORIGINAL_SOURCE_CHANGED:'+name)
 for name,expect in REQUIRED_SHA.items():
  if hashes.get(name)!=expect:raise Hold('HOLD_ORIGINAL_SOURCE_MISMATCH:'+name)
 sys.path.insert(0,str(SOURCES)) if str(SOURCES) not in sys.path else None
 import v21_core as v21
 import v22_core as v22
 if pathlib.Path(v21.__file__).resolve()!=SOURCES/'v21_core.py' or pathlib.Path(v22.__file__).resolve()!=SOURCES/'v22_core.py':
  raise Hold('HOLD_WRONG_ORIGINAL_MODULE_IMPORTED')
 return v21,v22,hashes

def locate_base(base):
 names=('V16_HGPS_PAIRED_DEEP_COMPUTE','V18_LONAFARNIB_APPARENT_PK_BRIDGE',
        'V19_MASS_BALANCE_TOX_INDEPENDENCE','V20_CONJUGATE_RELEASE_MASS_BALANCE')
 user=pathlib.Path.home()
 candidates=[pathlib.Path(base)] if base else []
 candidates += [user/'HGPS_TOX_CODE',user/'Desktop'/'２６。０８。３０ 소아조로증 ＡＩ 신약개발 연구']
 # Explicit root must be respected; no guessing modules from unrelated runs.
 for p in candidates:
  if p.is_dir() and all((p/n).is_dir() for n in names):return p.resolve()
 raise Hold('HOLD_BASE_MODULES_NOT_FOUND: --base must contain exactly '+','.join(names)+' ; checked '+', '.join(map(str,candidates)))

def find_protocol(base,explicit):
 if explicit:
  p=pathlib.Path(explicit).resolve()
  if not p.is_file():raise Hold('HOLD_PROTOCOL_NOT_FOUND:'+str(p))
  return p
 # Only select a verified protocol with unique bytes; never synthesize doses.
 candidates=[]
 for directory in (base/'V21_PAIRED_CONDITIONAL_VIRTUAL_TRIAL',base/'V21_HGPS_PAIRED_CONDITIONAL_VIRTUAL_TRIAL',
                   base/'V22_HGPS_MULTISCALE_RESEARCH',base):
  if not directory.is_dir():continue
  # Flat and one-level protocol files, to avoid scanning huge checkpoints.
  files=list(directory.glob('*PROTOCOL*.json'))+list(directory.glob('*protocol*.json'))
  for sub in directory.iterdir():
   if sub.is_dir() and any(x in sub.name.lower() for x in ('protocol','config','v21')):
    files.extend(sub.glob('*.json'))
  for p in files:
   if not p.is_file() or p.stat().st_size>1_000_000:continue
   try:
    j=read_json(p)
    if j.get('scope')=='SYNTHETIC_CONDITIONAL_NOT_CLINICAL' and 'dose_schedules_nmol' in j:
     candidates.append(p.resolve())
   except (OSError,UnicodeError,ValueError,AttributeError):pass
 uniq={sha(p):p for p in candidates}
 if len(uniq)==1:return next(iter(uniq.values()))
 if not uniq:raise Hold('HOLD_ORIGINAL_V21_PROTOCOL_NOT_FOUND: supply --protocol original .json')
 raise Hold('HOLD_MULTIPLE_DISTINCT_PROTOCOLS: supply --protocol; matches: '+', '.join(map(str,uniq.values())))

def select_indices(n,seed):
 import numpy as np
 if not 1<=n<=TOTAL_V22_PER_SEED:raise Hold('HOLD_SAMPLES_PER_SEED_RANGE')
 # Prespecified uniform-without-replacement design-index subset, not patients.
 return sorted(int(x) for x in np.random.default_rng(20261112+seed).choice(TOTAL_V22_PER_SEED,n,replace=False))

def axis(mask):return tuple(bool(mask&(1<<i)) for i in range(3))

def factorial_terms(values):
 """Möbius decomposition on the original 3-bit Boolean intervention lattice.
 Values dict maps int mask -> float. Zero is intact conjugate with all payloads off.
 Identifies contrast in this specified model, not empirical biochemical synergy.
 """
 if set(values)!=set(MASKS):raise Hold('HOLD_MISSING_8_FACTORIAL_CONDITIONS')
 f=lambda i:float(values[i])
 return {
  'P1':f(1)-f(0),'TBKI':f(2)-f(0),'MIR4651':f(4)-f(0),
  'P1xTBKI':f(3)-f(1)-f(2)+f(0),
  'P1xMIR4651':f(5)-f(1)-f(4)+f(0),
  'TBKIxMIR4651':f(6)-f(2)-f(4)+f(0),
  'THREE_WAY':f(7)-f(6)-f(5)-f(3)+f(4)+f(2)+f(1)-f(0),
  'ALL_VS_NO_PAYLOAD':f(7)-f(0),
 }

def make_manifest(base,proto,orig_hashes,args,source):
 import numpy as np,scipy
 return dict(software='HGPS_V23_TRUE_BDF_FACTORIAL',model='ORIGINAL_UNCHANGED_V21_V22_BDF',
  scope='SYNTHETIC_QMC_NOT_PATIENT',base_path=str(base),protocol_path=str(proto),
  original_v21_v22_file_sha256=orig_hashes,
  original_dependency_source_sha256=source,
  protocol_sha256=sha(proto),
  seeds=list(SEEDS),original_v22_design_n_per_seed=TOTAL_V22_PER_SEED,
  cases_per_seed=args.cases_per_seed,
  selection_algorithm='numpy.default_rng(20261112+seed).choice(65536,n,replace=False)',
  horizons_days=list(HORIZONS),release_modes=list(MODES),factorial_masks=list(MASKS),
  ablation_convention='axis bits P1,TBKI,MIR4651; parent conjugate PK retained',
  rtol_original=3e-7,atol_original=3e-9,
  refinement_every=args.refinement_every,rtol_refined=3e-9,atol_refined=3e-11,
  python_version=platform.python_version(),numpy_version=np.__version__,scipy_version=scipy.__version__,
  not_measured_treatment_efficacy=True,not_patient_PKPD=True)

def open_db(path):
 db=sqlite3.connect(path,timeout=120)
 db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
 db.execute('PRAGMA busy_timeout=120000')
 db.execute('''CREATE TABLE IF NOT EXISTS runs (
   seed INTEGER NOT NULL, idx INTEGER NOT NULL, scenario_sha TEXT NOT NULL,
   status TEXT NOT NULL CHECK(status IN ('PASS','HOLD')),
   payload BLOB, nfev INTEGER NOT NULL DEFAULT 0,
   seconds REAL NOT NULL DEFAULT 0, error TEXT,
   PRIMARY KEY(seed,idx))''')
 db.commit();return db

def check_v22_db(base,seed,idxs,design):
 orig=base/'V22_HGPS_MULTISCALE_RESEARCH'/'OUTPUT_RESEARCH'/f'V22_SEED_{seed}.sqlite'
 if not orig.is_file():return {'status':'NO_ORIGINAL_DB_AVAILABLE','compared':0}
 con=sqlite3.connect('file:'+orig.as_posix()+'?mode=ro',uri=True,timeout=20)
 try:
  names={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
  if 'scenario' not in names:raise Hold('HOLD_V22_ORIGINAL_DB_SCHEMA:'+str(orig))
  found=0
  for i in idxs:
   row=con.execute('SELECT design_sha FROM scenario WHERE idx=?',(i,)).fetchone()
   if row is None:continue
   if row[0]!=design[i]['scenario_sha256']:
    raise Hold('HOLD_ORIGINAL_V22_DESIGN_DIFFERENT:seed='+str(seed)+' idx='+str(i))
   found+=1
  return {'status':'MATCHED_ORIGINAL_SQLITE' if found else 'NO_OVERLAPPING_COMPLETED_DESIGNS',
          'compared':found,'file':str(orig)}
 finally:con.close()

def _init_worker(base,proto):
 os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
 import numpy as np
 v21,v22,_=source_verify()
 paths=v21.Paths(str(base/'V16_HGPS_PAIRED_DEEP_COMPUTE'),
                   str(base/'V18_LONAFARNIB_APPARENT_PK_BRIDGE'/'OUTPUT'),
                   str(base/'V19_MASS_BALANCE_TOX_INDEPENDENCE'),
                   str(base/'V19_MASS_BALANCE_TOX_INDEPENDENCE'/'OUTPUT'),
                   str(base/'V20_CONJUGATE_RELEASE_MASS_BALANCE'),
                   str(base/'V20_CONJUGATE_RELEASE_MASS_BALANCE'/'OUTPUT_DEEP'))
 model,audit,release,source=v21.load_modules(paths)
 config=v21.load_protocol(proto)
 CTX.update(dict(v21=v21,model=model,audit=audit,release=release,proto=config))

def _record(model,run):
 import numpy as np
 sel=[model.IDX[k] for k in CTX['v21'].OUTPUTS]
 a=run['states'][:,sel]
 if not np.all(np.isfinite(a)) or a.min()<-1e-5:raise Hold('HOLD_STATE_VALUE_INVALID')
 return a.tolist()

def _calc_one(seed,idx,s):
 import numpy as np
 v=CTX['v21'];model=CTX['model'];audit=CTX['audit'];release=CTX['release'];proto=CTX['proto']
 start=time.monotonic()
 p=s['params'];days=730.5
 times=sorted(set((0.,*HORIZONS,*[float(x) for x in np.linspace(0.,days,25)])))
 untreated=v.run_untreated(model,p,days,times)
 lona,ml=v.run_lonafarnib(model,audit,p,proto,days,times,s['volume_ratio'])
 nfev=int(untreated['nfev'])+int(lona['nfev']);refinement=None
 if float(ml['max_abs_parent_relative_residual'])>2e-5:raise Hold('HOLD_LONA_MASS_BALANCE')
 output=dict(seed=seed,idx=idx,scenario_sha256=s['scenario_sha256'],
             pk_candidate_id=s['pk_candidate_id'],times_days=times,outputs=list(v.OUTPUTS),
             untreated=_record(model,untreated),lonafarnib=_record(model,lona),
             lona_nfev=int(lona['nfev']),conjugate={},
             dose_scale_hypothesis=float(s['dose_scale']),
             volume_ratio_hypothesis=float(s['volume_ratio']))
 for mode in MODES:
  runs={}
  for m in MASKS:
   rec,ledger=v.run_conjugate(model,release,p,proto,days,times,s['volume_ratio'],mode,
                             s['dose_scale'],axis(m))
   nfev+=int(rec['nfev'])
   if float(ledger['parent_max_relative'])>2e-5 or max(float(x) for x in ledger['payload_max_relative'].values())>2e-5:
    raise Hold('HOLD_MASS_BALANCE:seed='+str(seed)+' idx='+str(idx)+' mode='+mode+' mask='+str(m))
   # Keep original exposure trajectories to detect PK confounding of axis ablations.
   x=rec['states']
   runs[str(m)]=dict(endpoint_states=_record(model,rec),nfev=int(rec['nfev']),
                     parent_mass_relative=float(ledger['parent_max_relative']),
                     max_payload_relative=max(float(x) for x in ledger['payload_max_relative'].values()),
                     parent_conj_plasma_AUC_nM_day=[float(z) for z in x[:,27]],
                     parent_conj_vascular_AUC_nM_day=[float(z) for z in x[:,-1]],
                     p1_cyt_nM=[float(z) for z in x[:,6]],
                     tbki_cyt_nM=[float(z) for z in x[:,7]],
                     mir_cyt_nM=[float(z) for z in x[:,8]],
                     ago2_loaded_nM=[float(z) for z in x[:,9]])
  output['conjugate'][mode]=runs
 # Only prespecified subset; no parameter or model changes, solver tolerance test separated.
 if CTX.get('refine') and idx%CTX['refine']==0:
  ref_l,_=v.run_lonafarnib(model,audit,p,proto,days,times,s['volume_ratio'],rtol=3e-9,atol=3e-11)
  ref_c,_=v.run_conjugate(model,release,p,proto,days,times,s['volume_ratio'],'independent',
                         s['dose_scale'],axis(7),rtol=3e-9,atol=3e-11)
  nfev+=int(ref_l['nfev'])+int(ref_c['nfev'])
  base=np.r_[np.asarray(output['lonafarnib']),np.asarray(output['conjugate']['independent']['7']['endpoint_states'])]
  fine=np.r_[np.asarray(_record(model,ref_l)),np.asarray(_record(model,ref_c))]
  err=float(np.max(np.abs(base-fine)/np.maximum(1.,np.abs(fine))))
  refinement=dict(max_scaled_abs_error=err,gate='PASS' if err<1e-3 else 'HOLD_NUMERIC_REFINEMENT',
                  original_rtol=3e-7,refined_rtol=3e-9)
 output['refinement']=refinement
 output['nfev']=nfev
 return output,time.monotonic()-start

def worker(job):
 seed,idx,s,refine_every=job
 try:
  CTX['refine']=refine_every
  data,elapsed=_calc_one(seed,idx,s)
  refined_hold=bool(data.get('refinement') and data['refinement']['gate']!='PASS')
  return dict(seed=seed,idx=idx,scenario_sha=data['scenario_sha256'],
              status='HOLD' if refined_hold else 'PASS',
              payload=zlib.compress(json.dumps(data,separators=(',',':'),allow_nan=False).encode('utf-8'),level=6),
              nfev=data['nfev'],seconds=elapsed,
              error='HOLD_NUMERIC_REFINEMENT' if refined_hold else '')
 except Exception as e:
  return dict(seed=seed,idx=idx,scenario_sha=s['scenario_sha256'],status='HOLD',payload=None,
              nfev=0,seconds=0.,error=f'{type(e).__name__}: {e}\n'+traceback.format_exc()[-2000:])

def original_inputs(args):
 v21,v22,hashes=source_verify()
 base=locate_base(args.base)
 proto=find_protocol(base,args.protocol)
 core,paths,model,audit,source,candidates,selection=v22.preflight(str(SOURCES),str(base),str(proto))
 if len(candidates)!=12:raise Hold('HOLD_ORIGINAL_V22_CANDIDATES')
 return v21,v22,hashes,base,proto,model,audit,source,candidates,selection

def run(args):
 import numpy as np
 v21,v22,hashes,base,proto,model,audit,source,candidates,selection=original_inputs(args)
 if args.cases_per_seed<1 or args.cases_per_seed>TOTAL_V22_PER_SEED:raise Hold('HOLD_CASES_PER_SEED')
 if args.workers<1 or args.workers>(os.cpu_count() or 1):raise Hold('HOLD_WORKERS_OUT_OF_RANGE')
 if args.refinement_every<0:raise Hold('HOLD_REFINEMENT_FREQUENCY')
 out=pathlib.Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
 manifest=make_manifest(base,proto,hashes,args,source)
 lock_path=out/'V23_EXPERIMENT_LOCK.json'
 if lock_path.is_file():
  if read_json(lock_path)!=manifest:raise Hold('HOLD_EXISTING_RUN_SOURCE_OR_CONFIGURATION_MISMATCH')
 else:atomic_json(lock_path,manifest)
 db=open_db(out/'V23_DIRECT_BDF.sqlite')
 expected_total=len(SEEDS)*args.cases_per_seed
 start=time.monotonic()
 log('DEEP_START',n_synthetic_cases=expected_total,per_case_conjugate_BDF=24,
     per_case_lonafarnib_BDF=1,per_case_untreated_BDF=1,
     n_minimum_true_BDF_integrations=expected_total*26,workers=args.workers,
     old_model_sha=hashes['v21_core.py'],out=str(out))
 try:
  for seed in SEEDS:
   # Existing V22 design domain and 65,536 index universe unchanged.
   design=v22.seed_design(v21,model,audit,candidates,TOTAL_V22_PER_SEED,seed)
   idxs=select_indices(args.cases_per_seed,seed)
   audit_row=check_v22_db(base,seed,idxs,design)
   log('SOURCE_LINK',seed=seed,**audit_row)
   # Exact key + design SHA audit on every restart.
   previous={int(idx):(name,status) for idx,name,status in db.execute(
    'SELECT idx,scenario_sha,status FROM runs WHERE seed=?',(seed,))}
   if any(i not in idxs or v[0]!=design[i]['scenario_sha256'] for i,v in previous.items()):
    raise Hold('HOLD_V23_CHECKPOINT_DESIGN_SHA_INVALID:seed='+str(seed))
   pending=[i for i in idxs if i not in previous or (args.retry_failed and previous[i][1]=='HOLD')]
   log('SEED_START',seed=seed,previous=len(previous),to_run=len(pending))
   jobs=iter((seed,i,design[i],args.refinement_every) for i in pending)
   # Always submit at most 2 * workers jobs, avoiding hundreds of thousands
   # of queued objects and allowing a graceful stop/resume.
   failed_this_seed=0
   with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
          initializer=_init_worker,initargs=(base,proto)) as pool:
    running={}
    def submit():
     try:job=next(jobs)
     except StopIteration:return False
     fut=pool.submit(worker,job);running[fut]=job[1];return True
    for _ in range(min(2*args.workers,len(pending))):submit()
    last=0.;stop=False
    while running:
     done,_=concurrent.futures.wait(tuple(running),timeout=5,
                           return_when=concurrent.futures.FIRST_COMPLETED)
     for future in done:
      idx=running.pop(future)
      rec=future.result()  # worker crash: abort rather than impute
      if rec['idx']!=idx or rec['scenario_sha']!=design[idx]['scenario_sha256']:
       raise Hold('HOLD_WORKER_RETURN_MISMATCH')
      with db:db.execute('INSERT OR REPLACE INTO runs(seed,idx,scenario_sha,status,payload,nfev,seconds,error) '
                        'VALUES(?,?,?,?,?,?,?,?)',
                        (rec['seed'],rec['idx'],rec['scenario_sha'],rec['status'],rec['payload'],
                         rec['nfev'],rec['seconds'],rec['error']))
      if rec['status']!='PASS':
       failed_this_seed+=1
       log('PHYSICS_HOLD',seed=seed,idx=idx,error=rec['error'][:400])
       # Abort a failing input/model lock rather than burn days on invalid trajectories.
       if failed_this_seed>=5:
        raise Hold('HOLD_EARLY_FAILURE_LIMIT_5: review first failed cases in V23_DIRECT_BDF.sqlite')
     if (out/'REQUEST_STOP.flag').exists():stop=True
     if not stop:
      while len(running)<2*args.workers and submit():pass
     now=time.monotonic()
     if now-last>=15 or (not running):
      stat=db.execute('SELECT status,COUNT(*),SUM(nfev) FROM runs GROUP BY status').fetchall()
      progress=dict(completed=sum(int(x[1]) for x in stat),requested=expected_total,
        successful=sum(int(x[1]) for x in stat if x[0]=='PASS'),
        held=sum(int(x[1]) for x in stat if x[0]=='HOLD'),
        real_BDF_nfev=sum(int(x[2] or 0) for x in stat),
        elapsed_hours=round((now-start)/3600,4),seed=seed,
        active=len(running),paused=stop)
      atomic_json(out/'V23_LIVE_PROGRESS.json',progress);log('PROGRESS',**progress)
      last=now
    if stop:
     log('PAUSED',seed=seed,reason='REQUEST_STOP.flag detected; remove and run same command to resume')
     return 75
   del design
  results=db.execute('SELECT status,COUNT(*) FROM runs GROUP BY status').fetchall()
  n_pass=sum(int(n) for s,n in results if s=='PASS')
  n_hold=sum(int(n) for s,n in results if s=='HOLD')
  nfev=db.execute('SELECT COALESCE(SUM(nfev),0) FROM runs').fetchone()[0]
  gate=dict(status='PASS_DIRECT_BDF_FULL_FACTORIAL_SYNTHETIC_ONLY' if n_pass==expected_total and n_hold==0 else 'HOLD_INCOMPLETE_OR_FAILED',
    total_design_cases=expected_total,completed_pass=n_pass,completed_hold=n_hold,
    planned_conjugate_BDF_integrations=24*expected_total,
    planned_bdf_integrations=26*expected_total,
    real_BDF_rhs_evaluations=nfev,source_locked=True,
    original_ODE_unchanged=True,not_patient_prediction=True,
    not_clinical_efficacy=True,not_empirical_synergy=True,
    time_elapsed_hours=(time.monotonic()-start)/3600)
  # Hold if any failed refinement, even when each integration returned finite.
  fail_ref=0
  for (blob,) in db.execute("SELECT payload FROM runs WHERE status='PASS'"):
   data=json.loads(zlib.decompress(blob))
   if data['refinement'] and data['refinement']['gate']!='PASS':fail_ref+=1
  gate['refinement_hold_count']=fail_ref
  if fail_ref:gate['status']='HOLD_NUMERICAL_REFINEMENT_FAILED'
  atomic_json(out/'V23_DIRECT_BDF_FINAL_GATE.json',gate);log('FINAL',**gate)
  return 0 if gate['status'].startswith('PASS_') else 2
 finally:db.close()

def selftest():
 f={m: 3.+(2.*bool(m&1))+(4.*bool(m&2))+(6.*bool(m&4))+
    (7.*bool(m&1)*bool(m&2))+(11.*bool(m&1)*bool(m&4))+
    (13.*bool(m&2)*bool(m&4))+(17.*bool(m&1)*bool(m&2)*bool(m&4)) for m in MASKS}
 a=factorial_terms(f)
 for k,v in [('P1',2),('TBKI',4),('MIR4651',6),('P1xTBKI',7),('P1xMIR4651',11),('TBKIxMIR4651',13),('THREE_WAY',17)]:
  assert abs(a[k]-v)<1e-12,(k,a[k],v)
 assert len(select_indices(128,11))==128
 assert select_indices(128,11)==select_indices(128,11)
 try:factorial_terms({0:1.})
 except Hold:pass
 else:raise AssertionError('missing factorial condition not held')
 _,_,hashes=source_verify()
 print('SELFTEST_PASS original_sources_verified='+str(len(hashes))+' factorial_interactions=7')


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--base',default='',help='original root with V16,V18,V19,V20 directories')
 p.add_argument('--protocol',default='',help='original V21 synthetic conditional protocol JSON')
 p.add_argument('--output',default=str(pathlib.Path.home()/'Desktop'/'２６。０８。３０ 소아조로증 ＡＩ 신약개발 연구'/'HGPS_V23_TRUE_BDF_FACTORIAL_RESULTS'))
 p.add_argument('--cases-per-seed',type=int,default=2048,help='2048 x 5 seeds by default; subset of original 65536 x 5')
 p.add_argument('--workers',type=int,default=16)
 p.add_argument('--refinement-every',type=int,default=128,help='numerical tolerance cross-check on selected cases')
 p.add_argument('--retry-failed',action='store_true')
 p.add_argument('--selftest',action='store_true')
 args=p.parse_args()
 try:
  if args.selftest:selftest();return 0
  return run(args)
 except Exception as e:
  log('ERROR',message=f'{type(e).__name__}: {e}')
  traceback.print_exc()
  return 1

if __name__=='__main__':sys.exit(main())
