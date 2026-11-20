"""V22 independent randomized-QMC, real V21 BDF matched-trial ensemble.

Never clinical patient prediction. No CPU/GPU padding. Database is a durable
single-writer checkpoint; independent design seeds are not patient replicates.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json, math, os, signal, sqlite3, sys, time, zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

SEEDS=(11,23,37,51,73)
MODES=('independent','sequential_p1_tbki_mir','sequential_mir_tbki_p1')
HERE=Path(__file__).resolve().parent

class Hold(RuntimeError):pass

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for part in iter(lambda:f.read(1<<20),b''):h.update(part)
 return h.hexdigest()

def write_atomic(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.new')
 tmp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8')
 os.replace(tmp,path)

def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def log(tag,**kwargs):print('V22_'+tag+'='+json.dumps(kwargs,sort_keys=True,allow_nan=False),flush=True)

def preflight(v21_dir,base,protocol):
 root=Path(v21_dir).resolve();base=Path(base).resolve()
 if not (root/'v21_core.py').is_file():raise Hold('HOLD_V21_CORE_NOT_FOUND')
 if str(root) not in sys.path:sys.path.insert(0,str(root))
 import v21_core as core
 if core.HORIZONS!=(182.625,365.25,730.5) or core.SEEDS!=SEEDS or core.RELEASES!=MODES:
  raise Hold('HOLD_V21_API_MISMATCH')
 paths=core.Paths(str(base/'V16_HGPS_PAIRED_DEEP_COMPUTE'),
  str(base/'V18_LONAFARNIB_APPARENT_PK_BRIDGE'/'OUTPUT'),
  str(base/'V19_MASS_BALANCE_TOX_INDEPENDENCE'),
  str(base/'V19_MASS_BALANCE_TOX_INDEPENDENCE'/'OUTPUT'),
  str(base/'V20_CONJUGATE_RELEASE_MASS_BALANCE'),
  str(base/'V20_CONJUGATE_RELEASE_MASS_BALANCE'/'OUTPUT_DEEP'))
 model,audit,release,source=core.load_modules(paths)
 proto=core.load_protocol(protocol)
 if proto['dose_schedules_nmol']['lonafarnib'].get('dose_mode')!='FDA_LABEL_REFERENCE_BSA_1m2':
  raise Hold('HOLD_REQUIRE_LABEL_REFERENCE_FOR_V22')
 if source.get('v16_model_sha256')!='fa1869e0773cc249f33b4dc73b31f3e6df41c578a2f3912a2ab544a44b5b31c1':
  raise Hold('HOLD_V16_ORIGINAL_MODEL_HASH')
 candidates,selection=core.candidate_records(audit,paths,12)
 if len(candidates)!=12:raise Hold('HOLD_12_LONA_CANDIDATES')
 return core,paths,model,audit,source,candidates,selection

def connect(path):
 db=sqlite3.connect(path,timeout=120)
 db.execute('PRAGMA journal_mode=WAL')
 db.execute('PRAGMA synchronous=FULL')
 db.execute('PRAGMA busy_timeout=120000')
 db.execute('CREATE TABLE IF NOT EXISTS scenario (idx INTEGER PRIMARY KEY, design_sha TEXT NOT NULL, blob BLOB NOT NULL, nfev INTEGER NOT NULL)')
 db.commit();return db

def pack_result(results,include_timeline):
 if len(results)!=3 or {x['mode'] for x in results}!=set(MODES):raise Hold('HOLD_MISSING_RELEASE_ARM')
 slim=[]
 for x in results:
  if not x['rows']:raise Hold('HOLD_EMPTY_HORIZON')
  rows=sorted(x['rows'],key=lambda r:float(r['horizon_days']))
  if any(float(rows[i]['horizon_days']) not in (182.625,365.25,730.5) for i in range(len(rows))) and len(rows)>1:
   # Pilot has just the terminal day; accepted by pilot path.
   raise Hold('HOLD_UNEXPECTED_HORIZON')
  slim.append(dict(mode=x['mode'],rows=rows,knockout=x['knockout'],refinement=x['refinement'],
    timeline=x['monthly'] if include_timeline else [],nfev=x['nfev'],
    nfev_primary=x['nfev_primary'],nfev_ablations=x['nfev_ablations'],
    nfev_refinement=x['nfev_refinement']))
 return zlib.compress(json.dumps(slim,separators=(',',':'),allow_nan=False).encode(),level=5)

def unpack(blob):return json.loads(zlib.decompress(blob))

def count(db):return int(db.execute('SELECT COUNT(*) FROM scenario').fetchone()[0])

def state_lock(out,required):
 p=out/'V22_INPUT_LOCK.json'
 if p.exists():
  if load_json(p)!=required:raise Hold('HOLD_RESUME_CONFIGURATION_OR_SOURCE_CHANGED')
 else:write_atomic(p,required)

def save_case(db,idx,sha,result):
 blob=pack_result(result,include_timeline=(idx%64==0))
 nfev=sum(int(r['nfev']) for r in result)
 with db:
  db.execute('INSERT INTO scenario (idx,design_sha,blob,nfev) VALUES(?,?,?,?)',(idx,sha,blob,nfev))
 return nfev

def seed_design(core,model,audit,candidates,n,seed):
 # Every seed is an independent scrambled Sobol design over the same declared
 # synthetic-hypothesis domain, NEVER observed patient variation.
 return core.synthetic_design(model,audit,candidates,n,20261104+seed*10009)

def run(base,v21,out,protocol,*,n_per_seed=65536,workers=18,
        knockout_every=256,refine_every=512,test_days=None):
 if n_per_seed<2 or n_per_seed&(n_per_seed-1):raise Hold('HOLD_CASES_PER_SEED_MUST_BE_POWER_OF_TWO_AT_LEAST_2')
 if workers<1 or workers>(os.cpu_count() or 1):raise Hold('HOLD_INVALID_WORKER_COUNT')
 if knockout_every<0 or refine_every<0:raise Hold('HOLD_INVALID_TEST_CADENCE')
 days=730.5 if test_days is None else float(test_days)
 if test_days is not None and not (0<days<=14.):raise Hold('HOLD_TEST_DAYS_14_OR_LESS')
 out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
 core,paths,model,audit,source,candidates,selection=preflight(v21,base,protocol)
 lock=dict(software='V22_HGPS_MULTISCALE_RESEARCH',n_per_seed=n_per_seed,seed_list=list(SEEDS),
  days=days,protocol_sha256=digest(protocol),v21_core_sha256=digest(Path(v21)/'v21_core.py'),
  v22_core_sha256=digest(__file__),v22_source_hashes={x:digest(HERE/x) for x in ('v22_analyze.py','v22_vascular.py','v22_train.py')},
  source_sha256=source,knockout_every=knockout_every,refine_every=refine_every,
  selection=selection,synthetic_qmc_not_patient=True)
 state_lock(out,lock)
 total=5*n_per_seed
 first=time.monotonic()
 stats=[]
 log('START',seed_list=list(SEEDS),biology_scenarios_total=total,release_comparisons=total*3,
     experiment='REAL_24M_BDF_INDEPENDENT_SCRAMBLED_SOBOL' if test_days is None else 'TEST_ONLY_14DAY_BDF',
     workers=workers,knockout_every=knockout_every,refine_every=refine_every)
 for seed in SEEDS:
  design=seed_design(core,model,audit,candidates,n_per_seed,seed)
  db=connect(out/f'V22_SEED_{seed}.sqlite')
  existing={int(i):s for i,s in db.execute('SELECT idx,design_sha FROM scenario')}
  if any(i<0 or i>=n_per_seed or design[i]['scenario_sha256']!=sha for i,sha in existing.items()):
   raise Hold('HOLD_EXISTING_DB_DESIGN_SHA_MISMATCH')
  if len(existing)>n_per_seed:raise Hold('HOLD_DB_TOO_MANY_ROWS')
  remaining=(i for i in range(n_per_seed) if i not in existing)
  done=len(existing);nfev_all=int(db.execute('SELECT COALESCE(SUM(nfev),0) FROM scenario').fetchone()[0])
  running={};st=time.monotonic();last=st
  log('SEED_START',seed=seed,completed_from_checkpoint=done,remaining=n_per_seed-done)
  def make_job(i):
   s=design[i]
   k=(knockout_every>0 and i%knockout_every==0)
   r=(refine_every>0 and i%refine_every==0)
   return (paths,str(Path(protocol).resolve()),s,days,k,r)
  with ProcessPoolExecutor(max_workers=workers) as pool:
   def submit_one():
    try:i=next(remaining)
    except StopIteration:return False
    running[pool.submit(core.paired_scenario_job,make_job(i))]=i
    return True
   for _ in range(min(2*workers,n_per_seed-done)):submit_one()
   pause=False
   while running:
    ready,_=wait(tuple(running),timeout=10,return_when=FIRST_COMPLETED)
    for fut in ready:
     i=running.pop(fut)
     result=fut.result()  # Every failure aborts; NEVER fabricate or impute.
     nfev_all+=save_case(db,i,design[i]['scenario_sha256'],result)
     done+=1
    if (out/'REQUEST_STOP.flag').exists():pause=True
    if not pause:
     while len(running)<2*workers and submit_one():pass
    now=time.monotonic()
    if now-last>=10 or done==n_per_seed or pause:
     rec=dict(seed=seed,completed=done,total_seed=n_per_seed,all_seeds_total=total,
       elapsed_seconds=round(now-st,1),real_BDF_rhs_evaluations=nfev_all,
       in_flight=len(running),paused=pause)
     write_atomic(out/'LIVE_PROGRESS.json',rec)
     log('PROGRESS',**rec)
     last=now
    # Stop gracefully *after* all running tasks finished; retain database.
  db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
  db.close()
  if pause:
   write_atomic(out/'V22_PAUSED.json',dict(seed=seed,completed=done,pending=n_per_seed-done,
      reason='USER_REQUESTED_STOP_REMOVE_FLAG_BEFORE_RESUME'))
   log('PAUSED',seed=seed,completed=done)
   return {'status':'PAUSED','seed':seed,'completed':done}
  stats.append(dict(seed=seed,completed=done,nfev=nfev_all,seconds=round(time.monotonic()-st,2)))
  log('SEED_COMPLETE',**stats[-1])
 gate=dict(status='PASS_5_INDEPENDENT_QMC_DESIGNS_BDF_NUMERICS_NOT_CLINICAL' if test_days is None else 'PASS_TEST_ONLY_NOT_DEEP',
   n_biologies=total,n_real_paired_BDF=total*3,n_independent_scrambled_QMC_designs=5,
   design_sizes=[n_per_seed]*5,source_lock_sha256=digest(out/'V22_INPUT_LOCK.json'),
   per_seed=stats,not_patient_effect=True,not_CFD=True,not_empirical_posterior=True,
   clinical_approval=False,elapsed_hours=round((time.monotonic()-first)/3600,5))
 write_atomic(out/'V22_BDF_FINAL_GATE.json',gate)
 log('FINAL',**{k:gate[k] for k in ('status','n_biologies','n_real_paired_BDF','elapsed_hours')})
 return gate

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--v21',required=True)
 p.add_argument('--output',required=True);p.add_argument('--protocol',required=True)
 p.add_argument('--cases-per-seed',type=int,default=65536);p.add_argument('--workers',type=int,default=18)
 p.add_argument('--knockout-every',type=int,default=256);p.add_argument('--refine-every',type=int,default=512)
 p.add_argument('--test-days',type=float,default=None)
 a=p.parse_args()
 try:
  result=run(a.base,a.v21,a.output,a.protocol,n_per_seed=a.cases_per_seed,
   workers=a.workers,knockout_every=a.knockout_every,refine_every=a.refine_every,test_days=a.test_days)
  if result['status']=='PAUSED':sys.exit(75)
 except Exception as exc:
  log('HOLD',reason=repr(exc));raise
