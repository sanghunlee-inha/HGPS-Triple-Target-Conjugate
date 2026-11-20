"""Replicated scrambled-Sobol global sensitivity using the REAL supplied forward solver.
No synthetic surrogate or invented treatment formula. 5 replications, Jansen S1/ST,
bootstrap across independent scramblings, power-of-two N, checkpoint resume.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import importlib.util,json,os,time,hashlib
import numpy as np
import pandas as pd
from scipy.stats import qmc
from core import jd,emit,digest,hold,output_report

_forward=None

def _worker_init(path,func):
    global _forward
    spec=importlib.util.spec_from_file_location('hgps_external_verified_forward',path)
    if spec is None or spec.loader is None:raise RuntimeError('FIG8_FORWARD_MODULE_IMPORT_FAILED')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    _forward=getattr(m,func)

def _worker_eval(arg):
    row,names,metric=arg
    ans=_forward(dict(zip(names,map(float,row))))
    if isinstance(ans,dict):ans=ans[metric]
    val=float(ans)
    if not np.isfinite(val):raise ValueError('FIG8_NONFINITE_FORWARD_RESULT')
    return val

def build_design(low,high,N,seed):
    d=len(low)
    if N<=0 or N&(N-1):raise ValueError('FIG8_SOBOL_N_NOT_POWER_OF_TWO')
    engine=qmc.Sobol(d=2*d,scramble=True,seed=int(seed))
    sample=engine.random_base2(int(np.log2(N)))
    a=low+(high-low)*sample[:,:d]
    b=low+(high-low)*sample[:,d:]
    # rows: A, B, AB_0, ... AB_(d-1)
    design=np.empty((N*(d+2),d),dtype=np.float64)
    design[:N]=a;design[N:2*N]=b
    for j in range(d):
        block=a.copy();block[:,j]=b[:,j]
        design[(2+j)*N:(3+j)*N]=block
    return design

def indices(y,N,d):
    fa=y[:N];fb=y[N:2*N]
    var=np.var(np.r_[fa,fb],ddof=1)
    if not np.isfinite(var) or var<=1e-25:raise ValueError('FIG8_OUTPUT_VARIANCE_TOO_SMALL')
    fst=[];tot=[]
    for j in range(d):
        fab=y[(2+j)*N:(3+j)*N]
        # Jansen (1999): first order vs B; total order vs A.
        fst.append(1-np.mean((fb-fab)**2)/(2*var))
        tot.append(np.mean((fa-fab)**2)/(2*var))
    return np.asarray(fst),np.asarray(tot)

def run(conf_path,out,workers=16):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not conf_path or not Path(conf_path).is_file():return hold(out,'FIG8','UNSPECIFIED_FORWARD_MODEL_AND_PARAMETER_DISTRIBUTIONS')
    conf=json.loads(Path(conf_path).read_text(encoding='utf-8-sig'))
    mod=conf.get('forward_module')
    if not mod or not Path(mod).is_file():return hold(out,'FIG8','FORWARD_MODULE_MISSING')
    if conf.get('source_type')=='SYNTHETIC_TEST_ONLY':
        # Tests are allowed but never research results.
        scope='SYNTHETIC_TEST_ONLY_NOT_RESEARCH'
    elif conf.get('source_type')=='VERIFIED_RESEARCH_FORWARD':
        if not conf.get('model_provenance_file') or not Path(conf['model_provenance_file']).is_file():
            return hold(out,'FIG8','VERIFIED_MODEL_PROVENANCE_REQUIRED')
        scope='RESEARCH_FORWARD_EXECUTED_NUMERICAL_VALIDITY_ONLY'
    else:return hold(out,'FIG8','FORWARD_SOURCE_TYPE_NOT_VERIFIED')
    params=conf.get('parameters') or []
    if not 2<=len(params)<=32:raise ValueError('FIG8_PARAMETERS_REQUIRE_2_TO_32')
    names=[p['name'] for p in params]
    if len(set(names))!=len(names):raise ValueError('FIG8_DUPLICATE_PARAM_NAME')
    low=np.array([float(p['lower']) for p in params]);high=np.array([float(p['upper']) for p in params])
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or not np.all(high>low):raise ValueError('FIG8_BAD_DISTRIBUTION')
    if any(p.get('distribution')!='uniform' for p in params):raise ValueError('FIG8_ONLY_INDEPENDENT_UNIFORM_DISTRIBUTIONS_IMPLEMENTED')
    n=int(conf.get('base_n',1024));reps=int(conf.get('replicates',5))
    if n<256 or n&(n-1) or reps<3:raise ValueError('FIG8_REQUIRE_N_POWER_OF_2_GTE_256_AND_3_REPLICATES')
    metric=conf['output_key']
    source_sha=digest(mod);config_sha=digest(conf_path)
    n_evals=n*(len(names)+2)*reps
    emit('FIG8_TRUE_FORWARD_EVALUATIONS',n_evals,'n',n,'k',len(names),'reps',reps,'workers',workers)
    all_first=[];all_total=[];rep_summ=[]
    import sys
    old_path=sys.path[:]
    try:
        for r in range(reps):
            seed=int(conf.get('seed',20261114))+r*101
            cache=out/f'FIG8_FORWARD_RESULTS_REP_{r+1}.npz'
            if cache.is_file():
                z=np.load(cache)
                if str(z['config_sha'])!=config_sha or str(z['module_sha'])!=source_sha or len(z['y'])!=(len(names)+2)*n:
                    raise ValueError('FIG8_RESTART_CACHE_HASH_OR_SHAPE_MISMATCH')
                y=z['y'];emit('FIG8_RESUME_REP',r+1,'/',reps)
            else:
                design=build_design(low,high,n,seed)
                y=np.empty(len(design),dtype=float)
                start=time.perf_counter();done=0
                import multiprocessing
                # supports Windows spawn; external module must be import-safe.
                with ProcessPoolExecutor(max_workers=workers,initializer=_worker_init,initargs=(str(Path(mod).resolve()),conf.get('forward_callable','evaluate'))) as pool:
                    tasks=((design[i],names,metric) for i in range(len(design)))
                    for i,res in enumerate(pool.map(_worker_eval,tasks,chunksize=int(conf.get('chunksize',16)))):
                        y[i]=res
                        if (i+1)%max(256,len(design)//20)==0 or i+1==len(design):
                            emit('FIG8_PROGRESS',r+1,'/',reps,i+1,'/',len(design),'elapsed_s',round(time.perf_counter()-start,2))
                np.savez_compressed(cache,y=y,config_sha=config_sha,module_sha=source_sha,n=n,seed=seed)
            s,t=indices(y,n,len(names));all_first.append(s);all_total.append(t)
            # convergence diagnostic: N/2 subset of the same independently scrambled design
            half=n//2
            yh=np.r_[y[:half],y[n:n+half],*[y[(2+j)*n:(2+j)*n+half] for j in range(len(names))]]
            sh,th=indices(yh,half,len(names))
            rep_summ.append({'replicate':r+1,'seed':seed,'N':n,'source_sha256':source_sha,
                             'first_order':dict(zip(names,s.tolist())),'total_order':dict(zip(names,t.tolist())),
                             'convergence_abs_change_N_to_N_half_S1':dict(zip(names,np.abs(s-sh).tolist())),
                             'convergence_abs_change_N_to_N_half_ST':dict(zip(names,np.abs(t-th).tolist()))})
            jd(out/f'FIG8_REPLICATE_{r+1}_AUDIT.json',rep_summ[-1])
    finally:sys.path[:]=old_path
    ff=np.vstack(all_first);tt=np.vstack(all_total)
    result=[]
    for i,name in enumerate(names):
        result.append({'parameter':name,'S1_mean':float(ff[:,i].mean()),'ST_mean':float(tt[:,i].mean()),
             'S1_across_rep_sd':float(ff[:,i].std(ddof=1)),
             'ST_across_rep_sd':float(tt[:,i].std(ddof=1)),
             'S1_rep_min':float(ff[:,i].min()),'S1_rep_max':float(ff[:,i].max()),
             'ST_rep_min':float(tt[:,i].min()),'ST_rep_max':float(tt[:,i].max())})
    pd.DataFrame(result).to_csv(out/'FIG8_SOBOL_REPLICATED_S1_ST.csv',index=False,encoding='utf-8-sig')
    summary={'scope':scope,'forward_model':str(mod),'forward_model_sha256':source_sha,
             'config':str(conf_path),'config_sha256':config_sha,'parameter_distribution':'INDEPENDENT_UNIFORM_ONLY',
             'replications':reps,'base_n':n,'forward_calls':n_evals,
             'replicate_results':rep_summ,'summary':result,
             'model_is_clinically_validated':False,
             'uncertainties':'across independent scrambled Sobol replications, not confidence limits for human clinical effects',
             'explicit_limitations':['global sensitivity conditional on provided forward model and prior ranges',
               'model source/PKPD-to-vessel linkage must be independently calibrated','Sobol indices may be negative under Monte Carlo error']}
    jd(out/'FIG8_REPLICATED_GLOBAL_SENSITIVITY_REPORT.json',summary)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':23,'figure.facecolor':'white','savefig.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(max(13,len(names)*1.25),8))
    x=np.arange(len(names));ax.bar(x-.2,ff.mean(axis=0),.4,label='S1');ax.bar(x+.2,tt.mean(axis=0),.4,label='ST')
    ax.errorbar(x-.2,ff.mean(axis=0),yerr=ff.std(axis=0,ddof=1),fmt='none',ecolor='black')
    ax.errorbar(x+.2,tt.mean(axis=0),yerr=tt.std(axis=0,ddof=1),fmt='none',ecolor='black')
    ax.set_xticks(x,names,rotation=35,ha='right');ax.set_ylabel('Sobol index');ax.legend();fig.tight_layout()
    fig.savefig(out/'FIG8_S1_ST_REPLICATED.png',dpi=240);plt.close(fig)
    return output_report(out,'FIG8','FORWARD_MODEL_GSA_COMPLETED_'+scope,evaluations=n_evals,replicates=reps,
                         output_key=metric)
