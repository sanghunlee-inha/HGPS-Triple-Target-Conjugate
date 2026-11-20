"""Deep paired V22 0D *source-outcome* analysis. Not a new mechanistic PK/PD solver."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from core import digest,jd,emit,output_report,hold,V22_KNOWN_SHA

def load_v22(path,allow_other_hash=False):
    sha=digest(path)
    if sha!=V22_KNOWN_SHA and not allow_other_hash:
        raise ValueError('V22_HASH_DIFFERENT_FROM_VERIFIED_SOURCE; rerun with explicit --accept-new-v22 only after verifying lineage')
    df=pd.read_csv(path,low_memory=False)
    cols={'seed','end_R_conjugate','end_R_lonafarnib','delta_R','zero_D_valid'}
    if not cols.issubset(df):raise ValueError('V22_REQUIRED_COLUMNS_MISSING '+str(cols-set(df.columns)))
    valid=df['zero_D_valid'].astype(str).str.lower().isin(['true','1'])
    for col in ['seed','end_R_conjugate','end_R_lonafarnib','delta_R']:
        df[col]=pd.to_numeric(df[col],errors='coerce')
        valid &= np.isfinite(df[col].to_numpy(float))
    x=df.loc[valid].copy()
    if x.empty:raise ValueError('V22_NO_VALID_ROWS')
    mismatch=np.abs((x.end_R_conjugate-x.end_R_lonafarnib)-x.delta_R)
    if mismatch.max()>1.e-7*max(1,float(np.max(np.abs(x.delta_R)))):raise ValueError('V22_PAIRED_DIFFERENCE_NOT_REPRODUCIBLE')
    if x.seed.nunique()<3:raise ValueError('V22_FEWER_THAN_3_INDEPENDENT_SEEDS')
    return x,sha,{'all_rows':len(df),'valid_rows':len(x),'max_diff_error':float(mismatch.max())}

def robust(a):
    a=np.asarray(a,dtype=float)
    lo,hi=np.quantile(a,[0.01,0.99])
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
      'p01':float(np.quantile(a,0.01)),'p05':float(np.quantile(a,.05)),
      'p25':float(np.quantile(a,.25)),'p75':float(np.quantile(a,.75)),
      'p95':float(np.quantile(a,.95)),'p99':float(np.quantile(a,.99)),
      'min':float(a.min()),'max':float(a.max()),
      'winsor01_99_mean':float(np.mean(np.clip(a,lo,hi))),
      'p_conjugate_metric_greater_than_lona':float((a>0).mean()),
      'p_metric_equal':float((a==0).mean())}



def _bootstrap_part(task):
    """A task samples actual within-seed 0D outcomes, not clinical patients."""
    task_id,count,arrays,base_seed=task
    rng=np.random.default_rng(int(base_seed)+task_id*7919)
    block=np.empty((count,4),dtype=np.float64)
    nk=len(arrays)
    for r in range(count):
        sampled=rng.integers(nk,size=nk)
        med=[];wins=[];pos=[]
        for i in sampled:
            original=arrays[int(i)]
            draw=original[rng.integers(len(original),size=len(original))]
            med.append(float(np.median(draw)))
            pos.append(float(np.mean(draw>0)))
            lo,hi=np.quantile(draw,[.01,.99]);wins.append(float(np.mean(np.clip(draw,lo,hi))))
        block[r]=[np.median(med),np.mean(med),np.mean(pos),np.mean(wins)]
    return task_id,block

def hierarchical_bootstrap(byseed,bootstrap,seed,workers):
    values=[byseed[k] for k in sorted(byseed)]
    batch=128
    tasks=[]
    for i,st in enumerate(range(0,bootstrap,batch)):
        tasks.append((i,min(batch,bootstrap-st),values,seed))
    out=[None]*len(tasks)
    if workers>1 and bootstrap>=256:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs=[pool.submit(_bootstrap_part,task) for task in tasks]
            completed=0
            for f in as_completed(futs):
                k,v=f.result();out[k]=v;completed+=len(v)
                emit('FIG5_REAL_BOOTSTRAP_PROGRESS',completed,'/',bootstrap)
    else:
        for task in tasks:
            k,v=_bootstrap_part(task);out[k]=v
            emit('FIG5_REAL_BOOTSTRAP_PROGRESS',sum(len(x) for x in out if x is not None),'/',bootstrap)
    return np.vstack(out)


def source_parameter_associations(df,out):
    """Associations only; no independent PK/PD input or causal interpretation is inferred."""
    reserved={'seed','idx','end_R_conjugate','end_R_lonafarnib','delta_R','zero_D_valid',
              'valid_CFD','valid_patient_IMT','release_mode'}
    rows=[];y=df.delta_R.to_numpy(dtype=np.float64)
    yy=np.sign(y)*np.log1p(np.abs(y))
    for col in df.columns:
        if col in reserved:continue
        x=pd.to_numeric(df[col],errors='coerce').to_numpy(dtype=float)
        good=np.isfinite(x)&np.isfinite(yy)
        if good.sum()<500 or np.unique(x[good][:min(30000,len(x))]).size<4:continue
        corr,pval=spearmanr(x[good],yy[good])
        if not (np.isfinite(corr) and np.isfinite(pval)):continue
        q=np.unique(np.quantile(x[good],[0,.2,.4,.6,.8,1]))
        if len(q)<4:continue
        for bi in range(len(q)-1):
            cut=(x>=q[bi])&((x<=q[bi+1]) if bi==len(q)-2 else (x<q[bi+1]))&good
            if cut.sum()==0:continue
            rows.append({'source_parameter':col,'source_parameter_units':'SOURCE_UNVERIFIED',
                'bin':bi+1,'bin_lo':q[bi],'bin_hi':q[bi+1],
                'n':int(cut.sum()),'median_deltaR':float(np.median(y[cut])),
                'p_deltaR_positive':float(np.mean(y[cut]>0)),
                'spearman_rho_signlog_deltaR':float(corr),'spearman_p_unadjusted':float(pval)})
    if rows:
        result=pd.DataFrame(rows)
        result.to_csv(out/'FIG5_SOURCE_PARAMETER_ASSOCIATIONS_EXPLORATORY.csv',index=False,encoding='utf-8-sig')
    return {'source_parameter_count':int(len({r['source_parameter'] for r in rows})),
            'gate':'OBSERVATIONAL_MODEL_INPUT_ASSOCIATIONS_NOT_MECHANISM_NOT_CAUSALITY'}

def run(path,out,bootstrap=8192,allow_other_hash=False,seed=20261114,workers=12):
    if bootstrap<1 or workers<1: raise ValueError('FIG5_BAD_BOOTSTRAP_SETTINGS')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    df,sha,audit=load_v22(path,allow_other_hash)
    value=df.delta_R.to_numpy(float)
    byseed={int(k):v.delta_R.to_numpy(float) for k,v in df.groupby('seed',sort=True)}
    seeds=list(byseed)
    results=[]
    for s in seeds:
        d=robust(byseed[s]);d['seed']=s;results.append(d)
    pd.DataFrame(results).to_csv(out/'FIG5_SEED_LEVEL_ROBUST_ESTIMATES.csv',index=False,encoding='utf-8-sig')
    by_seed_median=np.array([np.median(byseed[s]) for s in seeds])
    by_seed_winsor=np.array([robust(byseed[s])['winsor01_99_mean'] for s in seeds])
    emit('FIG5_FULL_DATA_BOOTSTRAP_START',len(df),'valid_0D_cases',bootstrap,'draws',workers,'workers')
    bs=hierarchical_bootstrap(byseed,bootstrap,seed,workers)
    ci={lab:{'lower95':float(np.quantile(bs[:,j],.025)),'upper95':float(np.quantile(bs[:,j],.975))} for j,lab in enumerate(['median_of_seed_medians','mean_of_seed_medians','win_probability_by_seed','winsor_mean_by_seed'])}
    loso=[]
    for s in seeds:
        x=np.array([np.median(byseed[k]) for k in seeds if k!=s]);loso.append({'held_out_seed':s,'median_remaining_seed_medians':float(np.median(x)),'mean_remaining_seed_medians':float(x.mean())})
    pd.DataFrame(loso).to_csv(out/'FIG5_LEAVE_ONE_SEED_OUT.csv',index=False,encoding='utf-8-sig')
    param_screen=source_parameter_associations(df,out)
    summary={'source':str(path),'source_sha256':sha,'audit':audit,'source_parameter_screen':param_screen,'seed_ids':seeds,
      'all_simulated_cases':robust(value),'simulated_seed_medians':by_seed_median.tolist(),
      'simulated_seed_winsor_means':by_seed_winsor.tolist(),
      'hierarchical_bootstrap':{'draws':bootstrap,'seed':seed,'ci95':ci,
         'method':'resample simulation seeds and within-seed scenario observations; NOT donor/patient bootstrapping'},
      'definition':'delta_R=end_R_conjugate-end_R_lonafarnib; biological meaning/direction must be obtained from the original model spec',
      'interpretation_gate':'ORIGINAL_0D_OUTCOMES; NOT_NEW_ODE_INTEGRATION; NOT_HUMAN_EFFICACY; NOT_3D_CFD',
      'source_colnames':list(df.columns)}
    jd(out/'FIG5_DEEP_V22_STATISTICS.json',summary)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':23,'axes.titlesize':23,'figure.facecolor':'white','savefig.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(15,9))
    for s in seeds:
        a=byseed[s];z=np.sign(a)*np.log10(1+np.abs(a))
        ax.hist(z,bins=np.linspace(-2,7,110),histtype='step',density=True,label=f'Seed {s}',linewidth=1.3)
    ax.axvline(0,c='k',lw=1)
    ax.set(xlabel='Signed log10(1 + |delta_R|), original 0D model units',ylabel='Density',title='V22 original paired 0D model: heavy-tail and seed variability')
    ax.legend();fig.tight_layout();fig.savefig(out/'FIG5_PAIRED_HEAVY_TAIL.png',dpi=230);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,9))
    ax.boxplot([np.sign(byseed[s])*np.log10(1+np.abs(byseed[s])) for s in seeds],tick_labels=[str(s) for s in seeds],showfliers=False)
    ax.set(xlabel='Simulation seed',ylabel='Signed log10(1+|delta_R|)',title='Within- and between-seed variability')
    fig.tight_layout();fig.savefig(out/'FIG5_SEED_HETEROGENEITY.png',dpi=230);plt.close(fig)
    return output_report(out,'FIG5','ANALYZED_VERIFIED_0D_RAW_OUTCOMES_NOT_NEW_PKPD_SOLVE',source_sha256=sha,n_valid=len(df),bootstrap_draws=bootstrap,source_status=summary['interpretation_gate'])
