#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Source-locked HGPS V22 lonafarnib comparator audit and real 24-month BDF replay.

The source design is deterministic synthetic QMC, not independent patients.
Never alter the user's original model, protocol, CSV, or SQLite files.
Never silently impute missing outputs, clip extreme values or infer clinical effects.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sqlite3
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEEDS = (11,23,37,51,73)
EXPECTED_V22_CSV_SHA = '5ae6ffef4fbf777984e3b1052362770da8fcfb83f8400a438789094874677a6c'
EXPECTED_SHA = {
    'v22_core.py':'6c55d4468b2f1e75a9b3fdace4e5698109f82836c5bd35c5a1155f6439ebd840',
    'v22_analyze.py':'ca4542ef4a12cb54f81a7b8e424713658df4b9a800a7a5a55afd82421d0abad5',
    'v22_vascular.py':'c82bbfe9cb525738b527485038806e6c5bc667b23a2f93d0b1429e96ba8aa54b',
    'v21_core.py':'fe9b814ca3342b7448f4f1dde23437b3e2287ddeb8748c2cdb284b83117b8cf8',
}
MODES=('independent','sequential_p1_tbki_mir','sequential_mir_tbki_p1')
AXES=('ec_activation','vsmc_dysfunction','ecm_stiffness')
MODEL_ENDS=('progerin_total','farnesyl_progerin','ec_activation',
            'vsmc_dysfunction','ecm_stiffness','tox_cardiac','tox_immune',
            'prpc_function','tbk1_activity','srsf1','lmna_cryptic')
PAIR=['seed','idx','release_mode']
RATES={'low':0.0001,'high':0.001}
K_REPAIR=0.001
DAYS=730.5
AUDIT_K=1000.0
REFINEMENT_LIMIT=1e-3 # original V21/V22 threshold, not a new accuracy claim


class Hold(RuntimeError):
    pass


def require(cond,code):
    if not cond: raise Hold(code)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''): h.update(data)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.new')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,sort_keys=True),encoding='utf-8')
    os.replace(tmp,path)


def write_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig',float_format='%.16g')


def log(stage,**fields):
    print('DEEP_V22_'+stage+'='+json.dumps(fields,ensure_ascii=False,sort_keys=True,default=str,allow_nan=False),flush=True)


def source_lock(v22, v21, output_research, csv_path, *, preflight_replay=False, base=None, protocol=None):
    """Check exact collected source files plus immutable V22 input lock, before computing."""
    v22=Path(v22).resolve(); v21=Path(v21).resolve(); output_research=Path(output_research).resolve()
    for name in ('v22_core.py','v22_analyze.py','v22_vascular.py'):
        file=v22/name
        require(file.is_file(),'HOLD_V22_SOURCE_MISSING:'+str(file))
        require(sha256(file)==EXPECTED_SHA[name],'HOLD_V22_SOURCE_CHANGED:'+name)
    file=v21/'v21_core.py'
    require(file.is_file(),'HOLD_V21_SOURCE_MISSING:'+str(file))
    require(sha256(file)==EXPECTED_SHA['v21_core.py'],'HOLD_V21_SOURCE_CHANGED')
    source_sha=sha256(csv_path)
    require(source_sha==EXPECTED_V22_CSV_SHA,'HOLD_V22_CSV_SHA_MISMATCH:'+source_sha)
    lock=read_json(output_research/'V22_INPUT_LOCK.json')
    gate=read_json(output_research/'V22_BDF_FINAL_GATE.json')
    analysis_gate=read_json(output_research/'V22_ANALYSIS_FINAL_GATE.json')
    require(lock.get('software')=='V22_HGPS_MULTISCALE_RESEARCH','HOLD_INPUT_LOCK_SOFTWARE')
    require(lock.get('seed_list')==list(SEEDS),'HOLD_DESIGN_SEEDS')
    require(lock.get('n_per_seed')==65536,'HOLD_DESIGN_SIZE')
    require(abs(float(lock.get('days',-1))-DAYS)<1e-9,'HOLD_DESIGN_DAYS')
    require(lock.get('v22_core_sha256')==EXPECTED_SHA['v22_core.py'],'HOLD_V22_INPUT_CODE_LOCK')
    require(lock.get('v21_core_sha256')==EXPECTED_SHA['v21_core.py'],'HOLD_V21_INPUT_CODE_LOCK')
    require(gate.get('status')=='PASS_5_INDEPENDENT_QMC_DESIGNS_BDF_NUMERICS_NOT_CLINICAL','HOLD_INCOMPLETE_BDF_GATE')
    require(gate.get('design_sizes')==[65536]*5,'HOLD_INCOMPLETE_BDF_SEEDS')
    require(analysis_gate.get('n_vascular_0D_case_rate_comparisons')==10240,'HOLD_0D_ANALYSIS_GATE')
    if protocol is not None:
        require(sha256(protocol)==lock.get('protocol_sha256'),'HOLD_PROTOCOL_SHA_CHANGED')
    require(base is None or Path(base).is_dir(),'HOLD_V21_BASE_DIR')
    return {'source_csv_sha256':source_sha,'v22_source':str(v22),'v21_source':str(v21),
            'input_lock_sha256':sha256(output_research/'V22_INPUT_LOCK.json'),
            'bdf_gate_sha256':sha256(output_research/'V22_BDF_FINAL_GATE.json'),
            'original_model_hashes':lock.get('source_sha256'),
            'original_n_sobol_biologies':327680,'n_independent_scrambles':5,
            'not_patient_effect':True,'not_3d_CFD':True}


def load_v22_csv(path):
    df=pd.read_csv(path,encoding='utf-8-sig')
    required=set(PAIR+['hypothetical_0D_rate','end_R_conjugate','end_R_lonafarnib','delta_R','zero_D_valid'])
    require(required.issubset(df.columns),'HOLD_MISSING_COLUMN:'+','.join(sorted(required-set(df.columns))))
    require(len(df)==10240,'HOLD_EXPECTED_10240_RATES')
    require(df['seed'].notna().all() and df['idx'].notna().all(),'HOLD_MISSING_PAIR_ID')
    df['seed']=df['seed'].astype('int64');df['idx']=df['idx'].astype('int64')
    require(set(df['seed'])==set(SEEDS),'HOLD_MISSING_SEEDS')
    require(set(df['hypothetical_0D_rate'])=={'low','high'},'HOLD_RATE_LABELS')
    require(set(df['release_mode'])=={'independent'},'HOLD_RELEASE_LABELS')
    require(df['idx'].between(0,65535).all() and (df['idx']%64==0).all(),'HOLD_64_STRIDE')
    require(df['zero_D_valid'].astype(str).str.lower().isin(('true','1')).all(),'HOLD_INVALID_0D_CASE')
    for name in ['end_R_conjugate','end_R_lonafarnib','delta_R']:
        df[name]=pd.to_numeric(df[name],errors='coerce')
    require(np.isfinite(df[['end_R_conjugate','end_R_lonafarnib','delta_R']].to_numpy(dtype=float)).all(),'HOLD_NONFINITE_0D')
    require((df['end_R_conjugate']>0).all() and (df['end_R_lonafarnib']>0).all(),'HOLD_NONPOSITIVE_R')
    require(not df.duplicated(PAIR+['hypothetical_0D_rate']).any(),'HOLD_0D_DUPLICATE_PAIR')
    chk=df.groupby(['seed','hypothetical_0D_rate']).size().to_dict()
    require(all(chk.get((seed,r))==1024 for seed in SEEDS for r in RATES),'HOLD_RATE_SEED_IMBALANCE')
    delta_check=df['end_R_conjugate']-df['end_R_lonafarnib']
    require(np.allclose(delta_check,df['delta_R'],rtol=1e-10,atol=1e-9),'HOLD_DELTA_RECOMPUTE_MISMATCH')
    rate=df.pivot(index=PAIR,columns='hypothetical_0D_rate',values='delta_R')
    require(len(rate)==5120 and rate.notna().all().all(),'HOLD_RATE_PAIRING')
    high=df[df.hypothetical_0D_rate=='high']['delta_R'].to_numpy()
    low=df[df.hypothetical_0D_rate=='low']['delta_R'].to_numpy()
    require(np.isclose(high.mean(),1759.6081956721725,rtol=1e-10,atol=1e-8),'HOLD_HIGH_MEAN_CHANGED')
    require(np.isclose(np.median(high),-0.0022664738787246856,rtol=0,atol=1e-8),'HOLD_HIGH_MEDIAN_CHANGED')
    require(np.isclose(np.median(low),-0.0019360240451415933,rtol=0,atol=1e-8),'HOLD_LOW_MEDIAN_CHANGED')
    require(int((np.abs(high)>AUDIT_K).sum())==16,'HOLD_EXTREME_COUNT_CHANGED')
    require(np.isclose(high[np.abs(high)<=AUDIT_K].mean(),1.3697851650901418,rtol=1e-8,atol=1e-7),'HOLD_OUTLIER_REMOVED_MEAN_CHANGED')
    return df


def open_readonly_sqlite(path):
    """URI read-only: cannot write into original V22 checkpoint."""
    path=Path(path).resolve()
    require(path.is_file(),'HOLD_MISSING_ORIGINAL_SQLITE:'+str(path))
    conn=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=120)
    conn.execute('PRAGMA query_only=ON')
    return conn


def bdf_cache_record(conn, idx):
    r=conn.execute('SELECT design_sha,blob,nfev FROM scenario WHERE idx=?',(int(idx),)).fetchone()
    require(r is not None,'HOLD_CACHE_IDX_MISSING:'+str(idx))
    import zlib
    data=json.loads(zlib.decompress(r[1]))
    bymode={x['mode']:x for x in data}
    require(set(bymode)==set(MODES),'HOLD_CACHE_RELEASE_INCOMPLETE')
    return {'design_sha':r[0],'by_mode':bymode,'nfev':int(r[2])}


def add_verified_t0(months):
    """Exactly original v22_analyze.py: initial vascular states are all 1.0."""
    require(len(months)>=3,'HOLD_MONTHLY_TIMELINE_TOO_SHORT')
    row={'horizon_days':0.0}
    for arm in ('lona','conj','untreated'):
        for axis in AXES: row[arm+'_'+axis]=1.0
    timeline=[row]+sorted(months,key=lambda x:float(x['horizon_days']))
    t=np.asarray([float(x['horizon_days']) for x in timeline])
    require(len(t)>=4 and abs(t[0])<1e-12 and np.all(np.diff(t)>0),'HOLD_TIMELINE_ORDER')
    require(abs(t[-1]-DAYS)<1e-6,'HOLD_TIMELINE_LAST_DAY')
    for rec in timeline:
        for arm in ('lona','conj','untreated'):
            for axis in AXES:
                key=arm+'_'+axis
                require(key in rec and math.isfinite(float(rec[key])),'HOLD_MONTHLY_MISSING_AXIS:'+key)
    return timeline


def exp_integral_factors(z):
    """Stable exact factors for a linear forcing in y'=-a*y-(f0+slope*t)."""
    if abs(z)<1e-5:
        # phi1=(1-e^-z)/z; phi2=(z-1+e^-z)/z^2
        p1=1-z/2+z*z/6-z**3/24+z**4/120
        p2=.5-z/6+z*z/24-z**3/120+z**4/720
        return p1,p2
    m=-math.expm1(-z)
    return m/z,(z-m)/(z*z)


def exact_linear_ode(t,forcing,k_repair=K_REPAIR):
    """Closed-form integration of original 0D ODE on every linear-interpolation interval.

    This is a second numerical reference method, not a replacement therapy model.
    """
    x=np.asarray(t,dtype=float);f=np.asarray(forcing,dtype=float)
    require(x.ndim==f.ndim==1 and len(x)==len(f) and len(x)>=4,'HOLD_EXACT_0D_SHAPE')
    require(np.all(np.diff(x)>0) and np.isfinite(x).all() and np.isfinite(f).all(),'HOLD_EXACT_0D_NONFINITE')
    require(k_repair>=0,'HOLD_NEGATIVE_REPAIR_RATE')
    y=0.
    for i in range(len(x)-1):
        h=float(x[i+1]-x[i]);z=k_repair*h
        p1,p2=exp_integral_factors(z)
        y=math.exp(-z)*y-h*(float(f[i])*p1+(float(f[i+1])-float(f[i]))*p2)
    return y


def exact_remodel_from_timeline(timeline,rate):
    """Compute exact original per-axis log-radius trajectories for both therapy arms."""
    require(rate in RATES,'HOLD_UNKNOWN_RATE')
    t=np.asarray([float(r['horizon_days']) for r in timeline],dtype=float)
    coeff=RATES[rate]
    contrib={}
    out={'rate':rate}
    for arm in ('lona','conj'):
        terms=[]
        for axis in AXES:
            f=np.asarray([float(row[arm+'_'+axis])-float(row['untreated_'+axis]) for row in timeline])
            v=exact_linear_ode(t,coeff*f)
            contrib[(arm,axis)]=-4*v
            terms.append(v)
        logr=-4*sum(terms)
        out['log_R_'+arm]=float(logr)
        out['R_'+arm]=float(math.exp(logr))
        out['radius_'+arm]=float(math.exp(-logr/4))
    out['delta_R_exact']=float(out['R_conj']-out['R_lona'])
    out['delta_logR_exact']=float(out['log_R_conj']-out['log_R_lona'])
    for axis in AXES:
        out['d_logR_'+axis]=float(contrib[('conj',axis)]-contrib[('lona',axis)])
    # The exact model is log-additive but the physical R transform is nonlinear.
    out['d_logR_sum_error']=float(out['delta_logR_exact']-sum(out['d_logR_'+x] for x in AXES))
    return out


def descriptive_stats(values):
    a=np.asarray(values,dtype=float)
    require(a.size>0 and np.isfinite(a).all(),'HOLD_STATS_INVALID')
    pos=int((a>0).sum());n=len(a)
    return {'n':n,'mean':float(a.mean()),'median':float(np.median(a)),
        'sd':float(a.std(ddof=1)) if n>1 else None,
        'p05':float(np.quantile(a,.05)),'p25':float(np.quantile(a,.25)),
        'p75':float(np.quantile(a,.75)),'p95':float(np.quantile(a,.95)),
        'positive':pos,'positive_fraction':pos/n,
        'trimmed_mean_05':float(stats.trim_mean(a,.05)),
        'trimmed_mean_10':float(stats.trim_mean(a,.10)),
        'winsorized_mean_05':float(np.mean(np.clip(a,np.quantile(a,.05),np.quantile(a,.95)))),
        'median_absolute_deviation':float(stats.median_abs_deviation(a)),
        'n_abs_gt_1000':int((np.abs(a)>1000).sum())}


def robust_table(paired):
    records=[]
    for rate in RATES:
        a=paired['delta_R_'+rate].to_numpy(dtype=float)
        for k in [0,1,5,10,16,26,52,103,256,512]:
            order=np.argsort(np.abs(a),kind='stable')
            keep=order[:len(a)-k] if k else order
            z=a[keep]
            r=descriptive_stats(z)
            records.append({'rate':rate,'excluded_top_abs':k,'cutoff_absolute':float(np.max(np.abs(z))),**r})
        z=a[np.abs(a)<=AUDIT_K]
        records.append({'rate':rate,'excluded_top_abs':'threshold_abs_gt_1000',
            'cutoff_absolute':AUDIT_K,**descriptive_stats(z)})
    return records


def independent_seed_diagnostics(paired):
    records=[]
    for rate in RATES:
        for seed in SEEDS:
            x=paired.loc[paired.seed==seed,'delta_R_'+rate].to_numpy()
            records.append({'rate':rate,'group':'one_seed','excluded_seed':'',
                            'seed':seed,**descriptive_stats(x)})
            y=paired.loc[paired.seed!=seed,'delta_R_'+rate].to_numpy()
            records.append({'rate':rate,'group':'leave_one_seed_out','excluded_seed':seed,
                            'seed':'',**descriptive_stats(y)})
    return records


def randomized_qmc_numerical_uncertainty(paired):
    """Independent QMC seed estimates; NOT patient confidence intervals."""
    rows=[]
    for rate in RATES:
        seed_means=[];seed_medians=[];seed_trim=[];seed_signs=[]
        for s in SEEDS:
            a=paired.loc[paired.seed==s,'delta_R_'+rate].to_numpy()
            seed_means.append(float(a.mean()))
            seed_medians.append(float(np.median(a)))
            seed_trim.append(float(stats.trim_mean(a,.05)))
            seed_signs.append(float((a>0).mean()))
        for label,z in [('mean',seed_means),('median_of_seed_medians',seed_medians),
                        ('mean_of_seed_trimmed_mean_05',seed_trim),('mean_of_seed_sign_fractions',seed_signs)]:
            z=np.asarray(z,dtype=float)
            rows.append({'rate':rate,'metric':label,
                'mean_across_five_scrambled_sobol_seeds':float(z.mean()),
                'seed_sd':float(z.std(ddof=1)),
                'qmc_integration_SE_NOT_CLINICAL_CI':float(z.std(ddof=1)/math.sqrt(5)),
                'n_independent_qmc_scrambles':5,'not_patient_probability':True})
    return rows


def axis_diagnostics(paired):
    rows=[]
    for rate in RATES:
        target=paired['d_logR_'+rate].to_numpy(float)
        for axis in AXES:
            a=paired['d_logR_'+rate+'_'+axis].to_numpy(float)
            nz=np.isfinite(a)&np.isfinite(target)
            rho,p=stats.spearmanr(a[nz],target[nz])
            # Mathematical contribution to logR, not a causal inference of PRNP/TBK1/miR action.
            rows.append({'rate':rate,'axis':axis,'mean_signed_logR_component':float(np.mean(a)),
                'median_logR_component':float(np.median(a)),
                'spearman_vs_total_logR':float(rho),'exploratory_rank_p_value_not_clinical':float(p),
                'mean_abs_component_fraction':float(np.mean(np.abs(a)/(np.sum(np.abs(paired[[f'd_logR_{rate}_{x}' for x in AXES]].to_numpy(float)),axis=1)+1e-30))),
                'interpretation':'mathematical 0D linear forcing contribution; not mechanistic causality'})
    return rows


def build_stress_design_plan(paired):
    """Prioritize extreme, sign/baseline/seed-stratified controls, then remaining cases.

    ALL 5120 cases appear exactly once. Training/test split is not used for this replay.
    """
    temp=paired.copy()
    temp['abs_high']=temp.delta_R_high.abs()
    temp['extreme']=(temp.abs_high>AUDIT_K)
    temp['high_sign']=np.where(temp.delta_R_high>0,'plus','nonplus')
    temp['magnitude_bin']=pd.qcut(temp.abs_high.rank(method='first'),q=10,labels=False).astype(int)
    extreme=temp[temp.extreme].sort_values(['abs_high','seed','idx'],ascending=[False,True,True]).copy()
    other=temp[~temp.extreme].copy()
    controls=[]
    for _,g in other.groupby(['seed','high_sign','magnitude_bin'],sort=True):
        controls.append(g.sort_values(['idx']).iloc[0].name)
    # 5 seeds x (two signs x 10 deciles): 100 controls if populated.
    ctrl=other.loc[controls].drop_duplicates().sort_values(['seed','magnitude_bin','idx'])
    rest=other.drop(index=ctrl.index).sort_values(['magnitude_bin','seed','idx'],ascending=[False,True,True])
    plan=pd.concat([extreme,ctrl,rest],ignore_index=True)
    require(len(plan)==5120 and not plan.duplicated(['seed','idx']).any(),'HOLD_PLAN_LOST_CASES')
    plan['plan_rank']=np.arange(len(plan))
    plan['selection_category']=np.where(plan.extreme,'PRIMARY_16_EXTREME',
        np.where(plan.plan_rank<len(extreme)+len(ctrl),'STRATIFIED_CONTROL','FULL_5120_REPLAY'))
    return plan


def create_figures(paired,robust,out):
    # Publication-oriented white-background editable SVG and high DPI PNG.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':23,'axes.titlesize':23,'axes.labelsize':23,
                         'xtick.labelsize':21,'ytick.labelsize':21,
                         'legend.fontsize':18,'axes.linewidth':1.2,
                         'figure.facecolor':'white','axes.facecolor':'white',
                         'savefig.facecolor':'white','svg.fonttype':'none'})
    files=[]
    a=paired['delta_R_high'].to_numpy()
    fig,ax=plt.subplots(figsize=(15,9),layout='constrained')
    ax.hist(np.arcsinh(a),bins=80,edgecolor='white',linewidth=.3)
    ax.axvline(np.arcsinh(np.median(a)),ls='--',color='black',lw=2,label='Median')
    ax.set(xlabel='asinh(Conjugate − lonafarnib ΔR)',ylabel='Design cases',
           title='a  High-rate paired contrast: complete distribution')
    ax.legend(frameon=False)
    fig.savefig(out/'FIG_V22_A_HIGH_RATE_DISTRIBUTION.png',dpi=300)
    fig.savefig(out/'FIG_V22_A_HIGH_RATE_DISTRIBUTION.svg');plt.close(fig)
    files.append('FIG_V22_A_HIGH_RATE_DISTRIBUTION.png')
    tab=pd.DataFrame(robust)
    tab=tab[(tab.rate=='high') & tab.excluded_top_abs.apply(lambda x:isinstance(x,(int,np.integer)))]
    fig,ax=plt.subplots(figsize=(15,9),layout='constrained')
    x=tab.excluded_top_abs.to_numpy(dtype=float)
    ax.plot(x,tab['mean'].to_numpy(dtype=float),marker='o',lw=2.5,label='Mean')
    ax.plot(x,tab['median'].to_numpy(dtype=float),marker='s',lw=2.5,label='Median')
    ax.set_yscale('symlog',linthresh=.1);ax.set_xlabel('Largest |ΔR| cases removed')
    ax.set_ylabel('ΔR (symlog)');ax.set_title('b  Influence of extreme design outputs',loc='left')
    ax.legend(frameon=False)
    fig.savefig(out/'FIG_V22_B_EXTREME_INFLUENCE.png',dpi=300)
    fig.savefig(out/'FIG_V22_B_EXTREME_INFLUENCE.svg');plt.close(fig)
    files.append('FIG_V22_B_EXTREME_INFLUENCE.png')
    fig,ax=plt.subplots(figsize=(15,9),layout='constrained')
    ax.scatter(paired['d_logR_low'],paired['d_logR_high'],s=8,alpha=.45)
    xy=paired['d_logR_low'].to_numpy(float)
    xx=np.linspace(float(xy.min()),float(xy.max()),60)
    ax.plot(xx,10*xx,ls='--',color='black',lw=2,label='High = 10 × Low (structural)')
    ax.set(xlabel='Low-rate Δlog(R)',ylabel='High-rate Δlog(R)',
           title='c  Exact rate-scaling identity in the 0D model')
    ax.legend(frameon=False)
    fig.savefig(out/'FIG_V22_C_RATE_SCALING_IDENTITY.png',dpi=300)
    fig.savefig(out/'FIG_V22_C_RATE_SCALING_IDENTITY.svg');plt.close(fig)
    files.append('FIG_V22_C_RATE_SCALING_IDENTITY.png')
    fig,ax=plt.subplots(figsize=(15,9),layout='constrained')
    for s in SEEDS:
        sub=paired.loc[paired.seed==s]
        ax.scatter(sub['d_logR_high'],np.arcsinh(sub['delta_R_high']),s=7,alpha=.35,label=f'Seed {s}')
    ax.set(xlabel='High-rate Δlog(R)',ylabel='asinh(ΔR)',
           title='d  Nonlinear transformation from log-resistance to ΔR')
    ax.legend(frameon=False,ncol=3)
    fig.savefig(out/'FIG_V22_D_LOG_VS_RAW_EFFECT.png',dpi=300)
    fig.savefig(out/'FIG_V22_D_LOG_VS_RAW_EFFECT.svg');plt.close(fig)
    files.append('FIG_V22_D_LOG_VS_RAW_EFFECT.png')
    return files


def audit_0d(csv_path,raw_out,out,*,v22_dir,v21_dir,base,protocol):
    out.mkdir(parents=True,exist_ok=True)
    locks=source_lock(v22_dir,v21_dir,raw_out,csv_path,base=base,protocol=protocol)
    df=load_v22_csv(csv_path)
    idx_lookup={(int(r.seed),int(r.idx),str(r.hypothetical_0D_rate)):r for r in df.itertuples(index=False)}
    matched=[];entries=[]
    max_original_exact_log_error=0.;total_missing=0
    for seed in SEEDS:
        db=open_readonly_sqlite(raw_out/f'V22_SEED_{seed}.sqlite')
        db_n=int(db.execute('SELECT COUNT(*) FROM scenario').fetchone()[0])
        require(db_n==65536,'HOLD_SEED_CHECKPOINT_COUNT:'+str(seed)+':'+str(db_n))
        log('SOURCE_SEED_START',seed=seed,total=1024)
        for idx in range(0,65536,64):
            data=bdf_cache_record(db,idx)
            original=data['by_mode']['independent']
            require(original.get('timeline'),'HOLD_V22_TIMELINE_NOT_SAVED:'+str(seed)+':'+str(idx))
            timeline=add_verified_t0(original['timeline'])
            rec={'seed':seed,'idx':idx,'release_mode':'independent',
                 'scenario_sha256':data['design_sha'],
                 'nfev_original_full_scenario':data['nfev']}
            for rate in RATES:
                source=idx_lookup[(seed,idx,rate)]
                sol=exact_remodel_from_timeline(timeline,rate)
                rec['delta_R_'+rate]=float(source.delta_R)
                rec['d_logR_'+rate]=sol['delta_logR_exact']
                rec['R_conj_source_'+rate]=float(source.end_R_conjugate)
                rec['R_lona_source_'+rate]=float(source.end_R_lonafarnib)
                rec['R_conj_exact_'+rate]=sol['R_conj']
                rec['R_lona_exact_'+rate]=sol['R_lona']
                rec['delta_R_exact_'+rate]=sol['delta_R_exact']
                rec['logR_conj_'+rate]=sol['log_R_conj']
                rec['logR_lona_'+rate]=sol['log_R_lona']
                rec['radius_conj_'+rate]=sol['radius_conj']
                rec['radius_lona_'+rate]=sol['radius_lona']
                rec['exact_radius_outside_source_valid_gate_'+rate]=bool(not(0.01<=sol['radius_conj']<=100 and 0.01<=sol['radius_lona']<=100))
                rec['original_minus_exact_dR_'+rate]=float(source.delta_R)-sol['delta_R_exact']
                for axis in AXES: rec['d_logR_'+rate+'_'+axis]=sol['d_logR_'+axis]
                err=max(abs(math.log(float(source.end_R_conjugate))-sol['log_R_conj']),
                        abs(math.log(float(source.end_R_lonafarnib))-sol['log_R_lona']))
                rec['max_absolute_logR_source_exact_error_'+rate]=err
                max_original_exact_log_error=max(max_original_exact_log_error,err)
            diff=rec['d_logR_high']-10*rec['d_logR_low']
            rec['rate_logR_tenfold_identity_error']=diff
            require(abs(diff)<1e-10*(1+abs(rec['d_logR_high'])),'HOLD_0D_EXACT_RATE_SCALING')
            entries.append(rec)
        db.close()
        log('SOURCE_SEED_COMPLETE',seed=seed,cases=1024)
    paired=pd.DataFrame(entries)
    require(len(paired)==5120,'HOLD_PAIRED_SIZE')
    require(int((paired.delta_R_high.abs()>AUDIT_K).sum())==16,'HOLD_INCONSISTENT_EXTREME_COUNT')
    # Same sign in high/low is a mathematical identity, NOT independent biological replication.
    original_sign=(np.sign(paired.delta_R_high)==np.sign(paired.delta_R_low)).mean()
    numerical_sign=(np.sign(paired.delta_R_exact_high)==np.sign(paired.delta_R_exact_low)).mean()
    report={'status':'NUMERIC_0D_AUDIT_COMPLETED_NOT_CLINICAL',
        'source_lock':locks,'seed_cases':5120,'case_rates':10240,
        'original_low_high_sign_concordance':float(original_sign),
        'analytic_low_high_sign_concordance':float(numerical_sign),
        'exact_high_to_low_d_logR_scaling':10,
        'this_sign_concordance_is_structurally_forced_not_independent_validation':True,
        'max_absolute_log_R_discrepancy_original_DOP853_vs_closed_form':float(max_original_exact_log_error),
        'n_cases_with_DOP853_closed_form_logR_error_gt_1e_minus_5':int(((paired['max_absolute_logR_source_exact_error_high']>1e-5) | (paired['max_absolute_logR_source_exact_error_low']>1e-5)).sum()),
        'n_cases_with_exact_radius_outside_original_gate':int((paired['exact_radius_outside_source_valid_gate_high'] | paired['exact_radius_outside_source_valid_gate_low']).sum()),
        'high_full':descriptive_stats(paired.delta_R_high.to_numpy()),
        'high_without_16_abs_gt_1000':descriptive_stats(paired.loc[paired.delta_R_high.abs()<=AUDIT_K,'delta_R_high'].to_numpy()),
        'low_full':descriptive_stats(paired.delta_R_low.to_numpy()),
        'clinical_efficacy_claim':False,'CFD_3D':False,'patient_effect_calibrated':False}
    robust=robust_table(paired);seed_stability=independent_seed_diagnostics(paired)
    qmc=randomized_qmc_numerical_uncertainty(paired)
    axes=axis_diagnostics(paired)
    plan=build_stress_design_plan(paired)
    write_csv(out/'V22_ORIGINAL_VS_EXACT_0D_5120.csv',paired)
    write_csv(out/'V22_OUTLIER_REMOVAL_SENSITIVITY.csv',robust)
    write_csv(out/'V22_SEED_AND_LOSO.csv',seed_stability)
    write_csv(out/'V22_QMC_NUMERICAL_SE_NOT_CLINICAL.csv',qmc)
    write_csv(out/'V22_0D_AXIS_LOGR_DECOMPOSITION.csv',axes)
    write_csv(out/'V22_BDF_REPLAY_PLAN_5120.csv',plan)
    write_csv(out/'V22_EXTREME_16_SOURCE_AND_EXACT.csv',paired.loc[paired.delta_R_high.abs()>AUDIT_K].sort_values('delta_R_high',key=abs,ascending=False))
    try:report['figure_files']=create_figures(paired,robust,out)
    except ImportError: report['figures']='HOLD_MATPLOTLIB_NOT_INSTALLED'
    atomic_json(out/'AUDIT_0D_STATUS.json',report)
    log('ANALYTIC_0D_COMPLETE',n=5120,extremes=16,
        high_mean=report['high_full']['mean'],high_mean_without_extremes=report['high_without_16_abs_gt_1000']['mean'],
        max_absolute_logR_discrepancy=max_original_exact_log_error,
        exact_rate_scaling='tenfold_logR_identity')
    return paired,plan,report

# -------------------- Genuine original V21 BDF reruns --------------------

def import_source_locked_modules(v22_dir,v21_dir):
    """The source files are imported from the user's unmodified installed trees."""
    v22_dir=Path(v22_dir).resolve(); v21_dir=Path(v21_dir).resolve()
    for name,folder in [('v22_core.py',v22_dir),('v21_core.py',v21_dir)]:
        require(sha256(folder/name)==EXPECTED_SHA[name],'HOLD_MODULE_HASH_DRIFT:'+name)
    for folder in (v21_dir,v22_dir):
        if str(folder) not in sys.path: sys.path.insert(0,str(folder))
    import v22_core
    import v21_core
    require(Path(v22_core.__file__).resolve()==v22_dir/'v22_core.py','HOLD_FOREIGN_V22_MODULE')
    require(Path(v21_core.__file__).resolve()==v21_dir/'v21_core.py','HOLD_FOREIGN_V21_MODULE')
    return v22_core,v21_core


def compare_stored_bdf(original,replayed):
    """Raw-state normalized errors, by mode and time. No rounding or averaging."""
    details=[]; largest=0.;bad=[]
    by_mode={r['mode']:r for r in replayed}
    for mode in MODES:
        old=original[mode]
        new=by_mode[mode]
        oldrows=sorted(old['rows'],key=lambda x:x['horizon_days'])
        newrows=sorted(new['rows'],key=lambda x:x['horizon_days'])
        require(len(oldrows)==len(newrows)==3,'HOLD_DIFFERENT_BDF_HORIZONS')
        for a,b in zip(oldrows,newrows):
            require(float(a['horizon_days'])==float(b['horizon_days']),'HOLD_MISMATCH_HORIZON')
            keys=[k for k in a if (k.startswith(('delta_','lona_','conj_','untreated_'))
                                     and k in b and isinstance(a[k],(int,float))
                                     and isinstance(b[k],(int,float))
                                     and k not in ('lona_nfev','conj_nfev'))]
            require(len(keys)>=len(MODEL_ENDS)*3,'HOLD_NO_BDF_COMPARISON_COLUMNS')
            for key in keys:
                x=float(a[key]);y=float(b[key])
                require(math.isfinite(x) and math.isfinite(y),'HOLD_NONFINITE_BDF_REPLAY')
                error=abs(x-y)/max(1.,abs(x))
                if error>largest:largest=error
                if error>REFINEMENT_LIMIT:bad.append((mode,float(a['horizon_days']),key,error))
            details.append({'mode':mode,'day':float(a['horizon_days']),
                            'n_values_compared':len(keys)})
    return largest,bad,details


def shapley3(values):
    """Exact three-axis Shapley decomposition in fixed model context."""
    require(set(values)==set(range(8)),'HOLD_ABLATION_INCOMPLETE')
    ret={}
    for i in range(3):
        effect=0.
        for coalition in range(8):
            if coalition&(1<<i):continue
            nbits=int(coalition.bit_count())
            w=1/3 if nbits in (0,2) else 1/6
            effect+=w*(values[coalition | (1<<i)]-values[coalition])
        ret[i]=float(effect)
    require(math.isclose(sum(ret.values()),values[7]-values[0],rel_tol=1e-9,abs_tol=1e-9),
            'HOLD_SHAPLEY_NONADDITIVE_MISSING')
    return ret


def true_3axis_bdf_ablation(core,model,release,scenario,protocol,primary):
    """Seven genuine 24-month BDF simulations with original release.rhs + axis masks.

    No synthetic labels are inserted; zero-axis state is NOT equivalent to untreated.
    """
    rows=primary['rows']
    months=primary['monthly']
    require(months,'HOLD_ABLATION_NEEDS_MONTHLY')
    times=np.asarray([0]+[float(m['horizon_days']) for m in months],dtype=float)
    require(np.all(np.diff(times)>0) and abs(times[-1]-DAYS)<1e-8,'HOLD_ABLATION_TIME_GRID')
    endpoint_values={k:{} for k in MODEL_ENDS}
    endpoint_values.update({'R_low':{},'R_high':{}})
    nfev=0
    for mask in range(8):
        if mask==7:
            for k in MODEL_ENDS: endpoint_values[k][mask]=float(rows[-1]['conj_'+k])
            timeline=add_verified_t0(months)
        else:
            bits=tuple(bool(mask&(1<<j)) for j in range(3))
            run,ledger=core.run_conjugate(model,release,scenario['params'],protocol,DAYS,times,
                scenario['volume_ratio'],'independent',scenario['dose_scale'],axis=bits)
            require(ledger['parent_max_relative']<=2e-5 and
                    max(ledger['payload_max_relative'].values())<=2e-5,'HOLD_ABLATION_MASS_BALANCE')
            nfev+=int(run['nfev'])
            for k in MODEL_ENDS:
                endpoint_values[k][mask]=float(run['states'][-1,model.IDX[k]])
            replaced=[]
            for i,old in enumerate(months):
                r=dict(old)
                for k in AXES: r['conj_'+k]=float(run['states'][i+1,model.IDX[k]])
                replaced.append(r)
            timeline=add_verified_t0(replaced)
        for rate in RATES:
            exact=exact_remodel_from_timeline(timeline,rate)
            endpoint_values['R_'+rate][mask]=exact['R_conj']
    names=('P1_PRNP_expression_pathway','TBK1_inhibitor_pathway','miR4651_SRSF1_pathway')
    records=[]
    for outcome,val in endpoint_values.items():
        sh=shapley3(val)
        for j,name in enumerate(names):
            records.append({'outcome':outcome,'axis':name,'shapley_effect':sh[j],
                            'full_111':val[7],'zero_axis_000':val[0],
                            'full_minus_zero_axis':val[7]-val[0],
                            'not_clinical_causal_effect':True,
                            'hypothetical_original_model_factorial_only':True})
    return records,nfev


def worker_replay(job):
    """This FUNCTION performs actual original 24m BDF; invoked in a worker process."""
    (v22_dir,v21_dir,base,protocol_path,scenario,seed,idx,original,refine,ablate)=job
    start=time.monotonic()
    core22,core21=import_source_locked_modules(v22_dir,v21_dir)
    core,paths,model,audit,source,candidates,sel=core22.preflight(v21_dir,base,protocol_path)
    require(core is core21,'HOLD_V21_MODULE_MISMATCH')
    require(scenario['scenario_sha256']==original['design_sha'],'HOLD_REPLAY_WRONG_SCENARIO')
    trial=core21.paired_scenario_job((paths,str(Path(protocol_path).resolve()),scenario,DAYS,False,bool(refine)))
    largest,bad,detail=compare_stored_bdf(original['by_mode'],trial)
    independently=next(r for r in trial if r['mode']=='independent')
    source_independent=original['by_mode']['independent']
    timeline0=add_verified_t0(source_independent['timeline'])
    timeline1=add_verified_t0(independently['monthly'])
    vascular=[]
    for rate in RATES:
        old=exact_remodel_from_timeline(timeline0,rate)
        new=exact_remodel_from_timeline(timeline1,rate)
        vascular.append({'rate':rate,
            'original_cached_BDF_exact_0D_delta_R':old['delta_R_exact'],
            'replayed_BDF_exact_0D_delta_R':new['delta_R_exact'],
            'BDF_replay_minus_original_0D_delta_R':new['delta_R_exact']-old['delta_R_exact'],
            'max_abs_logR_replay_difference':max(abs(old['log_R_conj']-new['log_R_conj']),
                abs(old['log_R_lona']-new['log_R_lona']))})
    refinement=independently.get('refinement')
    if refine:
        require(refinement is not None,'HOLD_REFINE_NOT_PERFORMED')
        require(int(refinement['original_nfev'])>0 and int(refinement['refined_nfev'])>0,
                'HOLD_REFINEMENT_BDF_SOLVER_NOT_USED')
    shapley=[];additional_nfev=0
    if ablate:
        # loader returns original v20_release, no surrogate equations used
        _,_,release,_=core21.load_modules(paths)
        shapley,additional_nfev=true_3axis_bdf_ablation(core21,model,release,
            scenario,core21.load_protocol(protocol_path),independently)
    wall=time.monotonic()-start
    return {'seed':seed,'idx':idx,'scenario_sha256':scenario['scenario_sha256'],
        'status':'PASS' if not bad and (not refine or refinement['pass_threshold_1e_minus_3']) else 'HOLD_NUMERIC_REPLAY',
        'max_scale_normalized_BDF_state_error':largest,'n_values_compared':sum(x['n_values_compared'] for x in detail),
        'bad_fields_count':len(bad),'bad_field_examples':bad[:30],
        'refinement':refinement,'do_refinement':bool(refine),'do_shapley_ablation':bool(ablate),
        'vascular':vascular,'mechanism_shapley':shapley,
        'original_cache_nfev':original['nfev'],
        'new_real_BDF_nfev':sum(int(x['nfev']) for x in trial)+additional_nfev,
        'additional_seven_axis_mask_nfev':additional_nfev,
        'new_BDF_wall_s':wall,'not_patient_prediction':True}


def init_checkpoint(db_path):
    c=sqlite3.connect(db_path,timeout=120)
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('PRAGMA synchronous=FULL')
    c.execute('PRAGMA busy_timeout=120000')
    c.execute('''CREATE TABLE IF NOT EXISTS job (
        seed INTEGER NOT NULL,idx INTEGER NOT NULL,scenario_sha TEXT NOT NULL,
        status TEXT NOT NULL,wall_s REAL NOT NULL,nfev INTEGER NOT NULL,
        result_json TEXT NOT NULL,PRIMARY KEY(seed,idx))''')
    c.commit()
    return c


def ckpt_save(conn,seed,idx,sha,result):
    with conn:
        conn.execute('INSERT INTO job (seed,idx,scenario_sha,status,wall_s,nfev,result_json) VALUES (?,?,?,?,?,?,?)',
            (seed,idx,sha,result['status'],float(result.get('new_BDF_wall_s',0)),
             int(result.get('new_real_BDF_nfev',0)),json.dumps(result,sort_keys=True,allow_nan=False)))


def checkpoint_results(conn):
    return {(int(s),int(i)):(h,stat) for s,i,h,stat in conn.execute('SELECT seed,idx,scenario_sha,status FROM job')}


def make_full_scenario_design_map(plan,raw_out,v22,core,model,audit,candidates):
    """Rebuild ORIGINAL Sobol 65,536 design per seed, retain ONLY requested scenarios.

    No replacement of the deterministic QMC sequence by independent RNG samples.
    """
    import gc
    wanted=defaultdict(set)
    for item in plan.itertuples(index=False): wanted[int(item.seed)].add(int(item.idx))
    result={}
    for seed in SEEDS:
        requested=wanted.get(seed)
        if not requested:continue
        require(max(requested)<65536,'HOLD_INVALID_SCENARIO_INDEX')
        log('RECONSTRUCT_ORIGINAL_SOBOL',seed=seed,n_replay=len(requested),original_full_seed_size=65536)
        original=v22.seed_design(core,model,audit,candidates,65536,seed)
        conn=open_readonly_sqlite(raw_out/f'V22_SEED_{seed}.sqlite')
        for idx in sorted(requested):
            case=original[idx]
            r=conn.execute('SELECT design_sha FROM scenario WHERE idx=?',(idx,)).fetchone()
            require(r is not None and case['scenario_sha256']==r[0],
                    'HOLD_SOBOL_RECONSTRUCTION_SHA_MISMATCH:'+str(seed)+':'+str(idx))
            result[(seed,idx)]=case
        conn.close()
        del original
        gc.collect()
    return result


def make_replay_job(item,scenario,raw_out,*,v22_dir,v21_dir,base,protocol,refine_controls=128,ablation_controls=64):
    seed=int(item['seed']);idx=int(item['idx'])
    rank=int(item['plan_rank']);cat=item['selection_category']
    refined=(cat=='PRIMARY_16_EXTREME' or (cat=='STRATIFIED_CONTROL' and rank<16+refine_controls))
    ablated=(cat=='PRIMARY_16_EXTREME' or (cat=='STRATIFIED_CONTROL' and rank<16+ablation_controls))
    conn=open_readonly_sqlite(raw_out/f'V22_SEED_{seed}.sqlite')
    original=bdf_cache_record(conn,idx);conn.close()
    require(original['design_sha']==scenario['scenario_sha256'],'HOLD_WRONG_ORIGINAL_SHA')
    return (str(v22_dir),str(v21_dir),str(base),str(protocol),scenario,seed,idx,original,refined,ablated)


def resume_replay(plan,raw_out,out,*,v22_dir,v21_dir,base,protocol,workers=16,max_hours=18,
                  max_replays=5120,refine_controls=128,ablation_controls=64):
    require(workers>=1 and workers <= (os.cpu_count() or 1),'HOLD_INVALID_CPU_WORKERS')
    require(0<max_hours<=200,'HOLD_INVALID_WALL_BUDGET')
    require(1<=max_replays<=5120,'HOLD_INVALID_NUMBER_REPLAYS')
    require(0<=refine_controls<=5120 and 0<=ablation_controls<=refine_controls,
            'HOLD_INVALID_ABLATION_REFINEMENT_COUNTS')
    require(len(plan)==5120,'HOLD_AUDIT_REQUIRED_BEFORE_REPLAY')
    t0=time.monotonic()
    v22,core=import_source_locked_modules(v22_dir,v21_dir)
    core_original,paths,model,audit,source,candidates,sel=v22.preflight(v21_dir,base,protocol)
    require(core is core_original,'HOLD_CORE_SHADOWED')
    original_input_lock=read_json(raw_out/'V22_INPUT_LOCK.json')
    require(source==original_input_lock.get('source_sha256'),'HOLD_REPLAY_V16_V19_V20_PHYSICS_SOURCE_MISMATCH')
    require(sel==original_input_lock.get('selection'),'HOLD_REPLAY_V18_PK_CANDIDATE_SELECTION_MISMATCH')
    lock={'v22_csv_sha256':EXPECTED_V22_CSV_SHA,
        'v22_source_sha256':EXPECTED_SHA['v22_core.py'],
        'v21_source_sha256':EXPECTED_SHA['v21_core.py'],
        'original_protocol_sha256':sha256(protocol),
        'original_input_lock_sha256':sha256(raw_out/'V22_INPUT_LOCK.json'),
        'original_v22_bdf_final_gate_sha256':sha256(raw_out/'V22_BDF_FINAL_GATE.json'),
        'total_plan_cases':5120,'max_replays':max_replays,
        'refinement_controls':refine_controls,
        'ablation_controls':ablation_controls,
        'all_24_month_original_BDF':True,
        'three_original_release_modes':list(MODES)}
    lock_path=out/'VALIDATION_REPLAY_INPUT_LOCK.json'
    if lock_path.exists():
        require(read_json(lock_path)==lock,'HOLD_REPLAY_CONFIG_CHANGED_ON_RESUME')
    else:atomic_json(lock_path,lock)
    ckpt=init_checkpoint(out/'VALIDATION_REPLAY.sqlite')
    done=checkpoint_results(ckpt)
    target=plan.sort_values('plan_rank').head(max_replays)
    requested=[]
    for item in target.to_dict('records'):
        key=(int(item['seed']),int(item['idx']))
        if key not in done:requested.append(item)
    log('REPLAY_START',requested=max_replays,already_finished=len(done),pending=len(requested),
        wall_budget_hours=max_hours,workers=workers,actual_BDF=True)
    if not requested:
        report=report_replays(ckpt,out,max_replays)
        ckpt.close();return report
    desired=pd.DataFrame(requested)
    designs=make_full_scenario_design_map(desired,raw_out,v22,core,model,audit,candidates)
    tasks=iter(requested)
    active={};remaining=len(requested)
    n_completed=0;new_nfev=0;last=time.monotonic();paused=False;error=False
    # Capped by the *work necessary* or budget, NOT a fake timer or padding task.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def submit_one():
            nonlocal remaining
            try:item=next(tasks)
            except StopIteration:return False
            key=(int(item['seed']),int(item['idx']))
            job=make_replay_job(item,designs[key],raw_out,v22_dir=v22_dir,v21_dir=v21_dir,
                base=base,protocol=protocol,refine_controls=refine_controls,ablation_controls=ablation_controls)
            future=pool.submit(worker_replay,job)
            active[future]=key
            remaining-=1
            return True
        for _ in range(min(workers,len(requested))):submit_one()
        while active:
            ready,_=wait(tuple(active),timeout=10,return_when=FIRST_COMPLETED)
            for f in ready:
                s,i=active.pop(f)
                try:result=f.result()
                except BaseException as exc:
                    result={'seed':s,'idx':i,'scenario_sha256':designs[(s,i)]['scenario_sha256'],
                            'status':'HOLD_BDF_WORKER_FAILED','error_type':type(exc).__name__,
                            'error_message':str(exc),'traceback':traceback.format_exception_only(type(exc),exc),
                            'new_BDF_wall_s':0.,'new_real_BDF_nfev':0,
                            'not_patient_prediction':True}
                    error=True
                ckpt_save(ckpt,s,i,designs[(s,i)]['scenario_sha256'],result)
                n_completed+=1;new_nfev+=int(result.get('new_real_BDF_nfev',0))
                if result['status']!='PASS':error=True
                if n_completed==1 or n_completed%16==0:
                    log('BDF_CASES',new_completed=n_completed,target=max_replays,nfev=new_nfev,
                        status=result['status'],elapsed_h=round((time.monotonic()-t0)/3600,4))
            now=time.monotonic()
            if (out/'REQUEST_STOP.flag').exists() or now-t0>=max_hours*3600:
                paused=True
            if not paused and not error:
                while len(active)<workers and submit_one():pass
            if now-last>=10 or paused:
                progress={'newly_completed':n_completed,'already_finished':len(done),
                    'requested_total':max_replays,'pending_not_started':remaining,
                    'running':len(active),'elapsed_h':(now-t0)/3600,
                    'new_real_BDF_rhs_evaluations':new_nfev,
                    'paused_or_budget_exhausted':paused,'hold_failure_detected':error}
                atomic_json(out/'LIVE_PROGRESS.json',progress)
                log('PROGRESS',**progress);last=now
            if error and not active:break
    report=report_replays(ckpt,out,max_replays)
    report['stopped_by_budget_or_user']=paused
    report['worker_failure_occurred']=error
    atomic_json(out/'REPLAY_FINAL_STATUS.json',report)
    ckpt.close()
    log('BDF_REPLAY_FINISHED',**{k:report[k] for k in ('status','n_completed','n_requested','n_validated','n_failed','n_real_BDF_rhs_evaluations')})
    return report


def report_replays(db,out,max_replays):
    rows=[];shapley=[]
    for seed,idx,hashv,stat,wall,nfev,j in db.execute('SELECT seed,idx,scenario_sha,status,wall_s,nfev,result_json FROM job ORDER BY seed,idx'):
        r=json.loads(j)
        rec={'seed':seed,'idx':idx,'sha256':hashv,'status':stat,'wall_s':wall,'nfev':nfev,
             'max_scale_normalized_BDF_state_error':r.get('max_scale_normalized_BDF_state_error'),
             'original_cache_nfev':r.get('original_cache_nfev'),
             'refinement_performed':bool(r.get('do_refinement')),
             'ablation_performed':bool(r.get('do_shapley_ablation')),
             'refinement_pass':r.get('refinement',{}).get('pass_threshold_1e_minus_3') if r.get('refinement') else None,
             'refinement_max_scale_error':r.get('refinement',{}).get('max_scale_normalized_ode_refinement_error') if r.get('refinement') else None,
             'failure_message':r.get('error_message','')}
        for v in r.get('vascular',[]):
            rate=v['rate']
            for k,x in v.items():
                if k!='rate':rec[k+'_'+rate]=x
        rows.append(rec)
        for entry in r.get('mechanism_shapley',[]):shapley.append({'seed':seed,'idx':idx,**entry})
    write_csv(out/'BDF_REAL_REPLAY_CASE_REPORT.csv',rows)
    write_csv(out/'BDF_REAL_THREE_AXIS_SHAPLEY_ABLATIONS.csv',shapley)
    passed=sum(r['status']=='PASS' for r in rows)
    n=sum(int(r['nfev']) for r in rows)
    report={'n_completed':len(rows),'n_requested':max_replays,'n_validated':passed,
        'n_failed':len(rows)-passed,'n_real_BDF_rhs_evaluations':n,
        'n_genuine_24m_refinements':sum(bool(x['refinement_performed']) for x in rows),
        'n_genuine_24m_seven_axis_ablations':sum(bool(x['ablation_performed']) for x in rows),
        'scenario_records_not_patient_replicates':True,'clinical_claim_allowed':False,
        'n_3d_CFD_runs':0,
        'status':('PASS_COMPLETED_REQUESTED_BDF_REPLAY_NOT_CLINICAL' if len(rows)>=max_replays and passed==max_replays
                  else 'HOLD_NUMERIC_BDF_FAILURE' if passed<len(rows)
                  else 'PARTIAL_BDF_REPLAY_RESUME_REQUIRED')}
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('v22','v21','base','protocol','research-output','out'):p.add_argument('--'+k,required=True)
    p.add_argument('--phase',choices=['audit','replay','all'],default='all')
    p.add_argument('--max-replays',type=int,default=5120)
    p.add_argument('--max-hours',type=float,default=18)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--refine-controls',type=int,default=128)
    p.add_argument('--ablation-controls',type=int,default=64)
    args=p.parse_args(argv)
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    import scipy,pandas
    atomic_json(out/'SOFTWARE_ENVIRONMENT.json',{'python':sys.version,'os':platform.platform(),
        'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pandas.__version__,
        'software_v22_deep_sha256':sha256(__file__),
        'original_v22_sha256':EXPECTED_SHA['v22_core.py'],
        'original_v21_sha256':EXPECTED_SHA['v21_core.py'],
        'origin_0D_sha256':EXPECTED_V22_CSV_SHA,'generated_as_model_numerics_not_clinical':True})
    original_output=Path(args.research_output).resolve()
    csv_path=original_output/'V22_0D_REMODELING_NOT_CFD.csv'
    require(csv_path.is_file(),'HOLD_SOURCE_CSV_MISSING:'+str(csv_path))
    start=time.monotonic()
    inputs={'v22_dir':Path(args.v22).resolve(),'v21_dir':Path(args.v21).resolve(),
            'base':Path(args.base).resolve(),'protocol':Path(args.protocol).resolve()}
    audit_path=out/'AUDIT_0D_STATUS.json'
    if args.phase in ('audit','all') or not audit_path.exists():
        paired,plan,report=audit_0d(csv_path,original_output,out,**inputs)
    else:
        lock=source_lock(inputs['v22_dir'],inputs['v21_dir'],original_output,csv_path,
                         base=inputs['base'],protocol=inputs['protocol'])
        plan=pd.read_csv(out/'V22_BDF_REPLAY_PLAN_5120.csv')
        require(len(plan)==5120,'HOLD_REPLAY_PLAN_CORRUPTED')
        report=read_json(audit_path)
        require(report['source_lock']['source_csv_sha256']==lock['source_csv_sha256'],
                'HOLD_AUDIT_SOURCE_CHANGED')
    if args.phase=='audit':
        log('AUDIT_ONLY_FINISHED',seconds=round(time.monotonic()-start,2))
        return 0
    result=resume_replay(plan,original_output,out,**inputs,workers=args.workers,
             max_hours=args.max_hours,max_replays=args.max_replays,
             refine_controls=args.refine_controls,ablation_controls=args.ablation_controls)
    final={'audit_completed':True,'replay':result,'runtime_hours':(time.monotonic()-start)/3600,
           'no_clinical_efficacy_claim':True,'not_patient_digital_twin':True}
    atomic_json(out/'FINAL_PACKAGE_STATUS.json',final)
    return 0 if result['status']=='PASS_COMPLETED_REQUESTED_BDF_REPLAY_NOT_CLINICAL' else 3

if __name__=='__main__':
    try:sys.exit(main())
    except (Hold,Exception) as e:
        log('HOLD_ERROR',kind=type(e).__name__,reason=str(e))
        traceback.print_exc()
        sys.exit(2)
