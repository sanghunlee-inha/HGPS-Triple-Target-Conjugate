#!/usr/bin/env python3
"""Evidence-locked re-simulation and manuscript audit of V9/V10 synthetic PK/PD.

No new pharmacological parameters, no interpolation of biological effects, no training.
Only the school RTX 5090 runs original V9 ODE at exact saved full-run priors.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import pandas as pd

V6_SHA='7ae85ce32c424fe5f7c3da58610ed11eb72072b9f4615ab02ea93eaad760eee4'
V9_FORK_SHA='d43ff7ac5789d2fa72cc1f12885bb92a1266f8603ebbce9e5ce3897048c6dd4c'
CFG_SHA='1c8af54663b17e503b0ed4f5c1fd1a3184b2e54b5f10c8e2ae0a976e9ea5365b'
CONTEXT='INHIBITION_SRSF1_COMPATIBLE'
DOSES=(0.001,0.1,3.0)
SEEDS=(11,23,37,51,73)
FACTORS=('F000','F100','F010','F001','F110','F101','F011','F111')
TRACE_STATE_NAMES=['plasma_intact_nM','vascular_free_intact_nM',
                   'PrPC_free_tissue_nM','PrPC_bound_tissue_nM','PrPC_internalized_tissue_nM',
                   'endosome_intact_cell_nM','cytosol_intact_cell_nM','PrPC_TBK1_remnant_cell_nM',
                   'TBK1_free_cell_nM','miR4651_free_cell_nM','AGO2_loaded_cell_nM',
                   'TBK1_activity_rel','pIRF3_rel','pATAD3A_S321_rel','SRSF1_rel','progerin_rel',
                   'endosome_TBK1_remnant_cell_nM','endosome_guide_cell_nM',
                   'TBK1_equivalent_elimination_plasma_nM','miRNA_equivalent_elimination_plasma_nM',
                   'SC_depot_nM_equiv','cumulative_administered_nM_equiv']

def sha(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def jread(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def savejson(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');os.replace(temp,p)

def require(flag,msg):
    if not flag:raise RuntimeError(msg)

def verify_hash(p,want):
    require(p.is_file(),'MISSING_SOURCE '+str(p))
    got=sha(p);require(got==want,f'SOURCE_HASH_FAIL {p}: {got} != {want}')
    return got

def preflight(v9,v10):
    a=v9.parent.parent;b=v10.parent.parent
    for p,w in [(a/'src/model.py',V6_SHA),(b/'src/model.py',V6_SHA),
                (a/'src/model_v9_schedule.py',V9_FORK_SHA),(b/'src/model_v9_schedule.py',V9_FORK_SHA),
                (a/'config/study.json',CFG_SHA),(b/'config/study.json',CFG_SHA)]:verify_hash(p,w)
    gate9=jread(v9/'V9_SCIENTIFIC_GATE.json');gate10=jread(v10/'V10_SCIENTIFIC_GATE.json')
    require(gate9.get('n_unique_prior_profiles')==25000,'NOT_FULL_V9_25000')
    require(gate9.get('n_profile_regimen_integrations')==825000,'NOT_FULL_V9_825000')
    require(gate9.get('mass_balance_gate')=='PASS','MASS_BALANCE_NOT_PASS')
    require(gate10.get('n_v9_paired_prior_profiles')==25000,'NOT_FULL_V10_PAIRED')
    require(gate10.get('structural_TBK1_to_progerin_direct_coupling')=='ABSENT_IN_LOCKED_V6_EQUATIONS',
            'UNEXPECTED_TBK1_LINK_REQUIRES_MODEL_REVIEW')
    csv9=v9/'V9_DOSE_PKPD_5SEED_DISTRIBUTIONS.csv'
    csv10=v10/'V10_PAIRED_MECHANISM_DOSE_SUMMARY.csv'
    for p in (csv9,csv10):require(p.is_file(),'MISSING_ORIGINAL_OUTPUT '+str(p))
    d9=pd.read_csv(csv9,encoding='utf-8-sig');d10=pd.read_csv(csv10,encoding='utf-8-sig')
    require(set(d9.seed)==set(SEEDS) and len(d9)==660,'V9_SUMMARY_ROW_SEED_FAIL')
    require(set(d10.seed)==set(SEEDS),'V10_SEEDS_MISSING')
    d9=d9[(d9.scenario==CONTEXT)&(d9.route=='iv')&(np.isclose(d9.interval_h,24.))&(d9.dose_mgkg>0)].copy()
    require(len(d9)==5*8 and set(np.round(d9.dose_mgkg,6))=={.001,.003,.01,.03,.1,.3,1.,3.},'V9_DOSE_GRID_MISMATCH')
    d10=d10[(d10.context==CONTEXT)&(d10.regimen=='IV_3_Q24')&(d10.scenario.isin(FACTORS))].copy()
    require(len(d10)==40 and set(d10.scenario)==set(FACTORS),'V10_FACTORIAL_INCOMPLETE')
    require(not d9.duplicated(['seed','dose_mgkg']).any(),'DUPLICATE_V9_RESPONSE')
    require(not d10.duplicated(['seed','scenario']).any(),'DUPLICATE_V10_RESPONSE')
    x=d9[np.isclose(d9.dose_mgkg,3.)].sort_values('seed')['progerin_suppression_mean_fraction_median'].to_numpy()
    y=d10[d10.scenario=='F000'].sort_values('seed')['progerin_suppression_mean_fraction_median'].to_numpy()
    diff=float(np.max(np.abs(x-y)))
    require(diff<=5e-6,'V9_V10_F000_BASELINE_DISAGREEMENT '+repr(diff))
    result={'source_v6_sha256':V6_SHA,'source_v9_fork_sha256':V9_FORK_SHA,
            'v9_gate_sha256':sha(v9/'V9_SCIENTIFIC_GATE.json'),
            'v10_gate_sha256':sha(v10/'V10_SCIENTIFIC_GATE.json'),
            'v9_csv_sha256':sha(csv9),'v10_csv_sha256':sha(csv10),
            'v9_v10_3mgkg_F000_max_absolute_fraction_difference':diff,
            'model_structural_TBK1_to_Progerin':'NOT_CONNECTED',
            'source_full_prior_profiles':25000,'expected_context_profiles_per_seed':1250,
            'real_observed_PKPD':'NOT_PROVIDED','dose_calibrated':'NO',
            'required_science_status':'HOLD_NO_OBSERVED_PK_PD'}
    return d9,d10,result

def source_import(v9):
    import importlib
    folder=str(v9.parent.parent/'src')
    sys.path.insert(0,folder)
    for module in ('model','model_v9_schedule','v9_engine'):
        if module in sys.modules:
            require(str(Path(sys.modules[module].__file__).resolve()).startswith(str(Path(folder).resolve())),
                    'MODULE_SHADOWING_'+module)
    m=importlib.import_module('model'); s=importlib.import_module('model_v9_schedule');e=importlib.import_module('v9_engine')
    require(Path(m.__file__).resolve()==Path(folder,'model.py').resolve(),'WRONG_MODEL_MODULE')
    return m,s,e

def prior(v9,seed):
    src=v9/f'V9_PRIOR_SEED_{seed}.npz';meta=src.with_suffix('.json')
    require(src.is_file() and meta.is_file(),'MISSING_SAVED_V9_PRIOR '+str(seed))
    md=jread(meta);require(sha(src)==md.get('sha256'),'CORRUPT_V9_PRIOR '+str(seed))
    require(md.get('n')==5000,'PRIOR_NOT_5000_'+str(seed))
    lock=jread(v9/'V9_RUN_LOCK.json')
    require(md.get('lock_sha')==lock.get('lock_sha'),'V9_PRIOR_RUN_LOCK_DISAGREEMENT')
    with np.load(src,allow_pickle=False) as z:
        ctx=z['context']
        indices=np.flatnonzero(ctx==CONTEXT)
        require(len(indices)==1250,'UNEXPECTED_CONTEXT_COUNTS '+str(seed))
        p={k[3:]:z[k][indices].copy() for k in z.files if k.startswith('P__')}
        phys={k[3:]:z[k][indices].copy() for k in z.files if k.startswith('X__')}
    return p,phys,indices,sha(src)

def checkpoint_reference(v9,seed,indices,arms,selected_regimen_names,e):
    proto=jread(v9.parent.parent/'config/v9_protocol.json')
    regmap={r['name']:i for i,r in enumerate(arms)}
    cols=[regmap[n] for n in selected_regimen_names]
    values=[]
    for start in range(0,5000,proto['profile_batch']):
        stop=min(start+proto['profile_batch'],5000)
        f=v9/'CHECKPOINTS'/f'SEED_{seed}_PROFILE_{start:05d}_{stop:05d}.npz'
        md=f.with_suffix('.json');require(f.is_file() and md.is_file(),'MISSING_V9_CHECKPOINT '+str(f))
        m=jread(md);require(sha(f)==m.get('sha256'),'V9_CHECKPOINT_CORRUPT '+str(f))
        local=indices[(indices>=start)&(indices<stop)]-start
        if len(local):
            with np.load(f,allow_pickle=False) as z:
                data=z['metrics']
                require(data.shape==(stop-start,len(arms),len(e.METRIC_KEYS)),
                        'V9_CHECKPOINT_SHAPE_FAIL')
                values.append(data[local][:,cols,:].copy())
    require(bool(values),'NO_CONTEXT_REFERENCE')
    return np.concatenate(values,axis=0)

def trace_from_direct(prior_p,phys,arms,cfg,proto,device,s,e):
    import torch
    allregs=[dict(r) for r in arms]
    out,tr=s.simulate_schedule(prior_p,allregs,cfg,**phys,device=device,duration_h=float(proto['duration_h']),
                                dt_h=float(proto['dt_h']),trace=True)
    require(len(tr)>0,'NO_REAL_ODE_TRACE')
    require(out.shape==(1250,len(arms),len(s.V9_METRICS)),'SIMULATED_METRIC_SHAPE_FAIL')
    require(np.isfinite(out).all(),'NONFINITE_SIMULATION')
    # Keep V9 mass/conservation thresholds unchanged; no permissive bypass.
    e._check_output(out,proto)
    rows=[]
    names=['time_h','nominal_dose_mgkg','route']+TRACE_STATE_NAMES
    require(len(tr[0])==len(names),'STATE_TRACE_SCHEMA_CHANGED')
    for i,ob in enumerate(tr):
        j=i%len(arms)
        require(abs(float(ob[1])-arms[j]['dose_mgkg'])<1e-8,'TRACE_REGIMEN_MISALIGNED')
        d=dict(zip(names,ob))
        occ=float(d['PrPC_bound_tissue_nM'])/float(cfg['PrPC_surface_capacity_tissue_nM'])*100.
        rows.append({'dose_mgkg':float(arms[j]['dose_mgkg']),'time_h':float(d['time_h']),
                     'plasma_intact_nM':float(d['plasma_intact_nM']),
                     'vascular_free_intact_nM':float(d['vascular_free_intact_nM']),
                     'PrPC_occupancy_pct':occ,
                     'AGO2_loaded_cell_nM':float(d['AGO2_loaded_cell_nM']),
                     'TBK1_inhibition_pct':max(0,1-float(d['TBK1_activity_rel']))*100,
                     'SRSF1_suppression_pct':max(0,1-float(d['SRSF1_rel']))*100,
                     'Progerin_suppression_pct':max(0,1-float(d['progerin_rel']))*100})
    return out,np.asarray(rows,dtype=object)

def run_resimulation(v9,out,source_info,device='cuda'):
    import torch
    require(device=='cuda' and torch.cuda.is_available(),'SCHOOL_CUDA_REQUIRED')
    require('RTX 5090' in torch.cuda.get_device_name(0),'SCHOOL_RTX5090_REQUIRED')
    m,s,e=source_import(v9)
    cfg=jread(v9.parent.parent/'config/study.json')
    proto=jread(v9.parent.parent/'config/v9_protocol.json')
    arms=e.get_regimens(proto)
    select=[next((r for r in arms if r['route']=='iv' and abs(r['interval_h']-24)<1e-9
                 and abs(r['dose_mgkg']-x)<1e-10),None) for x in DOSES]
    require(all(x is not None for x in select),'REQUIRED_IV_Q24_REGIMEN_MISSING')
    fields=['dose_mgkg','time_h','plasma_intact_nM','vascular_free_intact_nM',
            'PrPC_occupancy_pct','AGO2_loaded_cell_nM','TBK1_inhibition_pct',
            'SRSF1_suppression_pct','Progerin_suppression_pct']
    locked={'source_info':source_info,'seeds':SEEDS,'doses':DOSES,'context':CONTEXT,'duration_h':336.,
            'dt_h':0.125,'script_sha256':sha(Path(__file__))}
    lock=canon(locked)
    out.mkdir(parents=True,exist_ok=True)
    meta=out/'V11_RUN_LOCK.json'
    if meta.exists():require(jread(meta).get('lock_sha')==lock,'V11_OUTPUT_RUN_LOCK_MISMATCH_NEW_DIRECTORY_REQUIRED')
    else:savejson(meta,{'lock_sha':lock,'parameters':locked})
    audits=[]
    for seed in SEEDS:
        csvfile=out/f'V11_REAL_ODE_TRACE_SEED_{seed}.csv'; jsonfile=out/f'V11_REPRO_QA_SEED_{seed}.json'
        if csvfile.is_file() and jsonfile.is_file():
            q=jread(jsonfile)
            require(q.get('lock_sha')==lock and q.get('trace_sha256')==sha(csvfile) and q.get('checkpoint_replay')=='PASS',
                    'V11_CHECKPOINT_RESUME_FAIL '+str(seed))
            audits.append(q);print('V11_RESUME_VERIFIED_SEED',seed,flush=True);continue
        require(not csvfile.exists() and not jsonfile.exists(),'V11_PARTIAL_SEED_OUTPUT '+str(seed))
        p,x,ix,p_sha=prior(v9,seed)
        t=time.monotonic()
        sim,records=trace_from_direct(p,x,select,cfg,proto,device,s,e)
        ref=checkpoint_reference(v9,seed,ix,arms,[r['name'] for r in select],e)
        recomputed=sim[:,:,e.METRIC_INDICES].astype(np.float32)
        require(recomputed.shape==ref.shape,'V9_REPLAY_SHAPE_FAIL')
        err=np.abs(recomputed.astype('float64')-ref.astype('float64'))
        allowed=1e-6+3e-4*np.abs(ref.astype('float64'))
        ratio=err/np.maximum(allowed,1e-20)
        worst=float(np.max(ratio))
        require(worst<=1.,f'ODE_V9_CHECKPOINT_NUMERIC_MISMATCH seed={seed} normalized_max={worst:.5g}')
        rows=pd.DataFrame.from_records(records,columns=fields)
        rows.insert(0,'seed',seed)
        # Verify trace has exactly the same step sampling grid across the three doses.
        for dose in DOSES:
            r=rows[np.isclose(rows.dose_mgkg,dose)]
            require(len(r)>=150 and abs(r.time_h.max()-336)<1e-6,'INCOMPLETE_336H_TRACE')
            require(r.time_h.is_monotonic_increasing,'NONMONOTONIC_TRACE')
        temp=csvfile.with_suffix('.tmp')
        rows.to_csv(temp,index=False,encoding='utf-8-sig',float_format='%.10g');os.replace(temp,csvfile)
        q={'seed':seed,'lock_sha':lock,'prior_sha256':p_sha,'trace_sha256':sha(csvfile),
           'n_real_profile_resimulated':len(ix),'n_regimens':len(select),
           'n_metric_values_checked':int(recomputed.size),
           'max_normalized_replay_error':worst,'checkpoint_replay':'PASS',
           'original_model_source_unchanged':True,'duration_s':time.monotonic()-t}
        savejson(jsonfile,q);audits.append(q)
        print(f'V11_REPLAY_PASS seed={seed} profiles={len(ix)} '
              f'values_checked={recomputed.size} max_norm_error={worst:.4g} elapsed_s={q["duration_s"]:.1f}',flush=True)
    require(len(audits)==5,'NOT_ALL_SEED_AUDITS')
    return audits

def combined_trace(out):
    frames=[]
    for seed in SEEDS:
        f=out/f'V11_REAL_ODE_TRACE_SEED_{seed}.csv'
        q=jread(out/f'V11_REPRO_QA_SEED_{seed}.json')
        require(sha(f)==q['trace_sha256'],'CORRUPT_V11_TRACE '+str(seed))
        frames.append(pd.read_csv(f,encoding='utf-8-sig'))
    all_df=pd.concat(frames,ignore_index=True)
    fields=[c for c in all_df.columns if c not in ('seed','dose_mgkg','time_h')]
    grouped=all_df.groupby(['dose_mgkg','time_h'],as_index=False)
    result=grouped[fields].median()
    for c in fields:
        result[c+'_min_of_seed_means']=grouped[c].min()[c].to_numpy()
        result[c+'_max_of_seed_means']=grouped[c].max()[c].to_numpy()
    result.to_csv(out/'V11_TIMECOURSE_5SEED_MEDIAN.csv',index=False,encoding='utf-8-sig')
    return result

def collapse_series(d9,d10):
    dose_rows=[]
    specs={'TBK1_inhibition_peak_fraction':'TBK1 peak inhibition',
           'SRSF1_suppression_mean_fraction':'SRSF1 time-mean suppression',
           'progerin_suppression_mean_fraction':'Progerin time-mean suppression'}
    for dose,gr in d9.groupby('dose_mgkg',sort=True):
        item={'dose_mgkg':dose,'n_seeds':len(gr)}
        for k in specs:
            vals=gr[k+'_median'].to_numpy(dtype=float)*100
            item[k+'_five_seed_median_pct']=float(np.median(vals))
            item[k+'_five_seed_min_pct']=float(vals.min())
            item[k+'_five_seed_max_pct']=float(vals.max())
        dose_rows.append(item)
    fr=[]
    for code in FACTORS:
        gr=d10[d10.scenario==code]
        vals=gr['progerin_suppression_mean_fraction_median'].to_numpy(dtype=float)*100
        require(len(vals)==5,'MISSING_5_SEED_FACTOR '+code)
        fr.append({'factor':code[1:],'five_seed_median_pct':float(np.median(vals)),
                   'five_seed_min_pct':float(vals.min()),'five_seed_max_pct':float(vals.max())})
    return pd.DataFrame(dose_rows),pd.DataFrame(fr),specs

def make_figure(tr,d9,d10,out,dpi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    dose,fac,specs=collapse_series(d9,d10)
    dose.to_csv(out/'V11_PD_DOSE_RESPONSE_LOCKED.csv',index=False,encoding='utf-8-sig')
    fac.to_csv(out/'V11_V10_FACTORIAL_LOCKED.csv',index=False,encoding='utf-8-sig')
    c={.001:'#1D4F91',.1:'#078A91',3.:'#C66C36'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':23,
        'axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','text.color':'black',
        'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.right':False,'axes.spines.top':False})
    fig=plt.figure(figsize=(25,18),layout='constrained')
    grid=fig.add_gridspec(3,3,height_ratios=[1,1,.20])
    axes=np.asarray([[fig.add_subplot(grid[i,j]) for j in range(3)] for i in range(2)])
    foot=fig.add_subplot(grid[2,:]);foot.axis('off')
    titles=['Plasma intact conjugate','Vascular free conjugate','PrPᶜ surface occupancy',
            'Intracellular AGO2-loaded guide','Dose-dependent PD','Paired mechanistic perturbations']
    metrics=['plasma_intact_nM','vascular_free_intact_nM','PrPC_occupancy_pct','AGO2_loaded_cell_nM']
    ylabels=['Plasma concentration (nM)','Vascular concentration (nM)','Occupancy (%)','AGO2-loaded guide (nM)']
    for i,(met,ylab) in enumerate(zip(metrics,ylabels)):
        ax=axes.flat[i]
        for level in DOSES:
            z=tr[np.isclose(tr.dose_mgkg,level)].sort_values('time_h')
            x=z.time_h.to_numpy(float);y=z[met].to_numpy(float)
            lo=z[met+'_min_of_seed_means'].to_numpy(float)
            hi=z[met+'_max_of_seed_means'].to_numpy(float)
            ax.plot(x,y,color=c[level],lw=3,label=f'{level:g} mg/kg')
            ax.fill_between(x,lo,hi,color=c[level],alpha=.13,lw=0)
        ax.set(xlabel='Time after first dose (h)',ylabel=ylab,xlim=(0,336))
        ax.set_xticks([0,72,144,216,288,336]);ax.grid(alpha=.12)
        if i==0:ax.legend(loc='upper right',fontsize=23,frameon=False)
    ax=axes.flat[4]
    pdcolors=['#182F4D','#5596AD','#C66C36']
    for (metric,label),color in zip(specs.items(),pdcolors):
        y=dose[metric+'_five_seed_median_pct'].to_numpy(float)
        lo=dose[metric+'_five_seed_min_pct'].to_numpy(float)
        hi=dose[metric+'_five_seed_max_pct'].to_numpy(float)
        x=dose.dose_mgkg.to_numpy(float)
        ax.plot(x,y,marker='o',lw=3,ms=10,label={'TBK1_inhibition_peak_fraction':'TBK1 (peak)', 'SRSF1_suppression_mean_fraction':'SRSF1 (mean)', 'progerin_suppression_mean_fraction':'Progerin (mean)'}[metric],color=color)
        ax.fill_between(x,lo,hi,color=color,alpha=.12)
    ax.set_xscale('log');ax.set_xticks([.001,.01,.1,1,3]);ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
    ax.set(xlabel='Virtual IV q24h dose (mg/kg)',ylabel='Modeled response (%)');ax.grid(alpha=.12)
    ax.legend(loc='upper left',fontsize=21,frameon=False)
    ax=axes.flat[5]
    bars=ax.bar(range(8),fac.five_seed_median_pct.to_numpy(float),color=['#9AB0BB']+['#25829A']*6+['#C66C36'])
    for b,v in zip(bars,fac.five_seed_median_pct):
        ax.text(b.get_x()+b.get_width()/2,v+max(0.05,float(fac.five_seed_median_pct.max())*.025),
                f'{v:.3g}%',ha='center',va='bottom',fontsize=21,weight='bold',color='black')
    ax.set_xticks(range(8),fac.factor.to_list())
    ax.set(xlabel='Factor code (D / T / C)',ylabel='Time-mean Progerin reduction (%)')
    ax.set_ylim(0,max(fac.five_seed_median_pct)*1.19);ax.grid(axis='y',alpha=.12)
    for i,(a,t) in enumerate(zip(axes.flat,titles)):
        a.set_title(t,fontsize=26,weight='bold',pad=22)
        a.text(-.14,1.085,'ABCDEF'[i],transform=a.transAxes,fontsize=34,weight='bold',ha='left',va='center')
        a.tick_params(labelsize=23)
    fig.suptitle('V9–V10 | Source-verified semimechanistic PK/PD analysis',fontsize=31,weight='bold')
    foot.text(.5,.68,'Synthetic ODE only | IV q24h, 336 h | 5 seeds |  No measured PK/PD or validated dose',
              ha='center',va='center',fontsize=23)
    foot.text(.5,.2,'E: TBK1 peak; SRSF1/Progerin time-mean. F: D = delivery, T = Srsf1 targeting, C = coupling.',
              ha='center',va='center',fontsize=21)
    name=out/'V11_MAIN_FIGURE_SOURCE_VERIFIED'
    for ext in ('png','pdf','svg'):
        fig.savefig(str(name)+'.'+ext,dpi=dpi if ext=='png' else None,facecolor='white',bbox_inches='tight')
    plt.close(fig)
    return dose,fac

def observed_gate(root):
    obs=root/'inputs'/'observed_pkpd.csv'
    if not obs.exists():return 'HOLD_NO_OBSERVED_PK_PD'
    with obs.open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    if not rows:return 'HOLD_NO_OBSERVED_PK_PD'
    # We cannot treat unvalidated filled CSV as calibrated evidence.
    return 'HOLD_OBSERVED_ROWS_PRESENT_ASSAY_QC_AND_INDEPENDENT_VALIDATION_REQUIRED'

def report(out,dose,fac,source_info,audits,obs):
    sel=dose[np.isclose(dose.dose_mgkg,3.)].iloc[0]
    base=float(fac.loc[fac.factor=='000','five_seed_median_pct'].iloc[0]); triple=float(fac.loc[fac.factor=='111','five_seed_median_pct'].iloc[0])
    text=f'''# V11 근거잠금형 V9–V10 PK/PD 검증 보고서

**과학적 상태: {obs}**  
실제 접합체의 분자량·결합·마우스 혈중/조직 PK·Srsf1 표적성·독성이 미검증입니다. **가상 mg/kg는 투여 권고가 아닙니다.**

## 출처 및 계산 범위
- V6 원본 모델 SHA-256: `{V6_SHA}`; V9 반복투여 수식 SHA-256: `{V9_FORK_SHA}`.
- 저장된 V9 25,000 가상 프로파일의 5시드 각각에서 동일 가설 조건 `{CONTEXT}`에 속하는 1,250개를 사용했습니다. **총 {sum(q['n_real_profile_resimulated'] for q in audits):,}개**의 기존 프로파일을 새로 생성하지 않고 직접 재적분했습니다.
- 동일 IV q24h, 총 336시간, 세 가지 가상 명목 투여량 {', '.join(map(str,DOSES))} mg/kg를 적용했습니다.
- V9 체크포인트의 개별 결과 지표 **{sum(q['n_metric_values_checked'] for q in audits):,}개**와 대조해 수치 허용 범위 내 재현을 확인했습니다. Figure A–D의 선은 이 검증된 ODE 재적분 시계열입니다.
- Figure A–D: 5개 시드별 프로파일 평균의 시드 간 중앙값. 음영: 5개 시드 평균의 최솟값–최댓값. 통계적 95% 신뢰구간이 아닙니다.
- Figure E–F: 원본 V9/V10 CSV에서 5시드별 개별 프로파일 중앙값의 중앙값을 요약했습니다. A–D와 E–F는 서로 다른 요약 통계이며 동일한 개체별 중앙 시계열이 아닙니다.

## 정량적 주요 결과
- 3 mg/kg IV q24h, 가상 Progerin 시간평균 감소: **{base:.5g}%** (원래 V9와 V10 F000의 시드별 중앙값 일치).
- 3 mg/kg IV q24h, 세 가지 가상 병목 개선 F111 조건: **{triple:.5g}%**. 이는 실제 실험 효능이 아니라 가정값 교란의 ODE 출력입니다.
- 같은 3 mg/kg IV q24h 조건에서 TBK1 최대 억제의 5시드 중앙값 중앙값: **{sel['TBK1_inhibition_peak_fraction_five_seed_median_pct']:.5g}%**.
- 기존 V9의 96개 가설 조건부 용량창은 모두 후보 없음으로 판정됐으며, 이 수치를 재해석하여 최적 용량을 선정하지 않았습니다.

## 핵심 구조적 한계
- **TBK1은 IRF3·ATAD3A 출력에 연결되지만 SRSF1 및 Progerin 생성/제거 방정식에는 직접 연결되어 있지 않습니다.** TBK1 변화의 Progerin 영향이 0인 것은 현 ODE 구조의 결과이지 실험적 무효성 입증이 아닙니다.
- `mouse_Srsf1_compatibility`, 약물 분자량, 혈장 분포량, 유효 분율 등은 실측 보정된 값이 아닙니다.
- 노출–반응 함수는 외부 실측 PK/PD로 예측검증되지 않았으며, 현재 V6 대리모델 일반화 성능도 엄격 기준 미달입니다.
- 336시간 반복투여는 원래 V6 120시간 수치 검증 범위를 초과합니다. V9의 시간간격·물질수지 검증이 이를 임상적 검증으로 대체하지는 않습니다.
- 3,000 nM 초기농도 연구와 3 mg/kg 반복투여 연구는 동일한 실험조건이 아닙니다.
- 모듈 독립 활성 또는 실측 Srsf1 표적 결합자료 없이는 TBK1 억제와 Progerin 억제 사이 약리학적 인과관계를 주장할 수 없습니다.

## 논문 투고 관련 판정
- **ALLOWED:** `Mechanism-based in silico PK/PD modeling`, `source-locked ODE reproduction`, `conditional mechanistic bottleneck analysis`, `hypothesis-generating dose–exposure simulations`.
- **NOT ALLOWED:** `experimentally validated pharmacokinetics`, `mouse dose optimization`, `actual therapeutic efficacy`, `multimodal deep-learning efficacy prediction`, `TBK1-dependent Progerin efficacy`.
- V11 Figure는 **검증된 실제 생물학적 PK/PD가 아니라 원본 결과와 수치적으로 일치하는 계산 Figure**입니다. 생물학적 검증 자료가 없으면 Nature급 치료효능 증거로 취급할 수 없습니다.
'''
    (out/'V11_REPORT_KO.md').write_text(text,encoding='utf-8')
    claims=[('MODEL_SOURCE_REPRODUCIBILITY','PASS','5-seed V9 checkpoint-replayed ODE'),
            ('MECHANISTIC_MODEL_SCIENTIFIC_VALIDITY','HOLD','Uncalibrated priors and absent independent measured PD'),
            ('TBK1_TO_PROGERIN_IDENTIFIABILITY','HOLD','No direct structural coupling in locked equations'),
            ('REAL_MGKG_OPTIMIZATION','HOLD','MW, PK, binding, bioavailability, tolerability unmeasured'),
            ('MULTIMODAL_DEEP_LEARNING','NOT_SUPPORTED','V9/V10 are mechanistic ODE simulations'),
            ('UNVALIDATED_FIGURE_AS_EVIDENCE','REPLACED','Trace and checkpoints source-locked')]
    pd.DataFrame(claims,columns=['claim','status','basis']).to_csv(out/'V11_MANUSCRIPT_CLAIM_AUDIT.csv',index=False,encoding='utf-8-sig')
    en='''## Results — source-locked, in silico dose–PK/PD analysis (cautious version)\n\nUsing the original, SHA-256-locked semimechanistic ODE model, we re-integrated five saved synthetic parameter populations under identical virtual intravenous q24h regimens. We independently compared per-profile exposure and response summaries against the previously completed V9 checkpoints before generating concentration–time and intracellular-delivery profiles. This reproduces the numerical predictions of the specified model and does not constitute experimental pharmacological validation.\n\nAcross hypothetical doses, modeled plasma exposure and target engagement increased, whereas downstream SRSF1 and progerin changes remained small under the baseline priors. A paired perturbation analysis of delivery, mouse Srsf1 compatibility, and SRSF1–progerin coupling identified combinations of prior assumptions that substantially changed model-predicted progerin responses. These estimates are conditional on hypothetical parameter distributions and are not treatment effect estimates.\n\nThe locked model contains no direct TBK1-to-SRSF1/progerin coupling; thus, a null TBK1 sensitivity of the progerin endpoint is structural rather than evidence that TBK1 is pharmacologically dispensable. Measured compound-specific PK/PD, mouse Srsf1 binding, and safety data are required for calibration, independent validation, or dose selection.\n'''
    (out/'V11_MANUSCRIPT_RESULTS_EN.md').write_text(en,encoding='utf-8')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('audit','all'),default='audit')
    p.add_argument('--v9-project',type=Path,required=True)
    p.add_argument('--v10-project',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dpi',type=int,default=600)
    a=p.parse_args()
    v9=a.v9_project/'outputs/V9_VIRTUAL_DOSE_FULL';v10=a.v10_project/'outputs/V10_PAIRED_BREAKPOINT_FULL'
    df9,df10,src=preflight(v9,v10)
    print('V11_PRECHECK_PASS V6=',V6_SHA,'V9=',V9_FORK_SHA,flush=True)
    if a.mode=='audit':print('V11_AUDIT_PASS; NO_RECOMPUTATION_AND_NO_BIOLOGICAL_VALIDATION_CLAIM');return
    import torch
    require(torch.cuda.is_available() and 'RTX 5090' in torch.cuda.get_device_name(0),'SCHOOL_RTX5090_REQUIRED')
    torch.set_num_threads(22)
    a.out.mkdir(parents=True,exist_ok=True)
    audits=run_resimulation(v9,a.out,src)
    tr=combined_trace(a.out)
    dos,fac=make_figure(tr,df9,df10,a.out,a.dpi)
    obs=observed_gate(a.out.parent.parent)
    report(a.out,dos,fac,src,audits,obs)
    gate={**src,'checkpoint_replay':'PASS','n_original_profiles_reintegrated':sum(z['n_real_profile_resimulated'] for z in audits),
          'n_replayed_values':sum(z['n_metric_values_checked'] for z in audits),
          'figure_source':'RECOMPUTED_LOCKED_V9_ODE_AND_ORIGINAL_V9_V10_CSV',
          'figure_numerically_checked':True,'scientific_gate':obs,
          'TBK1_Progerin_structural_coupling':'ABSENT_NOT_ESTABLISHED',
          'in_vivo_optimal_dose':'HOLD_UNCALIBRATED'}
    savejson(a.out/'V11_SCIENTIFIC_GATE.json',gate)
    print('V11_COMPLETE',a.out,'SCIENCE=',obs,flush=True)
if __name__=='__main__':main()
