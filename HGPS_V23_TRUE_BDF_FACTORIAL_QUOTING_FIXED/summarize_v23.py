#!/usr/bin/env python3
"""Read-only Nature-targeted descriptive analysis of actual V23 BDF outputs.

No ODE reruns or invented values. All intervals describe the prespecified
synthetic QMC design domain; no patient outcomes or experimental synergy.
"""
from __future__ import annotations
import argparse,csv,json,math,pathlib,sqlite3,sys,zlib
from collections import defaultdict,Counter
import numpy as np

HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_v23_true_bdf import SEEDS,MODES,MASKS,HORIZONS,factorial_terms,Hold,read_json,atomic_json,sha

TERM_NAMES=('P1','TBKI','MIR4651','P1xTBKI','P1xMIR4651','TBKIxMIR4651','THREE_WAY','ALL_VS_NO_PAYLOAD','CONJ_111_MINUS_LONA')
TARGETS=('prpc_function','tbk1_activity','srsf1','lmna_cryptic','progerin_total',
         'farnesyl_progerin','ec_activation','vsmc_dysfunction','ecm_stiffness')

def write_csv(path,rows):
 path=pathlib.Path(path)
 if not rows:return
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def safe_vec(x):
 a=np.asarray(x,dtype=float)
 if not np.isfinite(a).all():raise Hold('HOLD_NONFINITE_INPUT_IN_SUMMARY')
 return a

def time_index(times,day):
 matches=np.flatnonzero(np.isclose(times,day,rtol=0,atol=1e-8))
 if len(matches)!=1:raise Hold('HOLD_HORIZON_NOT_UNIQUE:'+str(day))
 return int(matches[0])

def original_0d(t,base,lona,conj,model_outputs):
 """Model's unchanged v22_vascular 0D DOP853; invalid geometry remains HOLD."""
 from source_locked import v22_vascular
 keys=('ec_activation','vsmc_dysfunction','ecm_stiffness')
 loc={name:model_outputs.index(name) for name in keys}
 timeline=[]
 for j,day in enumerate(t):
  row={'horizon_days':float(day)}
  for nm,arr in (('untreated',base),('lona',lona),('conj',conj)):
   for name in keys:row[nm+'_'+name]=float(arr[j,loc[name]])
  timeline.append(row)
 out=[]
 for label,weight in (('low',0.0001),('high',0.001)):
  kwargs={'k_ec':weight,'k_vsmc':weight,'k_ecm':weight}
  try:
   result=v22_vascular.paired_remodel(timeline,**kwargs)
   c=float(result['conj']['resistance_relative_fixed_flow'][-1]);l=float(result['lona']['resistance_relative_fixed_flow'][-1])
   out.append(dict(hypothetical_0D_rate=label,delta_R=c-l,
                   R_conj=c,R_lona=l,status='PASS_NUMERIC_NOT_CLINICAL'))
  except v22_vascular.InvalidVascularInput as e:
   out.append(dict(hypothetical_0D_rate=label,delta_R='',R_conj='',R_lona='',
                   status='HOLD_'+str(e)))
 return out

def summarize(output,include_0d=False,partial=False):
 out=pathlib.Path(output).resolve();lock=read_json(out/'V23_EXPERIMENT_LOCK.json')
 expected=int(lock['cases_per_seed'])*len(SEEDS)
 gatefile=out/'V23_DIRECT_BDF_FINAL_GATE.json'
 if not partial:
  if not gatefile.is_file():raise Hold('HOLD_FULL_BDF_NOT_FINISHED: --partial only for preliminary summaries')
  gate=read_json(gatefile)
  if gate.get('status')!='PASS_DIRECT_BDF_FULL_FACTORIAL_SYNTHETIC_ONLY':
   raise Hold('HOLD_FINAL_GATE_NOT_PASS:'+str(gate.get('status')))
 db=sqlite3.connect('file:'+(out/'V23_DIRECT_BDF.sqlite').as_posix()+'?mode=ro',uri=True)
 try:
  statuses=Counter(dict(db.execute('SELECT status,COUNT(*) FROM runs GROUP BY status').fetchall()))
  count_pass=statuses.get('PASS',0)
  if not partial and (count_pass!=expected or statuses.get('HOLD',0)):
   raise Hold('HOLD_INCOMPLETE_OR_FAILED_SOURCE_ROWS')
  if count_pass<1:raise Hold('HOLD_NO_REAL_BDF_RESULTS')
  from source_locked import v21_core
  proto=v21_core.load_protocol(lock['protocol_path'])
  acc=defaultdict(list);classifier=Counter();strata=defaultdict(list)
  nums=Counter(); vascular_rows=[];pk_rows=[];nonfinite_count=0
  for seed,idx,scenario_sha,compressed in db.execute("SELECT seed,idx,scenario_sha,payload FROM runs WHERE status='PASS' ORDER BY seed,idx"):
   data=json.loads(zlib.decompress(compressed))
   if seed!=data['seed'] or idx!=data['idx'] or scenario_sha!=data['scenario_sha256']:
    raise Hold('HOLD_ROW_SOURCE_SHA_MISMATCH')
   times=safe_vec(data['times_days']);untreated=safe_vec(data['untreated']);lona=safe_vec(data['lonafarnib'])
   columns=data['outputs']
   if len(set(columns))!=len(columns) or any(x not in columns for x in TARGETS):raise Hold('HOLD_OUTPUTS_UNEXPECTED')
   col={s:i for i,s in enumerate(columns)}
   if set(data['conjugate'])!=set(MODES):raise Hold('HOLD_MISSING_RELEASE_MODE')
   for mode in MODES:
    runs=data['conjugate'][mode]
    if set(runs)!=set(map(str,MASKS)):raise Hold('HOLD_MISSING_FACTORIAL_MASK')
    arrays={m:safe_vec(runs[str(m)]['endpoint_states']) for m in MASKS}
    if any(v.shape!=lona.shape for v in arrays.values()):raise Hold('HOLD_FACTORIAL_SHAPE_MISMATCH')
    auc=[runs[str(m)].get('parent_conj_vascular_AUC_nM_day') for m in MASKS]
    if all(x is not None for x in auc):
     endpoint_aucs=safe_vec([float(x[-1]) for x in auc])
     denom=max(1.,float(np.max(np.abs(endpoint_aucs))))
     pk_rows.append(dict(seed=seed,idx=idx,release_mode=mode,
       auc_000_nM_day=float(endpoint_aucs[0]),auc_111_nM_day=float(endpoint_aucs[7]),
       peak_to_peak_auc_over_masks=float(np.ptp(endpoint_aucs)),
       relative_auc_span=float(np.ptp(endpoint_aucs)/denom),
       same_parent_exposure_within_1e_minus_6=bool(float(np.ptp(endpoint_aucs)/denom)<1e-6),
       interpretation='ON_OFF_RELEASE_MASK_NOT_PURE_TARGET_INTERVENTION'))
    elif not partial:
     raise Hold('HOLD_MISSING_PAYLOAD_PK_EXPOSURE_TO_RULE_OUT_ABLATION_CONFOUNDING')
    for day in HORIZONS:
     it=time_index(times,day)
     for name in columns:
      j=col[name]
      values={m:float(arrays[m][it,j]) for m in MASKS}
      terms=factorial_terms(values)
      terms['CONJ_111_MINUS_LONA']=values[7]-float(lona[it,j])
      for term,value in terms.items():
       if not math.isfinite(value):raise Hold('HOLD_NONFINITE_CONTRAST')
       acc[(seed,mode,day,name,term)].append(value)
       if term=='CONJ_111_MINUS_LONA' and name in TARGETS:
        for axis_name,v in (('dose_scale',data['dose_scale_hypothesis']),
                            ('volume_ratio',data['volume_ratio_hypothesis'])):
         # Prespecified source-derived design quartile thresholds: 0.25,1,4 for
         # log uniform [0.25,4]. Boundaries represent design, not patient doses.
         bin_name=('Q1' if v<0.5 else 'Q2' if v<1. else 'Q3' if v<2. else 'Q4')
         strata[(mode,day,name,axis_name,bin_name)].append(value)
     # Model's original conditional classification: NOT equivalence or clinic.
     la={name:float(lona[it,col[name]]) for name in columns}
     co={name:float(arrays[7][it,col[name]]) for name in columns}
     # Same unmodified classification logic, original protocol available at runtime.
     label,better,worse=v21_core.classified(la,co,proto)
     classifier[(seed,mode,day,label)]+=1
    # Optional 0D at the *same original V22 idx%64 source cadence*.
    if include_0d and idx%64==0:
     for mask in MASKS:
      for row in original_0d(times,untreated,lona,arrays[mask],columns):
       vascular_rows.append(dict(seed=seed,idx=idx,release_mode=mode,mask=mask,
          **row,clinical_beneficial_sign_verified=False,not_3D_CFD=True))
   nums[seed]+=1
  if any(nums[x]<1 for x in SEEDS) and not partial:raise Hold('HOLD_MISSING_INDEPENDENT_SEED')
  effect_rows=[]
  for mode in MODES:
   for day in HORIZONS:
    for name in next(iter([data['outputs']])):
     for term in TERM_NAMES:
      seed_means=[];pooled=[]
      for seed in SEEDS:
       values=acc.get((seed,mode,day,name,term),[])
       if not values:continue
       seed_means.append(float(np.mean(values)));pooled.extend(values)
      if not pooled:continue
      z=np.asarray(pooled)
      q=np.quantile(z,[.05,.25,.5,.75,.95]); nseed=len(seed_means)
      numerical_se=float(np.std(seed_means,ddof=1)/np.sqrt(nseed)) if nseed>=2 else float('nan')
      effect_rows.append(dict(release_mode=mode,horizon_days=day,endpoint=name,term=term,
        n_synthetic_design_cases=len(z),n_independent_qmc_seeds=nseed,
        design_mean=float(z.mean()),design_median=float(q[2]),design_p05=float(q[0]),
        design_p25=float(q[1]),design_p75=float(q[3]),design_p95=float(q[4]),
        across_seed_mean_min=float(min(seed_means)),across_seed_mean_max=float(max(seed_means)),
        qmc_se_numerical_not_clinical=numerical_se,
        source='actual_unmodified_v21_BDF',clinical_CI=False,causal_patient_effect=False,
        synergy_validated_in_vitro=False))
  strata_rows=[]
  for key,v in sorted(strata.items()):
   mode,day,name,axis_name,bin_name=key
   vals=np.asarray(v)
   strata_rows.append(dict(release_mode=mode,horizon_days=day,endpoint=name,
     design_coordinate=axis_name,original_design_quartile=bin_name,
     n_synthetic_design_cases=len(vals),median_C_minus_L=float(np.median(vals)),
     mean_C_minus_L=float(vals.mean()),p05=float(np.quantile(vals,.05)),p95=float(np.quantile(vals,.95)),
     fraction_positive_sign_only=float(np.mean(vals>0)),
     not_patient_frequency=True,beneficial_sign_clinically_verified=False))
  class_rows=[dict(seed=seed,release_mode=mode,horizon_days=day,
                   model_screening_class=label,n_design_cases=n,
                   not_clinical_treatment_probability=True)
              for (seed,mode,day,label),n in sorted(classifier.items())]
  suffix='PRELIMINARY' if partial else 'COMPLETE'
  write_csv(out/f'V23_{suffix}_FACTORIAL_DECOMPOSITION.csv',effect_rows)
  write_csv(out/f'V23_{suffix}_LONAFARNIB_CONDITIONAL_STRATA.csv',strata_rows)
  write_csv(out/f'V23_{suffix}_RELEASE_RANKINGS_NOT_CLINICAL.csv',class_rows)
  if include_0d:write_csv(out/f'V23_{suffix}_0D_SOURCE_SAMPLED_NOT_CFD.csv',vascular_rows)
  write_csv(out/f'V23_{suffix}_PAYLOAD_OFF_AXIS_PK_CONFOUNDING.csv',pk_rows)
  result=dict(status='PRELIMINARY_DO_NOT_PUBLISH' if partial else 'PASS_DIRECT_BDF_DECOMPOSITION_NOT_EXPERIMENTAL_VALIDATION',
    expected_cases=expected,read_pass_cases=count_pass,n_held=statuses.get('HOLD',0),
    cases_by_seed=dict(nums),n_factorial_terms=len(effect_rows),
    n_0d_rows=len(vascular_rows),n_parent_PK_confounding_ledger_rows=len(pk_rows),
    n_parent_PK_affected_mask_panels=sum(not x['same_parent_exposure_within_1e_minus_6'] for x in pk_rows),
    has_clinical_efficacy_validation=False,
    has_patient_specific_pharmacokinetics=False,has_experimental_target_validation=False,
    main_primary_claim='model-conditioned interaction decomposition, not measured efficacy',
    source_manifest_sha=sha(out/'V23_EXPERIMENT_LOCK.json'))
  atomic_json(out/f'V23_{suffix}_INTERPRETATION_GATE.json',result)
  print('V23_SUMMARY='+json.dumps(result,ensure_ascii=False),flush=True)
  if not partial:make_figure(out,effect_rows)
  return result
 finally:db.close()

def make_figure(out,rows):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':23,'axes.titlesize':24,'axes.labelsize':23,
   'xtick.labelsize':21,'ytick.labelsize':21,'legend.fontsize':18,
   'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
   'pdf.fonttype':42,'ps.fonttype':42})
 # A: selected axis mechanisms (only model-based contrasts)
 fig,axs=plt.subplots(2,2,figsize=(26,17),constrained_layout=True)
 def find(mode,day,endpoint,term):
  a=[r for r in rows if r['release_mode']==mode and r['horizon_days']==day and r['endpoint']==endpoint and r['term']==term]
  return a[0] if len(a)==1 else None
 axes=[('prpc_function','P1'),('tbk1_activity','TBKI'),('srsf1','MIR4651'),
       ('progerin_total','MIR4651')]
 x=np.arange(4)
 vals=[find('independent',730.5,a,b) for a,b in axes]
 axs[0,0].bar(range(4),[v['design_median'] if v else np.nan for v in vals],width=.55)
 axs[0,0].set_xticks(range(4),['PrP$^C$','TBK1','SRSF1','Progerin'],rotation=15)
 axs[0,0].set_ylabel('Factor-on minus factor-off')
 axs[0,0].set_title('a  Model-specific target responses',loc='left')
 modes=('independent','sequential_p1_tbki_mir','sequential_mir_tbki_p1')
 for i,endpoint in enumerate(('ec_activation','vsmc_dysfunction','ecm_stiffness')):
  xx=np.arange(3)+(i-1)*.22
  vals=[find(m,730.5,endpoint,'CONJ_111_MINUS_LONA') for m in modes]
  axs[0,1].bar(xx,[v['design_median'] if v else np.nan for v in vals],width=.22,label=endpoint)
 axs[0,1].set_xticks(range(3),['Independent','P1 first','miR first'])
 axs[0,1].set_ylabel('Conjugate minus lonafarnib')
 axs[0,1].set_title('b  Endpoint-dependent comparison',loc='left')
 axs[0,1].legend(frameon=False,fontsize=16)
 axs[0,1].axhline(0,c='gray',lw=1)
 three=['P1xTBKI','P1xMIR4651','TBKIxMIR4651','THREE_WAY']
 vals=[find('independent',730.5,'ec_activation',t) for t in three]
 axs[1,0].bar(range(4),[v['design_median'] if v else np.nan for v in vals],width=.58)
 axs[1,0].set_xticks(range(4),['P1 × TBK1','P1 × miR','TBK1 × miR','Three-way'],rotation=16)
 axs[1,0].set_ylabel('Factorial interaction (model units)')
 axs[1,0].set_title('c  Nonadditive endothelial responses',loc='left')
 axs[1,0].axhline(0,c='gray',lw=1)
 for endpoint in ('ec_activation','vsmc_dysfunction','progerin_total'):
  yy=[find('independent',d,endpoint,'CONJ_111_MINUS_LONA') for d in HORIZONS]
  axs[1,1].plot([d/30.4375 for d in HORIZONS],
    [v['design_median'] if v else np.nan for v in yy],marker='o',label=endpoint)
 axs[1,1].axhline(0,c='gray',lw=1)
 axs[1,1].set_xticks([6,12,24]);axs[1,1].set_xlabel('Months (modeled)')
 axs[1,1].set_ylabel('Conjugate minus lonafarnib')
 axs[1,1].set_title('d  Time-dependent modeled contrasts',loc='left')
 axs[1,1].legend(frameon=False,fontsize=16)
 for ax in axs.flat:
  ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
 fig.savefig(out/'V23_MAIN_FACTORIAL_NATURE_STYLE_23PT.png',dpi=300,bbox_inches='tight')
 plt.close(fig)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--output',required=True);p.add_argument('--partial',action='store_true')
 p.add_argument('--include-0d',action='store_true')
 a=p.parse_args()
 try:summarize(a.output,include_0d=a.include_0d,partial=a.partial)
 except Exception as e:
  print('V23_SUMMARY_HOLD='+str(type(e).__name__)+': '+str(e),flush=True);raise
