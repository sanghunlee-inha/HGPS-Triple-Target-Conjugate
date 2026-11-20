"""QMC replication, ablation, 0D sensitivity, true paired 10k null audit.

This characterizes a PREDEFINED synthetic input cube. It MUST NOT be used as
patient outcome probability, biological confidence interval, or causal effect.
"""
from __future__ import annotations
import argparse, csv, json, math, sqlite3, hashlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from v22_core import SEEDS, MODES, connect, unpack, load_json, write_atomic, log, Hold
from v22_vascular import paired_remodel, InvalidVascularInput

OUTCOMES=('progerin_total','farnesyl_progerin','ec_activation','vsmc_dysfunction','ecm_stiffness',
          'tox_neuro','tox_cardiac','tox_hepatic','tox_renal','tox_platelet','tox_immune','tox_mitophagy')
HORIZONS=(182.625,365.25,730.5)

def write_csv(path,rows):
 if not rows:return
 with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def signflip_null(values,iterations=10000,seed=437111):
 """Real paired sign flip randomization. Design-symmetry diagnostic ONLY.

 For quasi-random deterministic Sobol sequences this does not produce clinical
 p values or frequentist random sampling interpretation; never label as such.
 """
 a=np.asarray(values,dtype=float)
 if a.ndim!=1 or len(a)<2 or not np.isfinite(a).all():raise Hold('HOLD_INVALID_SIGN_FLIP_INPUT')
 if iterations<10000:raise Hold('HOLD_REQUIRE_10000_TRUE_SIGN_FLIPS')
 rng=np.random.default_rng(seed);observed=abs(float(a.mean()))
 exceeded=0;null_sum=0.;null_sq=0.;total=0
 # bounded memory (64 permutations × n cases), true scenario-level signs
 for st in range(0,iterations,64):
  n=min(64,iterations-st)
  # integers random Bernoulli, genuine independent paired labels per row
  flips=rng.integers(0,2,size=(n,len(a)),dtype=np.int8)
  signs=flips.astype(np.float32)*2.-1.
  v=(signs@a)/len(a)
  exceeded+=int(np.count_nonzero(np.abs(v)>=observed))
  null_sum+=float(v.sum());null_sq+=float((v*v).sum());total+=n
 return dict(randomization_count=total,design_only_extreme_fraction=(exceeded+1)/(total+1),
             observed_mean=round(float(a.mean()),12),null_mean=null_sum/total,
             null_sd=math.sqrt(max(0.,null_sq/total-(null_sum/total)**2)),
             null_is_clinical_p_value=False)

def analyze(output,*,expected_per_seed=None,null_iterations=10000):
 out=Path(output).resolve();gate=load_json(out/'V22_BDF_FINAL_GATE.json')
 if gate.get('status')!='PASS_5_INDEPENDENT_QMC_DESIGNS_BDF_NUMERICS_NOT_CLINICAL':
  raise Hold('HOLD_ANALYSIS_REQUIRES_FULL_24M_BDF_GATE')
 nper=int(gate['design_sizes'][0])
 if expected_per_seed is not None and nper!=expected_per_seed:raise Hold('HOLD_SEED_COUNT_MISMATCH')
 if len(set(gate['design_sizes']))!=1:raise Hold('HOLD_UNBALANCED_DESIGN')
 acc=defaultdict(list);count_classes=Counter();fail=0;nfev=0;ablations=[];refinements=[]
 hashset=set();vrows=[]
 for seed in SEEDS:
  db=connect(out/f'V22_SEED_{seed}.sqlite')
  if int(db.execute('SELECT COUNT(*) FROM scenario').fetchone()[0])!=nper:raise Hold('HOLD_INCOMPLETE_DESIGN')
  for idx,sha,blob,cost in db.execute('SELECT idx,design_sha,blob,nfev FROM scenario ORDER BY idx'):
   if sha in hashset:raise Hold('HOLD_DUPLICATE_ACROSS_QMC_REPLICATES')
   hashset.add(sha);nfev+=int(cost)
   results=unpack(blob)
   if {x['mode'] for x in results}!=set(MODES):raise Hold('HOLD_MISSING_RELEASE')
   for rec in results:
    mode=rec['mode']
    if rec.get('knockout'):ablations.append(rec['knockout'])
    if rec.get('refinement'):refinements.append(rec['refinement'])
    if len(rec['rows'])!=3:raise Hold('HOLD_THREE_HORIZON_ROWS_REQUIRED')
    for row in rec['rows']:
     h=float(row['horizon_days'])
     if h not in HORIZONS:raise Hold('HOLD_INVALID_HORIZON')
     for outcome in OUTCOMES:
      value=float(row['delta_'+outcome]);
      if not math.isfinite(value):raise Hold('HOLD_NONFINITE_BDF_OUTPUT')
      acc[(seed,mode,h,outcome)].append(value)
     count_classes[(mode,h,row['classification_screening_only'])]+=1
     if max(float(row[x]) for x in ('max_lona_mass_relative','max_conj_mass_relative','max_conj_payload_relative'))>2e-5:
      fail+=1
    # For prespecified 1/64 of all cases, retain V21 monthly 29-state trajectory
    # and integrate genuine (but uncalibrated) 0D remodeling ODE.
    if rec['timeline'] and mode=='independent':
     for rate_label,coeff in [('low',dict(k_ec=0.0001,k_vsmc=0.0001,k_ecm=0.0001)),
                              ('high',dict(k_ec=0.001,k_vsmc=0.001,k_ecm=0.001))]:
      # V15 model.y0 initializes all EC/VSMC/ECM states to exactly 1.
      # The recorded V21 timeline starts AFTER day zero, hence restore this
      # verified model baseline explicitly; it is NOT a patient measurement.
      first={'horizon_days':0.0}
      for arm in ('lona','conj','untreated'):
       for k in ('ec_activation','vsmc_dysfunction','ecm_stiffness'):
        first[arm+'_'+k]=1.0
      try:
       ans=paired_remodel([first,*rec['timeline']],**coeff)
       vrows.append(dict(seed=seed,idx=idx,release_mode=mode,hypothetical_0D_rate=rate_label,
        end_R_conjugate=float(ans['conj']['resistance_relative_fixed_flow'][-1]),
        end_R_lonafarnib=float(ans['lona']['resistance_relative_fixed_flow'][-1]),
        delta_R=float(ans['conj']['resistance_relative_fixed_flow'][-1]-ans['lona']['resistance_relative_fixed_flow'][-1]),
        zero_D_valid=True,zero_D_hold_reason='',valid_CFD=False,valid_patient_IMT=False))
      except InvalidVascularInput as exc:
       # An uncalibrated 0D radius failure cannot be "fixed" by clipping.
       # Preserve the failed hypothesis explicitly while retaining valid BDF.
       vrows.append(dict(seed=seed,idx=idx,release_mode=mode,hypothetical_0D_rate=rate_label,
        end_R_conjugate='',end_R_lonafarnib='',delta_R='',zero_D_valid=False,
        zero_D_hold_reason=str(exc),valid_CFD=False,valid_patient_IMT=False))
  db.close()
 if fail:raise Hold('HOLD_MASS_LEDGER_FAILED')
 if refinements and not all(x['pass_threshold_1e_minus_3'] for x in refinements):raise Hold('HOLD_REFINEMENT_FAILED')
 summary=[]
 for mode in MODES:
  for h in HORIZONS:
   for outcome in OUTCOMES:
    reps=[np.asarray(acc[(seed,mode,h,outcome)],dtype=float) for seed in SEEDS]
    if any(len(x)!=nper for x in reps):raise Hold('HOLD_INCOMPLETE_SEED_OUTCOME')
    mean=np.array([x.mean() for x in reps])
    # Independent randomized QMC seeds permit *numerical integration* error
    # estimates, not biological uncertainty or confidence intervals.
    sd=float(np.std(mean,ddof=1));mcse=sd/math.sqrt(5)
    merged=np.concatenate(reps)
    summary.append(dict(release_mode=mode,horizon_days=h,outcome=outcome,
       design_mean=float(mean.mean()),design_median=float(np.median(merged)),
       min_stress=float(merged.min()),max_stress=float(merged.max()),
       independent_scrambled_sobol_seed_mean_sd=sd,
       qmc_integration_standard_error=mcse,
       n_independent_scrambles=5,n_synthetic_scenarios=len(merged),
       clinical_CI=False,empirical_probability=False))
 write_csv(out/'V22_DESIGN_UNCERTAINTY_NOT_POPULATION.csv',summary)
 write_csv(out/'V22_0D_REMODELING_NOT_CFD.csv',vrows)
 classifications=[dict(release_mode=k[0],horizon_days=k[1],scenario_class=k[2],
                     n_synthetic_design_cases=v,is_patient_probability=False) for k,v in sorted(count_classes.items())]
 write_csv(out/'V22_MATCHED_CLASSIFICATION_COUNTS.csv',classifications)
 write_csv(out/'V22_ABLATION_PANELS.csv',ablations)
 write_csv(out/'V22_SOLVER_REFINEMENT.csv',refinements)
 nulls=[]
 for mode in MODES:
  primary=np.concatenate([np.asarray(acc[(seed,mode,730.5,'progerin_total')]) for seed in SEEDS])
  result=signflip_null(primary,iterations=null_iterations,seed=22000+MODES.index(mode))
  nulls.append(dict(release_mode=mode,outcome='progerin_total',horizon_days=730.5,**result))
 write_csv(out/'V22_10000_SIGN_FLIP_DESIGN_NULL_NOT_CLINICAL.csv',nulls)
 final=dict(status='PASS_CONDITIONAL_MULTISEED_QMC_AND_NUMERIC_AUDIT_NOT_CLINICAL',
  n_distinct_stress_biologies=5*nper,n_paired_drug_comparisons=15*nper,
  n_scrambled_sobol_designs=5,n_releases=3,n_horizons=3,
  n_real_BDF_rhs_evaluations=nfev,n_ablation_panels=len(ablations),
  n_refinement_panels=len(refinements),n_vascular_0D_case_rate_comparisons=len(vrows),
  n_true_sign_flip_randomizations_per_release=null_iterations,
  patient_effect_estimated=False,clinical_trial_probability_estimated=False,
  three_dimensional_CFD_performed=False,validated_PWV_cIMT_ABI=False,
  chemistry_and_toxicity_validated=False,posterior_probabilities_validated=False)
 write_atomic(out/'V22_ANALYSIS_FINAL_GATE.json',final)
 log('ANALYSIS_FINAL',**final)
 return final

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True)
 p.add_argument('--cases-per-seed',type=int,default=None)
 p.add_argument('--null-iterations',type=int,default=10000)
 a=p.parse_args();analyze(a.output,expected_per_seed=a.cases_per_seed,null_iterations=a.null_iterations)
