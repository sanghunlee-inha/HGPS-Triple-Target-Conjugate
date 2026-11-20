"""Mechanistic QSP calibration from independently measured time courses.
Requires source-anchored baseline parameter file; never uses synthetic model outputs as observations.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from qsp_engine import ModelHold, STATES, PARAMETERS, check_parameters, simulate

REQUIRED=['study_id','donor_id','condition_id','time_h','dose_nM',
          'tau_h','n_doses','analyte','value','sigma','evidence_type','source_reference']

def calibrate(input_csv,parameter_json,out,seed=20261115,starts=12):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dat=pd.read_csv(input_csv,dtype={'donor_id':str,'condition_id':str,'study_id':str})
    miss=set(REQUIRED)-set(dat)
    if miss:raise ModelHold('CALIBRATION_MISSING_COLUMNS: '+str(sorted(miss)))
    if not (dat.evidence_type=='measured').all():raise ModelHold('MODEL_OUTPUT_CANNOT_VALIDATE_ITSELF')
    if dat.source_reference.isna().any() or dat.source_reference.astype(str).str.strip().eq('').any():
        raise ModelHold('CALIBRATION_MISSING_EXPERIMENTAL_PROVENANCE')
    if not set(dat.analyte).issubset(STATES):raise ModelHold('UNKNOWN_CALIBRATION_ANALYTES')
    for k in ['time_h','dose_nM','tau_h','n_doses','value','sigma']:
        dat[k]=pd.to_numeric(dat[k],errors='raise')
        if not np.isfinite(dat[k]).all(): raise ModelHold('NONFINITE_CALIBRATION:'+k)
    if (dat.sigma<=0).any() or (dat.time_h<0).any() or (dat.dose_nM<0).any() or (dat.tau_h<=0).any():
        raise ModelHold('INVALID_CALIBRATION_UNITS')
    if ((dat.n_doses<1) | (dat.n_doses%1!=0)).any():raise ModelHold('INVALID_CALIBRATION_N_DOSES')
    if dat.donor_id.nunique()<3:raise ModelHold('CALIBRATION_REQUIRES_3_BIOLOGICAL_DONORS_FOR_HOLDOUT')
    param=json.loads(Path(parameter_json).read_text(encoding='utf-8-sig'))
    p={k:x.get('value') for k,x in param['parameters'].items() if isinstance(x,dict)}
    p=check_parameters(p)
    fit_keys=param.get('fit_parameters',[])
    if not 1<=len(fit_keys)<=6 or len(set(fit_keys))!=len(fit_keys):
        raise ModelHold('FIT_PARAM_COUNT_MUST_BE_1_TO_6')
    if any(k not in PARAMETERS for k in fit_keys): raise ModelHold('UNKNOWN_FIT_PARAM')
    bounds=[]
    for k in fit_keys:
        m=param['parameters'][k]
        if not m.get('source_reference') or not m.get('evidence_type'):
            raise ModelHold('NO_PARAMETER_EVIDENCE:'+k)
        lo,hi=m.get('lower'),m.get('upper')
        if lo is None or hi is None or not 0<float(lo)<float(hi):
            raise ModelHold('BOUNDS_MISSING:'+k)
        bounds.append((float(lo),float(hi)))
    donors=sorted(dat.donor_id.unique())
    rng=np.random.default_rng(seed)
    held_out=donors[int(rng.integers(0,len(donors)))]
    train=dat[dat.donor_id!=held_out].copy()
    test=dat[dat.donor_id==held_out].copy()
    if not len(test) or not len(train):raise ModelHold('EMPTY_SPLIT')
    if len(train)<4*len(fit_keys):raise ModelHold('INSUFFICIENT_INDEPENDENT_OBSERVATIONS_FOR_FIT')
    # Compartment/endpoint data are never pooled without matching nominal schedules.
    context=['study_id','donor_id','condition_id']
    for _,g in dat.groupby(context):
        for col in ['dose_nM','tau_h','n_doses']:
            if g[col].nunique()!=1:raise ModelHold('INCONSISTENT_SCHEDULE_WITHIN_CONDITION')
    if dat['time_h'].max()<=0:raise ModelHold('NO_TIMECOURSE')
    groups_train=list(train.groupby(context,sort=False))
    groups_test=list(test.groupby(context,sort=False))
    def prediction_rows(p0,groups):
        yall=[]
        for _,g in groups:
            dose=float(g.dose_nM.iloc[0]); tau=float(g.tau_h.iloc[0]);n=int(g.n_doses.iloc[0]);
            times=g.time_h.to_numpy(float)
            max_t=max(float(times.max()),1e-4)
            # explicit t_eval as raw times not interpolation; simulate uses fine grid for plotting,
            # and interpolate with dense grid. High resolution required for meaningful fit.
            gr_t,y=simulate(p0,dose_nM=dose,tau_h=tau,n_doses=n,horizon_h=max_t,
                            n_times=max(301,int(20*max_t/tau)))
            for analyte,t in zip(g.analyte.to_numpy(),times):
                yall.append(float(np.interp(t,gr_t,y[:,STATES.index(analyte)])))
        return np.asarray(yall)
    obs_train=np.concatenate([g.value.to_numpy(float) for _,g in groups_train])
    sigma_train=np.concatenate([g.sigma.to_numpy(float) for _,g in groups_train])
    lo=np.log([x[0] for x in bounds]);hi=np.log([x[1] for x in bounds])
    x0=np.clip(np.log([p[k] for k in fit_keys]),lo+1e-7,hi-1e-7)
    def setpars(x):
        q=p.copy()
        for k,v in zip(fit_keys,x):q[k]=float(np.exp(v))
        return q
    def resid(x):
        return (prediction_rows(setpars(x),groups_train)-obs_train)/sigma_train
    trials=[]
    for i in range(starts):
        init=x0 if i==0 else rng.uniform(lo+1e-6,hi-1e-6)
        res=least_squares(resid,init,bounds=(lo,hi),method='trf',
            x_scale='jac',max_nfev=300,verbose=0)
        trials.append(res)
    best=min(trials,key=lambda r:np.sum(r.fun**2))
    est=setpars(best.x)
    train_pred=prediction_rows(est,groups_train)
    val_pred=prediction_rows(est,groups_test)
    obs_test=np.concatenate([g.value.to_numpy(float) for _,g in groups_test])
    sig_test=np.concatenate([g.sigma.to_numpy(float) for _,g in groups_test])
    s=np.linalg.svd(best.jac,compute_uv=False)
    rank=int((s>max(float(s[0]),1e-14)*1e-7).sum()) if len(s) else 0
    def metrics(obs,pred,sigma):
        return {'n_points':len(obs),'rmse':float(np.sqrt(np.mean((pred-obs)**2))),
           'weighted_rmse':float(np.sqrt(np.mean(((pred-obs)/sigma)**2)))}
    report={'gate':'CALIBRATED_IN_VITRO_IF_MEASURED_INPUT_GENUINE_NOT_PATIENT_VALIDATED',
       'fit_parameters':fit_keys,'fit_values':{k:est[k] for k in fit_keys},
       'held_out_donor':held_out,'n_train_donors':int(train.donor_id.nunique()),
       'n_test_donors':int(test.donor_id.nunique()),
       'train':metrics(obs_train,train_pred,sigma_train),
       'test':metrics(obs_test,val_pred,sig_test),
       'jacobian_singular_values':s.tolist(),'jacobian_rank':rank,
       'rank_gate':'PASS_LOCAL_SENSITIVITY' if rank==len(fit_keys) else 'HOLD_PRACTICAL_IDENTIFIABILITY',
       'conditional_on_fixed_parameters':list(sorted(set(PARAMETERS)-set(fit_keys))),
       'notes':['Predictions interpolate within a dense time grid; tighten grid/solver tolerance for final report.',
                'Donor holdout is only independent if donors and sources are truly independent.',
                'No transfer to human dose or vascular CFD without bridging validation.']}
    (out/'40_QSP_CALIBRATION_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    from hashlib import sha256
    h=sha256(Path(input_csv).read_bytes()).hexdigest()
    calibrated=json.loads(json.dumps(param))
    for k in fit_keys:
        calibrated['parameters'][k]['value']=est[k]
        calibrated['parameters'][k]['evidence_type']='fit_to_measured_timecourse_conditional'
        calibrated['parameters'][k]['source_reference']='sha256:'+h
    calibrated['calibration_provenance']={'data_sha256':h,'held_out_donor':held_out,
         'fit_parameter_count':len(fit_keys),'jacobian_rank':rank,
         'validation':'independent_donor_holdout_not_clinically_validated'}
    (out/'42_QSP_CALIBRATED_PARAMETERS.json').write_text(json.dumps(calibrated,indent=2),encoding='utf-8')
    pd.DataFrame([{'parameter':k,'value':est[k], 'lower':param['parameters'][k].get('lower'),
                   'upper':param['parameters'][k].get('upper')} for k in fit_keys]).to_csv(out/'41_QSP_ESTIMATED_PARAMETERS.csv',index=False)
    print('QSP_MEASURED_CALIBRATION_COMPLETE', 'train_wRMSE',round(report['train']['weighted_rmse'],5),
          'test_wRMSE',round(report['test']['weighted_rmse'],5),'rank',rank,'/',len(fit_keys))
    return report
