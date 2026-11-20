"""Real mechanistic PK/PD ODE integration (BDF + Radau verification).

All kinetic constants and every dose must be provided with laboratory/source
provenance. No author kinetics are fabricated. Output is in-silico hypothesis;
not verified patient response unless independently calibrated and validated.

Single administered PRNP expression inhibitor–TBK1 inhibitor–miR-4651 conjugate.
The three payload activities result from cellular release of ONE conjugate.
"""
from __future__ import annotations
import csv, json, hashlib, time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from core import jd, digest, emit

STATES = ('conjugate_plasma_nmol','conjugate_peripheral_nmol',
          'conjugate_internalized_nmol','P1_in_nmol','TBK1_in_nmol',
          'miR_endosomal_nmol','miR_AGO2_nmol',
          'PRNP_rel','TBK1_activity_rel','SRSF1_rel','progerin_rel','injury_proxy_rel')
RATE_UNITS = 'day^-1'
K = (
 'k_clear_plasma','k_plasma_to_peripheral','k_peripheral_to_plasma',
 'k_cell_uptake','k_cell_payload_release','k_vesicle_loss',
 'fraction_P1','fraction_TBK1','fraction_miR',
 'k_clear_P1','k_clear_TBK1','k_load_AGO2',
 'k_clear_endosomal_miR','k_clear_AGO2',
 'PRNP_Emax','PRNP_EC50_nmol','PRNP_hill',
 'TBK1_Emax','TBK1_EC50_nmol','TBK1_hill',
 'SRSF1_Emax','SRSF1_EC50_nmol','SRSF1_hill',
 'k_turnover_PRNP','k_turnover_TBK1','k_turnover_SRSF1','k_turnover_progerin',
 'progerin_srsf1_hill','progerin_srsf1_half',
 'k_turnover_injury','injury_weight_progerin','injury_weight_PRNP',
 'injury_weight_TBK1')
FRACS = ('fraction_P1','fraction_TBK1','fraction_miR')
WEIGHTS = ('injury_weight_progerin','injury_weight_PRNP','injury_weight_TBK1')
EMAX=('PRNP_Emax','TBK1_Emax','SRSF1_Emax')
POSITIVE = set(K)-set(FRACS)-set(WEIGHTS)-set(EMAX)

def parameter_table(path):
    """CSV: parameter,value,units,reference,method. Explicit evidence on EVERY row."""
    params={}; evidence={}
    with open(path,encoding='utf-8-sig',newline='') as f:
        rows=csv.DictReader(f)
        if not {'parameter','value','units','reference','method'}.issubset(rows.fieldnames or []):
            raise ValueError('PKPD_PARAMETER_TABLE_REQUIRES_PARAMETER_VALUE_UNITS_REFERENCE_METHOD')
        for row in rows:
            name=row['parameter'].strip()
            if name not in K:raise ValueError('PKPD_UNKNOWN_PARAMETER_'+name)
            if name in params:raise ValueError('PKPD_DUPLICATE_PARAMETER_'+name)
            if not row['value'].strip():raise ValueError('PKPD_UNCALIBRATED_PARAMETER_'+name)
            v=float(row['value'])
            if not np.isfinite(v):raise ValueError('PKPD_NONFINITE_PARAMETER_'+name)
            if not row['units'].strip() or not row['reference'].strip() or not row['method'].strip():
                raise ValueError('PKPD_MISSING_SOURCE_PROVENANCE_'+name)
            params[name]=v;evidence[name]={k:row[k] for k in ('units','reference','method')}
    missing=set(K)-set(params)
    if missing:raise ValueError('PKPD_REQUIRED_KINETIC_PARAMETERS_MISSING: '+','.join(sorted(missing)))
    for k in POSITIVE:
        if params[k]<=0:raise ValueError('PKPD_RATE_OR_SCALE_NOT_POSITIVE_'+k)
    for k in set(FRACS)|set(WEIGHTS)|set(EMAX):
        if not 0<=params[k]<=1:raise ValueError('PKPD_FRACTION_OR_EMAX_OUTSIDE_0_1_'+k)
    if not np.isclose(sum(params[k] for k in FRACS),1,rtol=0,atol=1e-8):
        raise ValueError('PKPD_PAYLOAD_STOICHIOMETRY_FRACTIONS_MUST_SUM_ONE')
    if not np.isclose(sum(params[k] for k in WEIGHTS),1,rtol=0,atol=1e-8):
        raise ValueError('PKPD_INJURY_WEIGHTS_MUST_SUM_ONE')
    return params,evidence

def hilleffect(c,ec50,hill,emax):
    c=max(0.,float(c))
    if c==0:return 0.
    # stable calculation to prevent overflow
    return emax/(1+(ec50/c)**hill) if ec50/c<1e6 else 0.

def rhs(t,y,p):
    """12 coupled differential equations, t in days; PK amounts in nmol."""
    cp,per,ves,p1,tbi,endo,ago,prnp,tbk,srs,prog,inj=y
    k=p
    release=k['k_cell_payload_release']*ves
    Epr=hilleffect(p1,k['PRNP_EC50_nmol'],k['PRNP_hill'],k['PRNP_Emax'])
    Etb=hilleffect(tbi,k['TBK1_EC50_nmol'],k['TBK1_hill'],k['TBK1_Emax'])
    Esr=hilleffect(ago,k['SRSF1_EC50_nmol'],k['SRSF1_hill'],k['SRSF1_Emax'])
    srspos=max(0.,srs)
    srs_h=srspos**k['progerin_srsf1_hill']
    h0=k['progerin_srsf1_half']**k['progerin_srsf1_hill']
    pbase=1./(1+h0)
    prog_target=(srs_h/(h0+srs_h))/pbase if (h0+srs_h)>0 else 0.
    targetinj=k['injury_weight_progerin']*max(0.,prog)+k['injury_weight_PRNP']*max(0.,prnp)+k['injury_weight_TBK1']*max(0.,tbk)
    return np.array([
      -(k['k_clear_plasma']+k['k_plasma_to_peripheral']+k['k_cell_uptake'])*cp+k['k_peripheral_to_plasma']*per,
      k['k_plasma_to_peripheral']*cp-k['k_peripheral_to_plasma']*per,
      k['k_cell_uptake']*cp-release-k['k_vesicle_loss']*ves,
      k['fraction_P1']*release-k['k_clear_P1']*p1,
      k['fraction_TBK1']*release-k['k_clear_TBK1']*tbi,
      k['fraction_miR']*release-(k['k_load_AGO2']+k['k_clear_endosomal_miR'])*endo,
      k['k_load_AGO2']*endo-k['k_clear_AGO2']*ago,
      k['k_turnover_PRNP']*(1-Epr-prnp),
      k['k_turnover_TBK1']*(1-Etb-tbk),
      k['k_turnover_SRSF1']*(1-Esr-srs),
      k['k_turnover_progerin']*(prog_target-prog),
      k['k_turnover_injury']*(targetinj-inj)
    ],float)

def dose_table(path):
    with open(path,encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows or not {'day','dose_nmol','source_reference'}.issubset(rows[0]):
        raise ValueError('PKPD_DOSE_TABLE_REQUIRES_DAY_DOSE_NMOL_SOURCE_REFERENCE')
    doses=[]
    for row in rows:
        t=float(row['day']);amt=float(row['dose_nmol'])
        if not np.isfinite(t+amt) or t<0 or amt<=0 or not row['source_reference'].strip():
            raise ValueError('PKPD_INVALID_DOSE_TIME_AMOUNT_OR_SOURCE')
        doses.append((t,amt))
    if any(doses[i][0]>=doses[i+1][0] for i in range(len(doses)-1)):
        raise ValueError('PKPD_DOSES_MUST_BE_STRICTLY_INCREASING')
    return doses

def integrate(p,doses,end_day,rtol,atol,maxstep,eval_step=0.1,method='BDF'):
    if end_day<=doses[-1][0]:raise ValueError('PKPD_END_DAY_BEFORE_LAST_DOSE')
    y=np.array([0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,1.,1.])
    t=0.0; ts=[t]; ys=[y.copy()]; stats=[]
    for n, (next_time, dose) in enumerate(doses+[(float(end_day),0.)]):
        if next_time<t:raise ValueError('NONMONOTONIC_TIME')
        if next_time>t:
            t_eval=np.linspace(t,next_time,max(2,int(np.ceil((next_time-t)/eval_step))+1))
            sol=solve_ivp(rhs,(t,next_time),y,args=(p,),method=method,rtol=rtol,atol=atol,
                          max_step=maxstep,t_eval=t_eval)
            if not sol.success:raise RuntimeError(f'PKPD_{method}_FAILED_SEGMENT_{n}: {sol.message}')
            if np.any(~np.isfinite(sol.y)):raise RuntimeError('PKPD_NONFINITE_STATES')
            if np.min(sol.y)<-max(1e-8,atol*100):raise RuntimeError('PKPD_NEGATIVE_STATE')
            ts.extend(sol.t[1:].tolist());ys.extend(sol.y.T[1:].copy())
            y=sol.y[:,-1].copy();t=next_time
            stats.append({'from_day':float(sol.t[0]),'to_day':float(sol.t[-1]),'nfev':sol.nfev,'njev':sol.njev,'nlu':sol.nlu})
        if dose>0:
            y[0]+=dose
            ts.append(t);ys.append(y.copy())  # mark instantaneous IV bolus
    a=np.array(ys)
    if np.any(a[:,7:]>5):raise RuntimeError('PKPD_UNEXPECTED_NORMALIZED_BIOLOGY_ABOVE_FIVE')
    return np.array(ts),a,stats

def run(config_path,out):
    import pandas as pd
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    if cfg.get('geometry_scope')!='ADULT_TOPCOW_REFERENCE_NOT_PATIENT_SPECIFIC':
        raise ValueError('PKPD_MUST_ACKNOWLEDGE_REFERENCE_GEOMETRY')
    pfile=Path(cfg['parameter_table']);dfile=Path(cfg['dose_table'])
    if not pfile.is_file() or not dfile.is_file():raise ValueError('PKPD_PARAMETER_OR_DOSE_FILE_MISSING')
    p,ev=parameter_table(pfile);doses=dose_table(dfile)
    rtol=float(cfg.get('rtol',1e-8));atol=float(cfg.get('atol',1e-10));mstep=float(cfg.get('max_step_days',.25))
    if not (0<rtol<=1e-5 and 0<atol<=1e-7 and 0<mstep<=1.0):
        raise ValueError('PKPD_INTEGRATOR_TOLERANCE_NOT_RESEARCH_GRADE')
    end=float(cfg['end_day']);density=float(cfg.get('dense_step_days',.1))
    emit('PKPD_BDF_REINTEGRATION_STARTED',len(doses),'doses',end,'days')
    t,y,stats=integrate(p,doses,end,rtol,atol,mstep,density,'BDF')
    emit('PKPD_RADAU_INDEPENDENT_CROSSCHECK_STARTED')
    tt,yy,st2=integrate(p,doses,end,rtol/5,atol/5,mstep/2,density,'Radau')
    # times are identical because t_eval grids were matched
    if not np.allclose(t,tt,rtol=0,atol=1e-10):raise ValueError('PKPD_CROSSCHECK_TIME_GRID_INCONSISTENT')
    scale=np.maximum(np.maximum(np.abs(y),np.abs(yy)),np.array([1.]*len(STATES)))
    rel=float(np.max(np.abs(y-yy)/scale))
    if rel>float(cfg.get('max_cross_solver_relative_difference',1e-4)):
        raise RuntimeError('PKPD_BDF_RADAU_CROSSCHECK_FAILED_'+str(rel))
    df=pd.DataFrame(y,columns=STATES);df.insert(0,'day',t)
    df.to_csv(out/'FIG5_NEW_MECHANISTIC_ODE_TIMESERIES.csv',index=False,encoding='utf-8-sig')
    # Independent mass bookkeeping for PK parent and released payloads is possible only
    # after including clearance integrals. Check instantaneous bolus at each dose.
    initial={'PRNP_rel':float(y[0,7]),'TBK1_activity_rel':float(y[0,8]),'SRSF1_rel':float(y[0,9]),
             'progerin_rel':float(y[0,10]),'injury_proxy_rel':float(y[0,11])}
    endpt={k:float(y[-1,i]) for i,k in enumerate(STATES)}
    meta={'status':'REAL_MECHANISTIC_BDF_RADAU_INTEGRATION_NO_CLINICAL_VALIDATION',
        'model':'one_covalent_conjugate_12_state_explicit_ODE_parameterized',
        'scope':'MODEL_HYPOTHESIS_ONLY_UNLESS_INDEPENDENT_CALIBRATION_AND_EXTERNAL_VALIDATION',
        'input_sha256':{str(pfile):digest(pfile),str(dfile):digest(dfile),str(config_path):digest(config_path)},
        'initial_pd':initial,'terminal':endpt,'n_doses':len(doses),'time_days':end,
        'n_ode_states':len(STATES),'bdf_segments':stats,'radau_segments':st2,
        'max_scaled_BDF_Radau_difference':rel,
        'parameter_provenance':ev,
        'cannot_claim':['matches V22 original RHS without explicit equation-level audit',
                        'experimentally proven treatment','specific patient PK or drug-induced CFD change']}
    jd(out/'FIG5_MECHANISTIC_NUMERICAL_REPORT.json',meta)
    emit('PKPD_BDF_RADAU_CROSSCHECK_PASS',rel)
    return meta
