"""HGPS P1/TBK1_014/miR-4651 tri-conjugate conditional ODE model.
No efficacy or human dose parameters are asserted by this implementation.
See README_KR.md for evidence gates and model assumptions.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc
from concurrent.futures import ProcessPoolExecutor

PARAMETERS = [
 'k_clear_c','k_exchange','k_return','k_clear_t','k_uptake',
 'k_clear_ci','k_release','k_clear_p','k_clear_tbk','k_clear_mir',
 'ec50_p','ec50_tbk','ec50_mir','k_prnp','k_prpc','k_tbk1',
 'k_srsf1','k_progerin','k_inflammation','k_senescence',
 'weight_prpc','weight_tbk1','weight_progerin','weight_inflammation'
]
STATES = [
 'conjugate_plasma_nM','conjugate_tissue_nM','conjugate_intracellular_nM',
 'free_P1_nM','free_TBK1_014_nM','active_miR4651_nM',
 'PRNP_mRNA_rel','PrPC_protein_rel','TBK1_activity_rel',
 'SRSF1_rel','progerin_rel','inflammation_rel','senescence_rel'
]
MODS=('P1','TBK1_014','miR4651')
MASKS=[tuple(m for i,m in enumerate(MODS) if mask & (1<<i)) for mask in range(8)]

class ModelHold(ValueError): pass

def check_parameters(p):
    absent=[x for x in PARAMETERS if x not in p or p[x] is None]
    if absent: raise ModelHold('MISSING_PARAMETERS: '+','.join(absent))
    a={k:float(p[k]) for k in PARAMETERS}
    if not all(np.isfinite(v) for v in a.values()): raise ModelHold('NONFINITE_PARAMETERS')
    if any(a[k]<=0 for k in PARAMETERS if k.startswith('k_') or k.startswith('ec50_')):
        raise ModelHold('NONPOSITIVE_K_OR_EC50')
    if any(a[k]<0 for k in PARAMETERS if k.startswith('weight_')):
        raise ModelHold('NEGATIVE_WEIGHT')
    for ks in (('weight_prpc','weight_tbk1'), ('weight_progerin','weight_inflammation')):
        if not np.isclose(sum(a[k] for k in ks),1.0,atol=1e-6):
            raise ModelHold('WEIGHTS_NOT_SUM_ONE: '+str(ks))
    return a

def rhs(t,y,p,enabled):
    Cp,Ct,Ci,P,T,M,R,PC,TB,S,PRO,INF,SEN=y
    E_p = (P/(p['ec50_p']+max(0.,P))) if 'P1' in enabled else 0.
    E_t = (T/(p['ec50_tbk']+max(0.,T))) if 'TBK1_014' in enabled else 0.
    E_m = (M/(p['ec50_mir']+max(0.,M))) if 'miR4651' in enabled else 0.
    rel=p['k_release']*max(0.,Ci)
    pr=0. if 'P1' not in enabled else rel
    tr=0. if 'TBK1_014' not in enabled else rel
    mr=0. if 'miR4651' not in enabled else rel
    inf_target=p['weight_prpc']*PC+p['weight_tbk1']*TB
    sen_target=p['weight_progerin']*PRO+p['weight_inflammation']*INF
    return np.array([
       -(p['k_clear_c']+p['k_exchange'])*Cp+p['k_return']*Ct,
       p['k_exchange']*Cp-(p['k_return']+p['k_uptake']+p['k_clear_t'])*Ct,
       p['k_uptake']*Ct-(p['k_release']+p['k_clear_ci'])*Ci,
       pr-p['k_clear_p']*P,
       tr-p['k_clear_tbk']*T,
       mr-p['k_clear_mir']*M,
       p['k_prnp']*((1.-E_p)-R),
       p['k_prpc']*(R-PC),
       p['k_tbk1']*((1.-E_t)-TB),
       p['k_srsf1']*((1.-E_m)-S),
       p['k_progerin']*(S-PRO),
       p['k_inflammation']*(inf_target-INF),
       p['k_senescence']*(sen_target-SEN)
    ],dtype=float)

def simulate(p, dose_nM, tau_h, n_doses, horizon_h, enabled=MODS,
             n_times=161, rtol=1e-7, atol=1e-9):
    """IV nominal plasma nM-equivalent bolus; not converted to administered mg/kg."""
    p=check_parameters(p)
    if dose_nM < 0 or tau_h <= 0 or n_doses < 1 or horizon_h <= 0 or n_times<2:
        raise ModelHold('INVALID_SCHEDULE')
    if not set(enabled).issubset(set(MODS)): raise ModelHold('INVALID_MODULE')
    events=np.arange(int(n_doses),dtype=float)*float(tau_h)
    events=events[events <= horizon_h]
    t_all=np.linspace(0.,horizon_h,int(n_times))
    state=np.array([0.]*6+[1.]*7,dtype=float)
    result=np.zeros((len(t_all),len(state)))
    breaks=np.r_[events, horizon_h]
    # remove duplicate horizon, preserve event at t=0
    if len(breaks)>1 and breaks[-1]<=breaks[-2]: breaks=breaks[:-1]
    start=0.
    for i,e in enumerate(events):
        if i==0: start=float(e)
        state[0]+=float(dose_nM)
        end=float(breaks[i+1]) if i+1<len(breaks) else horizon_h
        if end<=start+1e-12:
            # event exactly at horizon
            if np.isclose(start,horizon_h): result[-1,:]=state
            start=end
            continue
        mask=(t_all>=start-1e-12)&(t_all<end-1e-12)
        if np.isclose(end,horizon_h): mask=(t_all>=start-1e-12)&(t_all<=end+1e-12)
        evals=np.unique(np.r_[t_all[mask],end])
        sol=solve_ivp(rhs,(start,end),state,args=(p,tuple(enabled)),method='BDF',
                      t_eval=evals,rtol=rtol,atol=atol)
        if not sol.success or not np.isfinite(sol.y).all(): raise ModelHold('BDF_FAILED:'+sol.message)
        for j,t in enumerate(t_all[mask]):
            loc=int(np.argmin(np.abs(evals-t)))
            if abs(evals[loc]-t)>1e-7: raise ModelHold('TIME_GRID_MISMATCH')
            result[np.where(mask)[0][j]]=sol.y[:,loc]
        state=sol.y[:,-1]
        start=end
    if np.min(result)<-1e-6: raise ModelHold('NEGATIVE_STATES_NUMERICAL')
    return t_all,result

def response_metric(p, schedule, enabled, output='senescence_rel'):
    t,y=simulate(p,enabled=enabled,**schedule)
    if output not in STATES: raise ModelHold('UNKNOWN_OUTPUT: '+output)
    ix=STATES.index(output)
    return float(1-y[-1,ix]) if output.endswith('_rel') else float(y[-1,ix])

def shapley_decomposition(p,schedule,outcome='senescence_rel'):
    """Exact 3-factor Shapley/Harsanyi on model-specific knockouts, NOT clinical synergy."""
    values={mask:response_metric(p,schedule,mask,outcome) for mask in MASKS}
    from math import factorial
    n=3
    shap={}
    for elem in MODS:
        val=0.
        for S in MASKS:
            if elem in S: continue
            k=len(S)
            w=factorial(k)*factorial(n-k-1)/factorial(n)
            val+=w*(values[tuple(x for x in MODS if x in S or x==elem)]-values[S])
        shap[elem]=float(val)
    full=tuple(MODS)
    triple=(values[full]-sum(values[tuple(x for x in MODS if x!=m)] for m in MODS)
            +sum(values[(m,)] for m in MODS)-values[()])
    return { 'baseline':values[()], 'full':values[full],
      'total_delta':values[full]-values[()], 'shapley':shap,
      'three_way_harsanyi_interaction':float(triple),
      'all_virtual_knockouts':{'|'.join(k) if k else 'NONE':v for k,v in values.items()},
      'label':'CONDITIONAL_IN_SILICO_TARGET_KNOCKOUT_NOT_DRUG_SYNERGY'}

def _sobol_run(task):
    p,schedule,outputs,mods,indices,base=task
    x=p.copy()
    for i,k in enumerate(mods): x[k]=float(indices[i])
    return [response_metric(x,schedule,MODS,outcome) for outcome in outputs]

def sobol_analysis(p, ranges, schedule, outputs=('progerin_rel','senescence_rel'), m=8, workers=1, seed=11):
    """Saltelli/Jansen estimators using independent Sobol A/B columns.
    n_solves = (d+2)*2**m. Input ranges must have explicit source provenance upstream.
    """
    keys=list(ranges)
    d=len(keys)
    if not 1<=d<=20 or not 5<=m<=16: raise ModelHold('INVALID_SOBOL_DIMENSIONS_OR_DEPTH')
    N=2**m
    q=qmc.Sobol(d=2*d,scramble=True,seed=seed)
    raw=q.random_base2(m)
    A=raw[:,:d]; B=raw[:,d:]
    def convert(arr):
        v=np.empty_like(arr)
        for j,k in enumerate(keys):
            lo,hi=ranges[k]
            if lo<=0 or hi<=lo: raise ModelHold('INVALID_SOBOL_RANGE:'+k)
            v[:,j]=np.exp(np.log(lo)+(np.log(hi)-np.log(lo))*arr[:,j])
        return v
    A=convert(A);B=convert(B)
    mat=np.concatenate([A,B]+[np.column_stack([B[:,j] if j==i else A[:,j] for j in range(d)]) for i in range(d)])
    if workers<1: raise ModelHold('INVALID_WORKERS')
    tasks=((p,schedule,outputs,keys,row,{}) for row in mat)
    if workers==1: res=[_sobol_run(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            res=list(pool.map(_sobol_run,tasks,chunksize=max(1,int(len(mat)/(workers*8)))))
    arr=np.asarray(res,dtype=float)
    if not np.isfinite(arr).all(): raise ModelHold('NONFINITE_SOBOL_OUTPUT')
    results=[]
    for oi,out in enumerate(outputs):
        YA,YB=arr[:N,oi],arr[N:2*N,oi]
        var=np.var(np.r_[YA,YB],ddof=1)
        if var<1e-16: raise ModelHold('ZERO_VARIANCE_SOBOL_'+out)
        for i,k in enumerate(keys):
            YC=arr[(i+2)*N:(i+3)*N,oi]
            s1=float(np.mean(YB*(YC-YA))/var)
            st=float(np.mean((YA-YC)**2)/(2*var))
            results.append({'outcome':out,'parameter':k,'S1':s1,'ST':st,
                            'N_base':N,'N_model_solves':len(mat),
                            'method':'Saltelli_S1_Jansen_ST_loguniform_bounds',
                            'note':'Finite-sample estimators can be negative or >1; do not silently clamp.'})
    return results

def synthetic_test_only_parameters():
    """UNPHYSICAL constant fixture for software verification ONLY; never publication inputs."""
    p={k:0.15 for k in PARAMETERS}
    for k in ('ec50_p','ec50_tbk','ec50_mir'):p[k]=1.
    for k in ('weight_prpc','weight_tbk1','weight_progerin','weight_inflammation'):p[k]=0.5
    return p
