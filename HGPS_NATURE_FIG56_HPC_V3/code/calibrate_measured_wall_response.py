#!/usr/bin/env python3
"""Measured PD -> ex vivo wall stiffness association with donor-grouped CV.
This is NOT FSI, drug efficacy, proof of causality, or approved patient parameter transfer.
"""
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
REQUIRED=['donor_id','arm','progerin_fraction','prpc_fraction','tbk1_activity_fraction','measured_elastic_modulus_pa']

def ridge(x,y,alpha):
    mu=x.mean(0);sd=x.std(0);sd[sd==0]=1
    z=(x-mu)/sd;model=np.column_stack([np.ones(len(z)),z])
    coef=np.linalg.solve(model.T@model+np.diag([0]+[alpha]*x.shape[1]),model.T@y)
    return coef,mu,sd

def predict(beta,mu,sd,x):return np.column_stack([np.ones(len(x)),(x-mu)/sd])@beta

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--input',required=True);pa.add_argument('--out',required=True);
    pa.add_argument('--approved-analysis-plan',required=True);a=pa.parse_args()
    plan=json.loads(Path(a.approved_analysis_plan).read_text(encoding='utf-8-sig'))
    if plan.get('model')!='donor_grouped_ridge_cross_validation' or plan.get('approved') is not True:
        raise RuntimeError('HOLD: no registered PD-to-stiffness calibration analysis plan')
    alpha=float(plan.get('ridge_regularization'))
    if not np.isfinite(alpha) or alpha<0:raise RuntimeError('HOLD invalid regularization')
    with open(a.input,encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows or any(k not in rows[0] for k in REQUIRED):raise RuntimeError('HOLD required measured source columns missing')
    ids=[];Xs=[];Ys=[];arms=[]
    for i,row in enumerate(rows):
        ids.append(row['donor_id']);arms.append(row['arm'])
        x=np.array([float(row[n]) for n in REQUIRED[2:5]],dtype=float)
        y=float(row['measured_elastic_modulus_pa'])
        if not np.isfinite(x).all() or np.any(x<0) or not np.isfinite(y) or y<=0:raise ValueError('Invalid physical measured row '+str(i))
        Xs.append(x);Ys.append(y)
    X=np.stack(Xs);Y=np.array(Ys);don=np.array(ids)
    unique=np.unique(don)
    if len(unique)<5:raise RuntimeError('HOLD: at least 5 independent donors for leave-one-donor-out audit')
    yhat=np.empty(len(Y))
    for u in unique:
        train=don!=u;test=~train
        coef,mu,sd=ridge(X[train],Y[train],alpha);yhat[test]=predict(coef,mu,sd,X[test])
    ss=float(np.sum((Y-Y.mean())**2));R2=float(1-np.sum((Y-yhat)**2)/ss) if ss>0 else None
    mae=float(np.mean(np.abs(Y-yhat)))
    coef,mu,sd=ridge(X,Y,alpha)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    with (out/'MEASURED_WALL_LODO_PREDICTIONS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['donor_id','arm','measured_E_pa','predicted_E_pa','residual_pa'])
        for i in range(len(Y)):w.writerow([ids[i],arms[i],repr(Y[i]),repr(yhat[i]),repr(Y[i]-yhat[i])])
    plt.rcParams.update({'font.size':23,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(9,8),constrained_layout=True)
    ax.scatter(Y,yhat,s=65,alpha=.65);bounds=[min(Y.min(),yhat.min()),max(Y.max(),yhat.max())]
    ax.plot(bounds,bounds,'k--',lw=1.5)
    ax.set_xlabel('Measured vessel elastic modulus (Pa)');ax.set_ylabel('Held-out donor prediction (Pa)')
    ax.set_title('Source-measured wall response | leave-one-donor-out CV',fontsize=22)
    fig.savefig(out/'FIG5_EXVIVO_WALL_CALIBRATION_23PT.png',dpi=230,facecolor='white')
    fig.savefig(out/'FIG5_EXVIVO_WALL_CALIBRATION_23PT.pdf');plt.close(fig)
    with open(a.input,'rb') as f:source_sha=hashlib.file_digest(f,'sha256').hexdigest()
    audit={'gate':'MODEL_ASSOCIATION_CV_ONLY_NOT_CAUSAL_OR_FSI_VALIDATION','n_rows':len(Y),'n_donors':len(unique),
      'LODO_R2':R2,'LODO_MAE_Pa':mae,'ridge_alpha_approved':alpha,'fitted_coef_standardized':coef.tolist(),
      'feature_center':mu.tolist(),'feature_scale':sd.tolist(),
      'input_sha256':source_sha,'measured_vascular_compliance_not_inferred':True,
      'clinical_patient_transfer_gate':'HOLD_UNTIL_EXTERNAL_WALL_MECHANICS_VALIDATION'}
    (out/'WALL_CALIBRATION_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))
if __name__=='__main__':sys.exit(main())
