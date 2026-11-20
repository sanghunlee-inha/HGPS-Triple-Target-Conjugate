import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis_pipeline import qsp_numeric_selftest,analyze_assays,analyze_v22,ModelHold,ARMS
from qsp_engine import sobol_analysis,synthetic_test_only_parameters,simulate,STATES
from types import SimpleNamespace

def test_dose_zero_and_shapley():
    x=qsp_numeric_selftest()
    assert x['test'].startswith('PASS')

def test_sobol_small():
    p=synthetic_test_only_parameters()
    v=sobol_analysis(p,{'k_clear_c':(.1,.2),'ec50_p':(.8,1.2)},
                     {'dose_nM':2.,'tau_h':24.,'n_doses':1,'horizon_h':36.,'n_times':41},
                     outputs=['progerin_rel','senescence_rel'],m=5,workers=1,seed=12)
    assert len(v)==4
    assert all(np.isfinite(x['ST']) for x in v)

def test_measured_assays_and_three_way(tmp_path):
    # SOFTWARE TEST FIXTURE, not scientific data.
    rows=[]
    names=['VEHICLE','P1','TBK1_014','miR4651','P1_TBK1_014',
           'P1_miR4651','TBK1_014_miR4651','TRIPLE_FREE','CONJUGATE']
    benefits=[0,.12,.17,.19,.26,.3,.34,.44,.49]
    for donor in ('D1','D2','D3'):
        for dose in (1.,2.,3.,4.):
            for arm,b in zip(names,benefits):
                for rep in (1,2):
                    rows.append({'study_id':'synthetic','cell_model':'synthetic_cell',
                        'donor_id':donor,'biological_replicate':f'r{rep}',
                        'design_id':str(dose),'arm':arm,'readout':'progerin',
                        'time_h':48.,'dose_value':dose if arm!='VEHICLE' else 0.,
                        'dose_unit':'synthetic_units','value':1-b*dose/(1+dose),
                        'direction':'lower','evidence_type':'measured'})
    fp=tmp_path/'test_fixture_only.csv';pd.DataFrame(rows).to_csv(fp,index=False)
    args=SimpleNamespace(assays=str(fp),bootstraps=100,seed=11)
    out=tmp_path/'out';out.mkdir()
    a=analyze_assays(args,out)
    assert a['n_donors']==3
    df=pd.read_csv(out/'12_MATCHED_THREE_WAY_INTERACTIONS.csv')
    assert len(df)==12 and df.bliss_delta.notna().all()
    assert (df.bliss_gate=='EXPLORATORY_MATCHED_NOMINAL_DOSE_NEEDS_EXPOSURE').all()
    assert (out/'13_EXPERIMENTAL_HILL_FITS.csv').exists()

def test_v22_cluster_paired(tmp_path):
    df=pd.DataFrame([{'id':i,'seed':i%5,'treatment':a,'response':float(i)+(0.4 if a=='C' else 0.)}
                     for i in range(25) for a in ('L','C')])
    csv=tmp_path/'v22_fixture.csv';df.to_csv(csv,index=False)
    conf={'pair_keys':['id'],'arm_column':'treatment','outcome_column':'response',
      'label_a':'C','label_b':'L','benefit_direction':'higher','independent_cluster':'seed'}
    m=tmp_path/'mapping.json';m.write_text(json.dumps(conf),encoding='utf-8')
    args=SimpleNamespace(v22=str(csv),v22_mapping=str(m),seed=3,bootstraps=100)
    out=tmp_path/'out';out.mkdir()
    res=analyze_v22(args,out,[])
    assert res['n_paired_cases']==25
    assert abs(res['cluster_mean_delta']-0.4)<1e-10
    df.loc[len(df)]={'id':0,'seed':0,'treatment':'C','response':0.9}
    df.to_csv(csv,index=False)
    with pytest.raises(ModelHold,match='DUPLICATE_WITHIN_ARM'):
        analyze_v22(args,out,[])

def test_measured_calibration_software_fixture(tmp_path):
    # Synthetic software test only. Explicitly never real experimental inference.
    from qsp_engine import synthetic_test_only_parameters,simulate,PARAMETERS
    from calibration import calibrate
    p=synthetic_test_only_parameters()
    param={ 'parameters':{k:{'value':v,'lower':v*.7,'upper':v*1.3,
                 'evidence_type':'synthetic_software_test','source_reference':'unit_test_only'} for k,v in p.items()},
            'fit_parameters':['k_clear_c']}
    cfg=tmp_path/'test_params.json';cfg.write_text(json.dumps(param))
    dat=[]
    for donor in ['a','b','c']:
        t,y=simulate(p,dose_nM=2.,tau_h=24.,n_doses=1,horizon_h=24.,n_times=25)
        for j in (2,4,6,8,10,12,16,20,24):
            dat.append({'study_id':'synthetic','donor_id':donor,'condition_id':'dose2',
                  'time_h':int(t[j]),'dose_nM':2.,'tau_h':24.,'n_doses':1,
                  'analyte':'conjugate_plasma_nM','value':float(y[j,0]),'sigma':.05,
                  'evidence_type':'measured','source_reference':'TEST_SYNTHETIC_NOT_PUBLISHABLE'})
    f=tmp_path/'TEST_ONLY_NOT_REAL.csv';pd.DataFrame(dat).to_csv(f,index=False)
    r=calibrate(f,cfg,tmp_path/'out',seed=33,starts=2)
    assert r['n_train_donors']==2 and r['n_test_donors']==1
    assert r['jacobian_rank']==1
