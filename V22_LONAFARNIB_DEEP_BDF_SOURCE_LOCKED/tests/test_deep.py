from __future__ import annotations
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

MOD=Path(__file__).resolve().parents[1]/'v22_deep.py'
spec=importlib.util.spec_from_file_location('v22_deep_user',MOD)
v22=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v22)


def synthetic_timeline():
    rows=[]
    for t in (0.,10.,40.,120.,365.25,730.5):
        d={'horizon_days':t}
        for arm in ('lona','conj','untreated'):
            for i,k in enumerate(v22.AXES):
                d[arm+'_'+k]=((1.+(i+1)*0.15*math.sin(t/400.)) +
                    (0. if arm=='untreated' else (0.25 if arm=='conj' else 0.02)*t/730.5))
        rows.append(d)
    return rows


class AnalyticODETests(unittest.TestCase):
    def test_closed_form_piecewise_vs_dop853(self):
        t=np.array([0.,1.,4.,15.,40.,90.,730.5])
        rng=np.random.default_rng(11)
        forc=rng.normal(0.,.6,len(t))
        val=v22.exact_linear_ode(t,forc,k_repair=.001)
        # split exactly at the non-smooth interpolation knots, high tolerance
        y=0.
        for i in range(len(t)-1):
            x0,x1=t[i:i+2];f0,f1=forc[i:i+2]
            def rhs(s,z):return [-0.001*z[0]-float(f0+(f1-f0)*(s-x0)/(x1-x0))]
            sol=solve_ivp(rhs,(x0,x1),[y],method='DOP853',rtol=2e-12,atol=2e-14)
            self.assertTrue(sol.success)
            y=sol.y[0,-1]
        self.assertAlmostEqual(y,val,places=9)

    def test_zero_decay_limit(self):
        self.assertAlmostEqual(v22.exact_linear_ode([0,2,5,9],[2,3,4,1],k_repair=0),
                               -((2+3)*1+(3+4)*1.5+(4+1)*2),places=10)

    def test_source_0d_rate_exact_tenfold_identity(self):
        rows=synthetic_timeline()
        low=v22.exact_remodel_from_timeline(rows,'low')
        high=v22.exact_remodel_from_timeline(rows,'high')
        self.assertAlmostEqual(high['delta_logR_exact'],10*low['delta_logR_exact'],places=10)
        self.assertAlmostEqual(sum(high['d_logR_'+x] for x in v22.AXES),
                               high['delta_logR_exact'],places=11)
        self.assertEqual(math.copysign(1,high['delta_R_exact']),math.copysign(1,low['delta_R_exact']))

    def test_no_perturbation_has_unity_resistance(self):
        data=synthetic_timeline()
        for r in data:
            for arm in ('lona','conj'):
                for k in v22.AXES:r[arm+'_'+k]=r['untreated_'+k]
        result=v22.exact_remodel_from_timeline(data,'high')
        self.assertAlmostEqual(result['delta_R_exact'],0.,places=13)
        self.assertAlmostEqual(result['R_conj'],1.,places=13)

    def test_linear_force_when_decay_zero(self):
        for z in [0.,1e-9,1e-6,1e-5,.01,.1,1.,2.]:
            p1,p2=v22.exp_integral_factors(z)
            ref1=math.expm1(-z)/(-z) if z else 1.
            self.assertAlmostEqual(p1,ref1,places=10)
            self.assertGreater(p2,0)

    def test_shapley_efficiency_nonlinear(self):
        values={mask:float((mask&1)*2+((mask>>1)&1)*3+((mask>>2)&1)*4+5*(mask==7)) for mask in range(8)}
        sh=v22.shapley3(values)
        self.assertAlmostEqual(sum(sh.values()),14.)

    def test_shapley_rejects_missing_arm(self):
        with self.assertRaises(v22.Hold):v22.shapley3({0:0,7:1})

    def test_original_source_hash_provenance(self):
        root=Path(__file__).resolve().parents[1]/'reference'
        for name,expect in v22.EXPECTED_SHA.items():
            self.assertEqual(v22.sha256(root/name),expect)

    def test_robust_outlier_filter_median_not_replaced(self):
        a=np.array([.1,-.1,-.15,.2,1.2,10**8])
        out=v22.descriptive_stats(a)
        self.assertGreater(out['mean'],10**6)
        self.assertLess(abs(out['median']),1.)

    def test_plan_contains_every_case_once(self):
        rows=[]
        for seed in v22.SEEDS:
            for i in range(1,33):
                x=float(i) if i%2 else -float(i)
                rows.append({'seed':seed,'idx':i*64,'release_mode':'independent',
                             'delta_R_low':x/10.,'delta_R_high':1001. if seed==11 and i==1 else x})
        with self.assertRaises(v22.Hold):
            v22.build_stress_design_plan(pd.DataFrame(rows))

    def test_compare_replay_detects_nonzero_drift(self):
        keys=['delta_'+k for k in v22.MODEL_ENDS]+['lona_'+k for k in v22.MODEL_ENDS]+['conj_'+k for k in v22.MODEL_ENDS]
        rows=[{'horizon_days':t,**{k:1. for k in keys}} for t in (182.625,365.25,730.5)]
        orig={mode:{'rows':rows} for mode in v22.MODES}
        changed=[{'mode':mode,'rows':[dict(x) for x in rows]} for mode in v22.MODES]
        largest,bad,_=v22.compare_stored_bdf(orig,changed)
        self.assertEqual(bad,[])
        changed[0]['rows'][1][keys[0]]=1.1
        largest,bad,_=v22.compare_stored_bdf(orig,changed)
        self.assertGreater(largest,.09)
        self.assertGreater(len(bad),0)


if __name__=='__main__': unittest.main()
