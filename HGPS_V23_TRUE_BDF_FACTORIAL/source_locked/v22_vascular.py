"""Explicitly hypothetical ZERO-dimensional remodeling and fluid resistance.

This is not computational fluid dynamics; no patient MRI anatomy, lumen
segmentation, vessel-specific blood pressure or clinical calibration.
It is a falsifiable 0D structural sensitivity model: R/R0=(r0/r)^4 under
steady Newtonian Poiseuille assumptions, NOT a prediction of clinical stenosis.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.integrate import solve_ivp

class InvalidVascularInput(ValueError):pass

def remodel(timeline,arm,*,k_repair=0.001,k_ec=0.00035,k_vsmc=0.00035,k_ecm=0.00025,
            baseline=1.0):
 if arm not in ('lona','conj','untreated'):raise InvalidVascularInput('UNKNOWN_ARM')
 if any(not np.isfinite(v) or v<0 for v in (k_repair,k_ec,k_vsmc,k_ecm)):
  raise InvalidVascularInput('NEGATIVE_RATE')
 t=np.array([float(row['horizon_days']) for row in timeline]);order=np.argsort(t)
 t=t[order]
 if len(t)<4 or abs(t[0])>1e-12 or (np.diff(t)<=0).any():
  raise InvalidVascularInput('TIMELINE_MUST_START_DAY_ZERO_MONOTONIC')
 cells={k:np.array([float(row[arm+'_'+k]) for row in timeline])[order]
        for k in ('ec_activation','vsmc_dysfunction','ecm_stiffness')}
 reference={k:np.array([float(row['untreated_'+k]) for row in timeline])[order]
        for k in ('ec_activation','vsmc_dysfunction','ecm_stiffness')}
 if any(not np.isfinite(v).all() for v in (*cells.values(),*reference.values())):
  raise InvalidVascularInput('NAN_IN_EVIDENCE_TIMELINE')
 def rhs(day,y):
  # All rates are uncalibrated. Compare against same-biology untreated
  # trajectory, avoiding an invented absolute vessel radius baseline.
  ec=np.interp(day,t,cells['ec_activation'])-np.interp(day,t,reference['ec_activation'])
  vs=np.interp(day,t,cells['vsmc_dysfunction'])-np.interp(day,t,reference['vsmc_dysfunction'])
  matrix=np.interp(day,t,cells['ecm_stiffness'])-np.interp(day,t,reference['ecm_stiffness'])
  return [-k_repair*y[0]-k_ec*ec-k_vsmc*vs-k_ecm*matrix]
 sol=solve_ivp(rhs,(float(t[0]),float(t[-1])),[0.],t_eval=t,
               method='DOP853',rtol=1e-9,atol=1e-11)
 if not sol.success:raise InvalidVascularInput('0D_INTEGRATION_FAIL')
 logradius=sol.y[0];r=np.exp(logradius)
 if not np.isfinite(r).all() or r.min()<0.01 or r.max()>100.:
  raise InvalidVascularInput('HOLD_UNBOUNDED_0D_RADIUS')
 return {'t_days':t,'radius_relative':r,
   'resistance_relative_fixed_flow':r**-4,
   'wall_shear_relative_fixed_flow':r**-3,
   'dimension':'0D_Poiseuille_hypothetical_NOT_CFD',
   'not_patient_vessel':True}

def paired_remodel(timeline,**coeff):
 return {arm:remodel(timeline,arm,**coeff) for arm in ('lona','conj','untreated')}
