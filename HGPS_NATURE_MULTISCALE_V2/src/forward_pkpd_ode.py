"""Actual BDF mechanistic PKPD forward function called by Figure 8 Sobol samples.
Requires env HGPS_PKPD_ODE_CONFIG pointing to the actual documented model file.
No surrogate, no precomputed outcome lookup, no synthetic treatment effect.
"""
import os,json,functools
from pathlib import Path
from numerical_pkpd import parameter_table,dose_table,integrate,K

@functools.lru_cache(maxsize=1)
def _load():
    p=os.environ.get('HGPS_PKPD_ODE_CONFIG')
    if not p or not Path(p).is_file():raise ValueError('FIG8_PKPD_CONFIG_ENV_NOT_SET')
    c=json.loads(Path(p).read_text(encoding='utf-8-sig'))
    baseline,_=parameter_table(c['parameter_table'])
    doses=dose_table(c['dose_table'])
    return c,baseline,doses

def evaluate(x):
    c,base,doses=_load()
    p=dict(base)
    for k,value in x.items():
        if k not in K:raise ValueError('FIG8_UNSUPPORTED_PKPD_SOBOL_PARAMETER_'+str(k))
        p[k]=float(value)
    t,y,_=integrate(p,doses,float(c['end_day']),float(c.get('rtol',1e-8)),
      float(c.get('atol',1e-10)),float(c.get('max_step_days',.25)),
      float(c.get('dense_step_days',.1)),'BDF')
    return {'injury_proxy_final':float(y[-1,11]),'progerin_final':float(y[-1,10]),
            'PRNP_final':float(y[-1,7]),'TBK1_activity_final':float(y[-1,8]),
            'SRSF1_final':float(y[-1,9])}
