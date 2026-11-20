#!/usr/bin/env python3
"""Read OpenFOAM surfaceFieldValue sum(phi) dat files without inventing values.
The sign is outward mesh-normal; flow balance is not a patient treatment effect.
"""
import csv,json,sys,re
from pathlib import Path
root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
records=[]
for p in sorted(root.glob('flow_port_*/**/*.dat')):
    name=re.search(r'flow_(port_\d+)',str(p))
    if not name:continue
    raw=[line.strip() for line in p.read_text(encoding='utf-8',errors='replace').splitlines() if line.strip() and not line.startswith('#')]
    vals=[]
    for s in raw:
        parts=s.split()
        if len(parts)>=2:
            try:vals.append((float(parts[0]),float(parts[1])))
            except ValueError:pass
    if vals:
        time,phi=vals[-1]
        records.append({'patch':name.group(1),'last_model_time':time,'signed_volume_flux_m3_s':phi,'file':str(p)})
if records:
    with (out/'REFERENCE_3D_PATCH_FLUX.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
report={'gate':'SOURCE_PATCH_FLUX_EXTRACTED_NOT_PHYSIOLOGICAL_VALIDATION' if records else 'HOLD_NO_FLUX_OUTPUT',
        'patch_count':len(records),'net_signed_flux_m3_s':sum(r['signed_volume_flux_m3_s'] for r in records),
        'flow_units_check_required':True,'no_claim_of_drug_recovery':True}
(out/'REFERENCE_FLUX_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
