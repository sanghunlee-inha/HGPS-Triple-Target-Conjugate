import math,sys,tempfile
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
sys.path.insert(0,str(Path(__file__).parent))
import fig7_delivery as model

def synthetic(prefix):
    rng=np.random.default_rng(42)
    theta=rng.uniform(0,2*math.pi,650)
    u=rng.uniform(-1,1,650)
    pts=np.stack([np.sqrt(1-u*u)*np.cos(theta),np.sqrt(1-u*u)*np.sin(theta),u],axis=1)*0.003
    hull=ConvexHull(pts)
    faces=hull.simplices.copy()
    # Wall occupies all but ten cap-like selected faces; source is explicitly synthetic.
    marker=np.ones(len(faces),int)
    centers=pts[faces].mean(axis=1)
    for id in range(2,12):
        ind=np.argmax(centers @ np.array([math.cos(id*2.399),math.sin(id*2.399),math.cos(id)]))
        while marker[ind]!=1:ind=(ind+1)%len(faces)
        marker[ind]=id
    with (Path(str(prefix)+'.node')).open('w') as f:
        f.write(f'{len(pts)} 3 0 0\n')
        for i,p in enumerate(pts,1): f.write(f'{i} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}\n')
    with (Path(str(prefix)+'.face')).open('w') as f:
        f.write(f'{len(faces)} 1\n')
        for i,(face,m) in enumerate(zip(faces,marker),1):
            a,b,c=face+1
            f.write(f'{i} {a} {b} {c} {int(m)}\n')

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as d:
        prefix=Path(d)/'quality'/'patient_11patch.1';prefix.parent.mkdir()
        synthetic(prefix)
        rc=model.main(['--prefix',str(prefix),'--out',str(Path(d)/'outputs'),'--samples','128','--no-figure'])
        assert rc==0
        j=__import__('json').loads((Path(d)/'outputs'/'AUDIT.json').read_text())
        assert j['max_onepass_mass_balance_abs_error']<1e-12
        assert j['surface_markers']['port_2']==1
        roi=Path(d)/'fake_roi.csv'
        roi.write_text('roi_name,x_raw,y_raw,z_raw,radius_raw,annotation_evidence\nlesion,0,0,0,2,UNVERIFIED\n')
        try:model.main(['--prefix',str(prefix),'--out',str(Path(d)/'roi'),'--samples','64','--roi-csv',str(roi),'--no-figure'])
        except ValueError as e: assert 'MANUAL_MRA_ANNOTATION_VERIFIED' in str(e)
        else:raise AssertionError('Bad ROI label was accepted')
        # Render at publication dpi for the real job, lower dpi for rapid smoke test.
        rc=model.main(['--prefix',str(prefix),'--out',str(Path(d)/'figure'),'--samples','64','--dpi','120'])
        assert rc==0
        assert (Path(d)/'figure'/'FIG7_PATIENT_SURFACE_DELIVERY_CONDITIONAL_a-f_600dpi.png').stat().st_size>10000
        print('SELFTEST_PASS_READONLY_SURFACE_11_MARKERS_GEODESIC_MASS_BALANCE_ROI_HOLD_PNG_PDF')
