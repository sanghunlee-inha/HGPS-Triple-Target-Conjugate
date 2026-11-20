#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HGPS Figure 7 patient surface / exploratory delivery-compartment analysis.

SOURCE: V10 patient 11-patch TetGen .node/.face, without invoking CFD.
The surface geodesic from *assumed* inlet port_2 is NOT blood flow or actual route.
Conditional one-pass partition is a mass-balanced mathematical thought experiment,
NOT measured dose, tissue concentration, lesion-target uptake, or target engagement.
No CFD or mesh mutations. No off-target organs predicted.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, time, re
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.stats import spearmanr

VERSION='HGPS_FIG7_PATIENT_SURFACE_DELIVERY_SCENARIO_V1'
MARKERS={i:('wall' if i==1 else f'port_{i}') for i in range(1,12)}
RNG_SEED=20261117


def die(msg):raise ValueError(msg)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def noncomment(path):
    with Path(path).open('r',encoding='ascii') as f:
        for line in f:
            s=line.partition('#')[0].strip()
            if s:yield s

def read_tetgen_surface(prefix):
    prefix=Path(prefix)
    node=Path(str(prefix)+'.node');face=Path(str(prefix)+'.face')
    if not node.is_file() or not face.is_file():die('MISSING_NODE_OR_FACE')
    niter=noncomment(node);head=next(niter).split()
    n=int(head[0]);ndim=int(head[1]);nattr=int(head[2]);nmark=int(head[3])
    if not (0<n<3_000_000 and ndim==3):die('INVALID_NODE_HEADER')
    pts=np.zeros((n,3),np.float64);ids={}
    for k in range(n):
        row=next(niter).split()
        if len(row)<4+nattr+nmark:die('NODE_ROW_TRUNCATED')
        idx=int(row[0]);
        if idx in ids:die('DUPLICATE_NODE_ID')
        ids[idx]=k
        pts[k]=[float(v) for v in row[1:4]]
    if not np.isfinite(pts).all() or len(ids)!=n:die('NONFINITE_OR_MISSING_NODES')
    fit=noncomment(face);fhead=next(fit).split(); nf=int(fhead[0]);hasmarker=int(fhead[1])
    if not (0<nf<5_000_000 and hasmarker==1):die('MARKED_FACE_REQUIRED')
    tri=np.empty((nf,3),np.int32);mrk=np.empty(nf,np.int8)
    for k in range(nf):
        row=next(fit).split()
        if len(row)<5:die(f'FACE_ROW_TRUNCATED_{k}')
        try:tri[k]=[ids[int(x)] for x in row[1:4]]
        except KeyError:die(f'FACE_POINT_ID_MISSING_{k}')
        mrk[k]=int(row[4]);
        if mrk[k] not in MARKERS:die(f'UNKNOWN_PATCH_MARKER_{k}')
        if len(set(tri[k]))!=3:die(f'NONTRIANGLE_{k}')
    if set(mrk.tolist())!=set(MARKERS):die(f'REQUIRE_ALL_11_PATCHES {set(mrk.tolist())}')
    if nf!=sum(np.unique(mrk,return_counts=True)[1]):die('FACE_COUNT_BAD')
    # We only consume the surface; unreferenced interior tet nodes can exist.
    used=np.unique(tri.ravel());p=pts[used]
    # remap to compact connected surface node indices
    tri=np.searchsorted(used,tri).astype(np.int32);pts=p
    v01=pts[tri[:,1]]-pts[tri[:,0]];v02=pts[tri[:,2]]-pts[tri[:,0]]
    ar=.5*np.linalg.norm(np.cross(v01,v02),axis=1)
    if not (ar>0).all():die('ZERO_AREA_TRIANGLE')
    cent=(pts[tri[:,0]]+pts[tri[:,1]]+pts[tri[:,2]])/3
    return pts,tri,mrk,ar,cent,node,face

def maybe_verify_v10_source(prefix,marks):
    # Quality is derived from full marked source. Audit optional but if present compare exact patch counts.
    p=Path(prefix).parents[1]/'SURFACE_AUDIT.json'
    if not p.is_file():return {'source_geometry_audit':'NOT_FOUND','exact_source_counts_gate':None}
    audit=json.loads(p.read_text(encoding='utf-8'))
    expected=audit.get('patches') or {}
    if set(expected)!=set(MARKERS.values()):die('SOURCE_AUDIT_PATCH_NAMES_MISMATCH')
    for i,name in MARKERS.items():
        expected_n=int(expected[name]['source_faces']);actual=int((marks==i).sum())
        if expected_n!=actual:die(f'NOT_ORIGINAL_PATCH_{name} expected={expected_n} actual={actual}')
    return {'source_geometry_audit':str(p),'exact_source_counts_gate':True}

def geodesic_from_port2(pts,tri,mrk):
    edges=np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]))
    edges=np.sort(edges,axis=1)
    edges=np.unique(edges,axis=0)
    d=np.linalg.norm(pts[edges[:,0]]-pts[edges[:,1]],axis=1)
    if np.any(d<=0):die('DEGENERATE_SURFACE_EDGE')
    i=np.r_[edges[:,0],edges[:,1]];j=np.r_[edges[:,1],edges[:,0]];v=np.r_[d,d]
    graph=coo_matrix((v,(i,j)),shape=(len(pts),len(pts))).tocsr()
    cc,labels=connected_components(graph,directed=False)
    if cc!=1:die(f'SURFACE_NOT_CONNECTED_COMPONENTS_{cc}')
    sources=np.unique(tri[mrk==2].ravel())
    if not len(sources):die('NO_ASSUMED_INLET_PORT_2')
    # A single virtual source connects to all port_2 vertices, avoiding one Dijkstra run per inlet node.
    origin=np.full(len(sources),len(pts),dtype=np.int64)
    gcoo=graph.tocoo()
    ii=np.concatenate((gcoo.row,origin,sources))
    jj=np.concatenate((gcoo.col,sources,origin))
    ww=np.concatenate((gcoo.data,np.full(2*len(sources),1e-100)))
    G=coo_matrix((ww,(ii,jj)),shape=(len(pts)+1,len(pts)+1)).tocsr()
    dist=dijkstra(G,directed=False,indices=len(pts))[:-1]
    if not np.isfinite(dist).all():die('DISTANCE_NONFINITE')
    return dist,{'surface_nodes':len(pts),'graph_edges':len(edges),'inlet_vertices':len(sources),'component_count':cc}

def construct_bands(d_tri,area,mask,bins=24):
    # Area-weighted quantile bands on the original patient wall surface.
    ids=np.flatnonzero(mask);order=ids[np.argsort(d_tri[ids])]
    cum=np.cumsum(area[order]);cum/=cum[-1]
    band_index=np.full(len(area),-1,np.int16)
    b=np.minimum(bins-1,np.floor((cum-1e-12)*bins).astype(np.int16))
    band_index[order]=b
    table=[];total=float(area[mask].sum())
    for k in range(bins):
        m=band_index==k
        if not m.any():die('EMPTY_GEODESIC_BAND_'+str(k))
        table.append({'band':k,'area_fraction':float(area[m].sum()/total),
                      'distance_raw_areaweighted':float(np.average(d_tri[m],weights=area[m])),
                      'faces':int(m.sum())})
    s=np.array([r['distance_raw_areaweighted'] for r in table]); denom=max(float(s[-1]-s[0]),1e-30)
    for row,z in zip(table,(s-s[0])/denom):row['geodesic_position_0to1']=float(z)
    return band_index,table

def uniform_or_log(rng,a,b,n,log=False):
    return np.exp(rng.uniform(math.log(a),math.log(b),n)) if log else rng.uniform(a,b,n)

def simulate(bands,n,seed):
    if n<64:die('SAMPLES_LT_64')
    rng=np.random.default_rng(seed);k=len(bands)
    area=np.array([x['area_fraction'] for x in bands]);s=np.array([x['geodesic_position_0to1'] for x in bands])
    params={
        'wall_capture_strength':uniform_or_log(rng,.02,2.0,n,True),
        'geodesic_attenuation_assumption':uniform_or_log(rng,0.,3.,n),
        'ec_endosomal_escape':uniform_or_log(rng,.01,.45,n),
        'transcytosis_fraction':uniform_or_log(rng,.01,.40,n),
        'smc_uptake_fraction':uniform_or_log(rng,.01,.80,n),
        'smc_endosomal_escape':uniform_or_log(rng,.01,.50,n),
        'release_p1':uniform_or_log(rng,.10,.95,n),
        'release_tbk1i':uniform_or_log(rng,.10,.95,n),
        'release_mir4651':uniform_or_log(rng,.05,.90,n),
        'mir_risc_loading_fraction':uniform_or_log(rng,.01,.75,n),
    }
    if (params['ec_endosomal_escape']+params['transcytosis_fraction']>1).any():die('INVALID_ENDOTHELIAL_MASS_SPLIT')
    lumen=np.ones(n);cap=np.zeros((n,k));ec=np.zeros((n,k));smc=np.zeros((n,k))
    for j in range(k):
        relative=np.exp(-params['geodesic_attenuation_assumption']*s[j])
        p=1-np.exp(-params['wall_capture_strength']*area[j]*relative)
        cap[:,j]=lumen*p
        lumen-=cap[:,j]
        ec[:,j]=cap[:,j]*params['ec_endosomal_escape']
        smc[:,j]=cap[:,j]*params['transcytosis_fraction']*params['smc_uptake_fraction']*params['smc_endosomal_escape']
    residual=cap.sum(axis=1)-ec.sum(axis=1)-smc.sum(axis=1)
    mb=lumen+ec.sum(axis=1)+smc.sum(axis=1)+residual
    maxerr=float(np.max(abs(mb-1)))
    if maxerr>3e-12 or np.any(residual<0):die(f'ONE_PASS_MASS_BUDGET_FAILED {maxerr}')
    # These are access prerequisites, never measured target binding or modulation.
    ec_access={c:ec*params[f'release_{c}'][:,None]*(params['mir_risc_loading_fraction'][:,None] if c=='mir4651' else 1)
               for c in ('p1','tbk1i','mir4651')}
    smc_access={c:smc*params[f'release_{c}'][:,None]*(params['mir_risc_loading_fraction'][:,None] if c=='mir4651' else 1)
                for c in ('p1','tbk1i','mir4651')}
    return params,cap,ec,smc,ec_access,smc_access,lumen,residual,maxerr

def write_csv(path,rows):
    rows=list(rows)
    if not rows: return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def interpret_roi(path,pts,tri,mrk,cent,area,band_index,cap,ec_acc,smc_acc):
    if not path:return []
    rows=[]
    with Path(path).open('r',encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
    if not records:die('EMPTY_ROI_LABELS')
    for record in records:
        if record.get('annotation_evidence','').strip()!='MANUAL_MRA_ANNOTATION_VERIFIED':
            die('ROI_LABELS_REQUIRE_MANUAL_MRA_ANNOTATION_VERIFIED')
        name=record.get('roi_name','').strip()
        if not name:die('ROI_NAME_EMPTY')
        try:
            xyz=np.array([float(record[z]) for z in ['x_raw','y_raw','z_raw']]);radius=float(record['radius_raw'])
        except Exception:die('ROI_COORDINATES_NOT_NUMERIC_'+name)
        if not np.isfinite(xyz).all() or not 0<radius<10:die('INVALID_ROI_RADIUS_OR_CENTER_'+name)
        sel=(mrk==1)&(np.linalg.norm(cent-xyz[None,:],axis=1)<=radius)
        if not sel.any():die('ROI_WITHOUT_MATCHING_WALL_FACES_'+name)
        # A band value applies to the *whole* geodesic band; ROI is a conditional redistribution,
        # not individually calculated surface transport on each triangle.
        aw=area[sel]/area[sel].sum();unique,grp=np.unique(band_index[sel],return_inverse=True)
        weights=np.bincount(grp,weights=aw)
        for comp,access in [('EC',ec_acc),('VSMC',smc_acc)]:
            for cargo in ('p1','tbk1i','mir4651'):
                # Area-matched average per-wall-area modeled access index.
                frac=access[cargo][:,unique]/np.array([area[band_index==u].sum() for u in unique])[None,:]
                val=np.sum(frac*weights[None,:],axis=1)
                p=np.percentile(val,[5,50,95]);rows.append({'roi':name,'compartment':comp,'module':cargo,
                    'mean_surface_access_index_p05':float(p[0]),'median':float(p[1]),'p95':float(p[2]),
                    'n_wall_faces':int(sel.sum()),'source_evidence':record['annotation_evidence'],
                    'is_actual_concentration':False})
    return rows

def plot(out,pts,tri,mrk,cent,ar,dist,band,tab,p,cap,ec,smc,eca,sca,lumen,resid,dpi=600):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':23,'axes.titlesize':23,
      'axes.labelsize':23,'xtick.labelsize':20,'ytick.labelsize':20,'legend.fontsize':17,
      'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white',
      'savefig.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42})
    fig=plt.figure(figsize=(27,18),facecolor='white',layout='constrained')
    axes=[]
    for j in range(6):
        ax=fig.add_subplot(2,3,j+1,projection='3d' if j==0 else None)
        axes.append(ax)
    aa=axes[0];wall=np.flatnonzero(mrk==1)
    rng=np.random.default_rng(7)
    inds=rng.choice(wall,size=min(6000,len(wall)),replace=False)
    m=dist[inds]
    norm=Normalize(vmin=float(np.min(dist[wall])),vmax=float(np.max(dist[wall])))
    mesh=Poly3DCollection(pts[tri[inds]],edgecolors='none',linewidth=0,alpha=.97)
    mesh.set_facecolor(cm.viridis(norm(m)))
    aa.add_collection3d(mesh)
    mm=pts[tri[inds]].reshape(-1,3)
    center=(np.max(mm,axis=0)+np.min(mm,axis=0))/2; span=float(np.max(np.ptp(mm,axis=0)))/2
    aa.set_xlim(center[0]-span,center[0]+span);aa.set_ylim(center[1]-span,center[1]+span);aa.set_zlim(center[2]-span,center[2]+span)
    aa.view_init(elev=15,azim=-55);aa.set_axis_off()
    aa.set_title('a  Patient surface geodesic (not flow)',loc='left',pad=12,fontweight='bold')
    x=np.array([r['geodesic_position_0to1'] for r in tab]); k=len(x)
    def q(t):return np.percentile(t,[5,50,95],axis=0)
    ax=axes[1];q5,q50,q95=q(cap*100)
    ax.fill_between(x,q5,q95,alpha=.2,color='#4C78A8');ax.plot(x,q50,lw=2.8,color='#4C78A8')
    ax.set(title='b  Conditional wall capture',xlabel='Geodesic band position (0–1)',ylabel='Input-dose allocation / band (%)')
    colors={'p1':'#333333','tbk1i':'#C66B05','mir4651':'#137EAD'}
    labels={'p1':'P1 (PRNP expression)','tbk1i':'TBK1 inhibitor','mir4651':'miR-4651 (RISC prerequisite)'}
    for a,store,title in [(axes[2],eca,'c  Endothelial cytosolic access'),(axes[3],sca,'d  VSMC cytosolic access')]:
        for cargo in ('p1','tbk1i','mir4651'):
            q5,q50,q95=q(store[cargo]*100)
            a.plot(x,q50,lw=2.7,color=colors[cargo],label=labels[cargo])
            a.fill_between(x,q5,q95,alpha=.12,color=colors[cargo])
        a.set(title=title,xlabel='Geodesic band position (0–1)',ylabel='Modeled input fraction / band (%)')
    axes[2].legend(loc='upper right',frameon=False,fontsize=14)
    ax=axes[4];idx=np.argsort(np.sum(sca['mir4651'],axis=1))[len(lumen)//2]
    vals=[lumen[idx]*100,ec[idx].sum()*100,smc[idx].sum()*100,resid[idx]*100]
    labs=['Outflow / unbound','EC cytosol','VSMC cytosol','Captured, not cytosolic']
    ax.barh(np.arange(4),vals,color=['#AEAEAE','#2A658D','#5E976D','#BB8B58'])
    ax.set_yticks(range(4),labs);ax.invert_yaxis();ax.set_xlim(0,100)
    ax.set(title='e  One-pass mass balance (example)',xlabel='Percentage of modeled input mass (%)')
    for i,v in enumerate(vals):ax.text(min(v+1,97),i,f'{v:.2f}%',va='center',fontsize=16)
    ax=axes[5];yy=sca['mir4651'].sum(axis=1)
    sens=[]
    for key,vec in p.items():
        rho=float(spearmanr(vec,yy).statistic)
        sens.append((key,rho))
    sens=sorted(sens,key=lambda t:abs(t[1]),reverse=True)[:7]
    ax.barh(np.arange(len(sens)),[r for _,r in sens],color=['#3A6685' if r>=0 else '#C47636' for _,r in sens]);ax.set_yticks(range(len(sens)),[s.replace('_',' ') for s,_ in sens],fontsize=15)
    ax.invert_yaxis();ax.set_xlim(-1,1);ax.axvline(0,color='#444444',lw=.8)
    ax.set(title='f  VSMC miR access sensitivity',xlabel='Spearman ρ (scenario assumptions)')
    fig.suptitle('HGPS triple-conjugate | conditional vascular delivery and cellular access',fontsize=26,fontweight='bold')
    fig.text(.06,.006,'PATIENT SURFACE ONLY · HYPOTHETICAL ONE-PASS TRANSPORT · NOT MEASURED UPTAKE OR TARGET ENGAGEMENT · NO VALIDATED CFD',fontsize=14)
    png=out/'FIG7_PATIENT_SURFACE_DELIVERY_CONDITIONAL_a-f_600dpi.png'
    pdf=out/'FIG7_PATIENT_SURFACE_DELIVERY_CONDITIONAL_a-f_VECTOR.pdf'
    fig.savefig(png,dpi=dpi,facecolor='white')
    fig.savefig(pdf,facecolor='white')
    plt.close(fig)
    return png,pdf

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--prefix',required=True,help='V10 Quality patient_11patch.1 prefix without extension')
    parser.add_argument('--out',required=True)
    parser.add_argument('--roi-csv',default='')
    parser.add_argument('--samples',type=int,default=1024)
    parser.add_argument('--seed',type=int,default=RNG_SEED)
    parser.add_argument('--no-figure',action='store_true')
    parser.add_argument('--dpi',type=int,default=600)
    a=parser.parse_args(argv)
    if a.dpi<100 or a.dpi>600:die('DPI_MUST_BE_100_TO_600')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):die('OUTPUT_DIRECTORY_MUST_BE_EMPTY_TO_PREVENT_OVERWRITE')
    prefix=Path(a.prefix).resolve()
    nfile=Path(str(prefix)+'.node');ffile=Path(str(prefix)+'.face')
    orig={str(nfile):digest(nfile),str(ffile):digest(ffile)}
    pts,tri,mrk,ar,cent,_,_=read_tetgen_surface(prefix)
    audit=maybe_verify_v10_source(prefix,mrk)
    dd,graphinfo=geodesic_from_port2(pts,tri,mrk)
    dtri=np.mean(dd[tri],axis=1)
    wall=(mrk==1)
    band,table=construct_bands(dtri,ar,wall,bins=24)
    par,cap,ec,smc,ecacc,smcacc,lumen,residual,masserr=simulate(table,a.samples,a.seed)
    def q(x):return np.percentile(x,[5,50,95],axis=0)
    band_rows=[]
    for j,row in enumerate(table):
        rec=dict(row)
        for key,vec in [('capture',cap),('ec_cyt',ec),('smc_cyt',smc)]+[(f'ec_{k}_target_access',v) for k,v in ecacc.items()]+[(f'smc_{k}_target_access',v) for k,v in smcacc.items()]:
            p5,p50,p95=q(vec[:,j]);rec[f'{key}_p05']=float(p5);rec[f'{key}_median']=float(p50);rec[f'{key}_p95']=float(p95)
        band_rows.append(rec)
    write_csv(out/'DELIVERY_BY_GEODESIC_BAND_SCENARIO.csv',band_rows)
    sample_rows=[]
    for i in range(a.samples):
        rec={'sample':i,'seed':a.seed,**{k:float(v[i]) for k,v in par.items()},
             'lumen_exit':float(lumen[i]),'ec_cytosol':float(ec[i].sum()),
             'smc_cytosol':float(smc[i].sum()),'retained_or_lost':float(residual[i]),
             'one_pass_mass_balance':float(lumen[i]+ec[i].sum()+smc[i].sum()+residual[i]),
             'ec_p1_target_access':float(ecacc['p1'][i].sum()),
             'ec_tbk1i_target_access':float(ecacc['tbk1i'][i].sum()),
             'ec_mir4651_risc_prerequisite':float(ecacc['mir4651'][i].sum()),
             'smc_p1_target_access':float(smcacc['p1'][i].sum()),
             'smc_tbk1i_target_access':float(smcacc['tbk1i'][i].sum()),
             'smc_mir4651_risc_prerequisite':float(smcacc['mir4651'][i].sum())}
        sample_rows.append(rec)
    write_csv(out/'HYPOTHETICAL_ONEPASS_PARAMETER_ENSEMBLE.csv',sample_rows)
    patch_rows=[]
    for i,name in MARKERS.items():
        idx=mrk==i
        patch_rows.append({'patch':name,'n_triangles':int(idx.sum()),'surface_area_raw2':float(ar[idx].sum()),
           'distance_from_assumed_port2_areaweighted_raw':float(np.average(dtri[idx],weights=ar[idx])),
           'not_organ_specific':True,'patient_lesion_label':'UNAVAILABLE'})
    write_csv(out/'PATIENT_PATCH_GEOMETRY_ONLY.csv',patch_rows)
    roi_rows=interpret_roi(a.roi_csv,pts,tri,mrk,cent,ar,band,cap,ecacc,smcacc)
    if roi_rows:write_csv(out/'MANUALLY_ANNOTATED_ROI_CONDITIONAL_ACCESS.csv',roi_rows)
    fig=('NOT_REQUESTED','NOT_REQUESTED') if a.no_figure else plot(out,pts,tri,mrk,cent,ar,dtri,band,table,par,cap,ec,smc,ecacc,smcacc,lumen,residual,dpi=a.dpi)
    original_ok=all(digest(Path(p))==hashval for p,hashval in orig.items())
    if not original_ok:die('SOURCE_SHA256_CHANGED')
    result={
        'version':VERSION,'source_prefix':str(prefix),'source_sha256':orig,
        'source_sha256_unchanged':original_ok,'surface_triangle_count':len(tri),
        'surface_wall_triangle_count':int(wall.sum()),'surface_markers':{name:int((mrk==i).sum()) for i,name in MARKERS.items()},
        'source_audit':audit,'surface_graph':graphinfo,
        'inlet_assumption':'port_2 (from prior CFD setup; NOT clinically verified)',
        'spatial_axis':'surface-geodesic bands only; NOT vessel centerline or advective particle trajectories',
        'n_geodesic_bands':len(table),'conditional_samples':a.samples,
        'max_onepass_mass_balance_abs_error':masserr,
        'parameter_status':'BROAD_HYPOTHETICAL_DISTRIBUTIONS_NOT_MEASURED_HGPS_CONJUGATE_PK',
        'cellular_access_status':'COMPARTMENT_ENTRY_AND_RELEASE_ASSUMPTIONS_NOT_TARGET_ENGAGEMENT',
        'mri_to_left_right_lesion_roi_mapping_verified':bool(roi_rows),
        'no_3d_cfd':True,'no_measured_local_flow':True,'no_measured_biodistribution':True,
        'no_patient_absolute_dose_to_tissue':True,'clinical_efficacy_not_established':True,
        'source_files_unchanged':True,
        'model_organ_specific_biodistribution':False,
        'final_gate':'PATIENT_SURFACE_GROUNDED_GEODESIC__HYPOTHETICAL_DELIVERY_ONLY',
        'png':str(fig[0]),'pdf':str(fig[1])
    }
    (out/'AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    # Summarize only model conditional accessible fractions, avoid false pharmacological inference.
    text=[f'VERSION={VERSION}',f'PATIENT_SURFACE_FACES={len(tri)}',
          'ASSUMED_INLET=port_2_UNVERIFIED', f'PATCHES={len(MARKERS)}',
          f'GEODESIC_BANDS={len(table)}', f'CONDITIONAL_SCENARIOS={a.samples}',
          f'MASS_BALANCE_MAX_ABS_ERROR={masserr:.3g}', f'PATIENT_LEFT_RIGHT_ICA_ROI_VERIFIED={bool(roi_rows)}',
          f'SOURCE_SHA256_UNCHANGED={original_ok}',
          'WSS=NOT_COMPUTED; VESSEL_FLOW=NOT_CALIBRATED; ABSOLUTE_DRUG_DOSE_IN_ROI=NOT_KNOWN',
          'PRNP_TBK1_SRSF1_TARGET_ENGAGEMENT=NOT_ESTABLISHED',
          'DELIVERY_FRACTIONS=HYPOTHETICAL_MODEL_ONEPASS_NOT_OBSERVED',
          'NO_CFD=True; NO_NEW_MESH=True',f'FINAL_GATE={result["final_gate"]}',f'OUTPUT={out}']
    for field in ('ec_p1_target_access','ec_tbk1i_target_access','ec_mir4651_risc_prerequisite','smc_p1_target_access','smc_tbk1i_target_access','smc_mir4651_risc_prerequisite'):
        x=np.array([r[field] for r in sample_rows]);p5,p50,p95=np.percentile(x,[5,50,95])
        text.append(f'MODEL_{field}_FRACTION_P05_MED_P95={p5:.6g},{p50:.6g},{p95:.6g}')
    (out/'SUMMARY.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('\n'.join(text),flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
