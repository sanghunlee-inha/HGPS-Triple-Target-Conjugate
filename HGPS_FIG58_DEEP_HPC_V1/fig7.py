"""Real unstructured 3D tetrahedral implicit finite-volume scalar convection/diffusion with wall uptake.

Scientific limitations: stationary imported velocity; TPFA diffusion is non-orthogonal-grid approximate;
NOT Navier-Stokes/FSI, NOT clinical efficacy, NOT patient-specific anatomy.
Requires documented physical units, cap identities, dosing boundary, diffusivity and permeability.
"""
from pathlib import Path
import os, json, math, gc
import numpy as np
from core import emit,jd,output_report,hold,digest

def config_check(conf):
    required=['coordinate_units','velocity_units','diffusivity_m2_s','wall_permeability_m_s',
        'inlet_patch_ids','outlet_patch_ids','wall_patch_ids', 'inlet_schedule_s_concentration',
        'time_end_s','dt_s','reference_geometry_acknowledged']
    missing=[x for x in required if x not in conf or conf[x] is None]
    if missing:raise ValueError('FIG7_MISSING_REQUIRED_CONFIG '+','.join(missing))
    if conf['coordinate_units']!='m' or conf['velocity_units']!='m/s':
        raise ValueError('FIG7_PHYSICAL_UNITS_MUST_BE_VERIFIED_m_AND_m_per_s')
    if not conf['reference_geometry_acknowledged']:
        raise ValueError('FIG7_REFERENCE_GEOMETRY_NOT_ACKNOWLEDGED')
    d=float(conf['diffusivity_m2_s']);k=float(conf['wall_permeability_m_s'])
    if not (np.isfinite(d) and d>0 and np.isfinite(k) and k>=0):raise ValueError('FIG7_INVALID_D_OR_K')
    if float(conf['dt_s'])<=0 or float(conf['time_end_s'])<=0:raise ValueError('FIG7_BAD_TIME')
    ids=[list(map(int,conf[x])) for x in ('inlet_patch_ids','outlet_patch_ids','wall_patch_ids')]
    if not all(ids):raise ValueError('FIG7_EMPTY_PATCH_GROUP')
    ss=[set(x) for x in ids]
    if ss[0]&ss[1] or ss[0]&ss[2] or ss[1]&ss[2]:raise ValueError('FIG7_OVERLAPPING_PATCH_GROUPS')
    sched=np.array(conf['inlet_schedule_s_concentration'],dtype=float)
    if sched.ndim!=2 or sched.shape[1]!=2 or len(sched)<2 or np.any(np.diff(sched[:,0])<=0) or np.any(sched[:,1]<0):
        raise ValueError('FIG7_INVALID_INLET_CONCENTRATION_SCHEDULE')
    if sched[0,0]>0 or sched[-1,0]<conf['time_end_s']:raise ValueError('FIG7_SCHEDULE_DOES_NOT_SPAN_TIME')
    return d,k,ids,sched

def require_metadata(conf,allow_exploratory):
    for field in ['mesh_quality_evidence_file','velocity_boundary_condition_evidence_file',
                  'transport_parameter_source_evidence_file']:
        f=conf.get(field)
        if not f and not allow_exploratory:raise ValueError('FIG7_METADATA_HOLD_'+field)
        if f and not Path(f).is_file():raise ValueError('FIG7_EVIDENCE_FILE_MISSING_'+field)
    return 'EXPLORATORY_ASSUMPTIONS_NOT_VALIDATED' if allow_exploratory else 'PHYSICS_INPUTS_DOCUMENTED_NOT_BIOLOGICALLY_VALIDATED'

def face_index(tets):
    """Sort all 4 face keys and pair occurrences; do not use a mesh graph proxy."""
    n=len(tets)
    face=np.concatenate((tets[:,[1,2,3]],tets[:,[0,2,3]],tets[:,[0,1,3]],tets[:,[0,1,2]]))
    face.sort(axis=1)
    emit('FIG7_FACE_KEYS_CREATED',len(face))
    order=np.lexsort((face[:,2],face[:,1],face[:,0]))
    sorted_keys=face[order]
    starts=np.r_[0,np.flatnonzero(np.any(sorted_keys[1:]!=sorted_keys[:-1],axis=1))+1]
    counts=np.diff(np.r_[starts,len(order)])
    if int(np.max(counts))!=2 or np.any(counts>2):raise ValueError('FIG7_NONMANIFOLD_FACE_COUNTS')
    owner=(order[starts]%n).astype(np.int32)
    inner=(counts==2)
    second=(order[starts[inner]+1]%n).astype(np.int32)
    a=owner[inner]; b=second
    tri_inner=sorted_keys[starts[inner]]
    boundary=~inner
    tri_bound=sorted_keys[starts[boundary]]
    cells_bound=owner[boundary]
    del face,sorted_keys,order,starts,counts,owner
    gc.collect()
    emit('FIG7_FACE_PAIRS',len(a),'BOUNDARY',len(cells_bound))
    return a,b,tri_inner,cells_bound,tri_bound

def _geometry(points,tets,tri,a,b,chunk=500000):
    """cell centroids, exact tetra volumes and interior center-to-face distances."""
    cent=np.empty((len(tets),3),dtype=np.float64)
    vols=np.empty(len(tets),dtype=np.float64)
    for st in range(0,len(tets),chunk):
        en=min(len(tets),st+chunk)
        x=points[tets[st:en]].astype(np.float64,copy=False)
        cent[st:en]=x.mean(axis=1)
        vols[st:en]=np.abs(np.einsum('ij,ij->i',np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]),x[:,3]-x[:,0]))/6
    if np.any(vols<=0) or not np.isfinite(vols).all():raise ValueError('FIG7_INVALID_TET_VOLUME')
    return cent,vols

def face_geometry(points,tris,left,right,centers,velocity,chunk=500000):
    n=len(left)
    area=np.empty(n,dtype=float);flux=np.empty(n,dtype=float)
    distance=np.empty(n,dtype=float);skew=np.empty(n,dtype=float)
    for st in range(0,n,chunk):
        en=min(n,st+chunk);tri=tris[st:en]
        x=points[tri].astype(np.float64,copy=False)
        normal=np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]);ar=np.linalg.norm(normal,axis=1)/2
        displacement=centers[right[st:en]]-centers[left[st:en]]
        side=np.sign(np.einsum('ij,ij->i',normal,displacement));side[side==0]=0
        unit=normal/np.maximum(2*ar[:,None],1e-300)*side[:,None]
        delta=np.einsum('ij,ij->i',displacement,unit)
        v=np.mean(velocity[tri],axis=1,dtype=np.float64)
        flux[st:en]=np.einsum('ij,ij->i',v,unit)*ar
        distance[st:en]=delta
        area[st:en]=ar
        skew[st:en]=delta/np.maximum(np.linalg.norm(displacement,axis=1),1e-300)
    if np.any(distance<=0) or np.any(area<=0) or not np.isfinite(flux).all():raise ValueError('FIG7_BAD_INTERNAL_FACE_GEOMETRY')
    return area,flux,distance,skew

def boundary_geometry(points,tris,cells,centers,velocity,chunk=500000):
    n=len(cells);area=np.empty(n);flux=np.empty(n);half_dist=np.empty(n)
    for st in range(0,n,chunk):
        en=min(n,st+chunk);tri=tris[st:en]
        x=points[tri].astype(float,copy=False)
        norm=np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]);a=np.linalg.norm(norm,axis=1)/2
        dx=x.mean(axis=1)-centers[cells[st:en]]
        sign=np.sign(np.einsum('ij,ij->i',norm,dx))
        if np.any(sign==0):raise ValueError('FIG7_ZERO_FACE_ORIENTATION')
        unit=norm/np.maximum(2*a[:,None],1e-300)*sign[:,None]
        flux[st:en]=np.einsum('ij,ij->i',velocity[tri].mean(axis=1),unit)*a
        half_dist[st:en]=np.einsum('ij,ij->i',dx,unit)
        area[st:en]=a
    if np.any(half_dist<=0) or np.any(area<=0):raise ValueError('FIG7_BAD_BOUNDARY_GEOMETRY')
    return area,flux,half_dist

def labels_from_reference(ref,btri,points,tol_m=2e-7):
    """3D boundary face centroids must coincide with actual reference triangular faces.
    A nearest centroid only accepted inside a tight geometric tolerance; never guess labels."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from scipy.spatial import cKDTree
    r=vtk.vtkXMLPolyDataReader();r.SetFileName(str(ref));r.Update();surf=r.GetOutput()
    if surf.GetNumberOfCells()==0:raise ValueError('FIG7_REF_EMPTY')
    conn=vtk_to_numpy(surf.GetPolys().GetConnectivityArray()).reshape(-1,3)
    sp=vtk_to_numpy(surf.GetPoints().GetData())
    faceid=surf.GetCellData().GetArray('ModelFaceID')
    if faceid is None:raise ValueError('FIG7_MODEL_FACE_ID_MISSING')
    ids=vtk_to_numpy(faceid)
    tree=cKDTree(sp[conn].mean(axis=1))
    labels=np.empty(len(btri),dtype=np.int16);md=0.0
    for st in range(0,len(btri),125000):
        en=min(len(btri),st+125000)
        dist,ix=tree.query(points[btri[st:en]].mean(axis=1),k=1,workers=8)
        md=max(md,float(dist.max()))
        if np.any(dist>tol_m):raise ValueError('FIG7_BOUNDARY_FACE_NOT_MATCHED_TO_REFERENCE count='+str(int(np.sum(dist>tol_m)))+' max_dist='+str(float(dist.max())))
        labels[st:en]=ids[ix]
    emit('FIG7_BOUNDARY_MATCH_PASS',len(btri),'max_distance_m',md)
    return labels,md

def _csr_operator(vol,dt,left,right,f,k,bcells,bflux,barea,bdist,bkind,D,kw):
    from scipy.sparse import coo_matrix,diags
    n=len(vol)
    coeff_k=D*k
    # each unordered internal face: conservative upwind + symmetric two point diffusion
    diag=np.bincount(left,weights=coeff_k+np.maximum(f,0),minlength=n)+np.bincount(right,weights=coeff_k+np.maximum(-f,0),minlength=n)
    other_lr=-coeff_k+np.minimum(f,0)
    other_rl=-coeff_k-np.maximum(f,0)
    # boundary kind 1 inlet, 2 outlet, 3 wall
    inlet=(bkind==1);outlet=(bkind==2);wall=(bkind==3)
    if kw>0: wall_transfer=barea[wall]/(bdist[wall]/D+1/kw)
    else:wall_transfer=np.zeros(int(wall.sum()))
    diag+=np.bincount(bcells[outlet],weights=np.maximum(bflux[outlet],0),minlength=n)
    diag+=np.bincount(bcells[inlet],weights=np.maximum(bflux[inlet],0)+D*barea[inlet]/bdist[inlet],minlength=n)
    diag+=np.bincount(bcells[wall],weights=wall_transfer,minlength=n)
    dia=vol/dt+diag
    idx=np.arange(n,dtype=np.int32)
    M=coo_matrix((np.r_[dia,other_lr,other_rl],
                  (np.r_[idx,left,right],np.r_[idx,right,left])),shape=(n,n)).tocsr()
    incoming=np.bincount(bcells[inlet],weights=-np.minimum(bflux[inlet],0)+D*barea[inlet]/bdist[inlet],minlength=n)
    wall_transfer_by_cell=np.bincount(bcells[wall],weights=wall_transfer,minlength=n)
    return M,incoming,wall_transfer_by_cell,wall_transfer,diag

def run(conf_path,source_vtu,source_ref,out,allow_exploratory=False):
    from scipy.sparse.linalg import bicgstab,LinearOperator
    from vtk.util.numpy_support import vtk_to_numpy
    from fig6 import _vtk_read,_tetra_arrays
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not conf_path or not Path(conf_path).is_file():return hold(out,'FIG7','CONFIG_REQUIRED_FOR_PHYSICAL_3D_TRANSPORT')
    if not source_vtu or not source_ref:return hold(out,'FIG7','VTU_AND_REFERENCE_REQUIRED')
    conf=json.loads(Path(conf_path).read_text(encoding='utf-8-sig'))
    D,kw,groups,schedule=config_check(conf);scope=require_metadata(conf,allow_exploratory)
    maxiter=int(conf.get('max_linear_iterations',1000))
    tolerance=float(conf.get('linear_rtol',1.e-8))
    max_nonorth=float(conf.get('max_internal_nonorth_deg',70.0))
    label_tolerance=float(conf.get('reference_face_match_tolerance_m',2e-7))
    if tolerance<=0 or maxiter<1:raise ValueError('FIG7_BAD_LINEAR_SOLVER_SETTINGS')
    emit('FIG7_LOADING_REAL_TETRA_CFD',source_vtu)
    g=_vtk_read(source_vtu)
    points,tets=_tetra_arrays(g)
    v=g.GetPointData().GetArray('Velocity')
    if v is None:raise ValueError('FIG7_VELOCITY_MISSING')
    velo=vtk_to_numpy(v).astype(np.float64,copy=False)
    if velo.shape!=(len(points),3) or not np.isfinite(velo).all():raise ValueError('FIG7_VELOCITY_INVALID')
    n=len(tets)
    # Cell count and memory gate prevent irreversible OOM; no automatic downsampling.
    min_ram_gb=float(conf.get('min_free_memory_gb',28.0))
    try:
        import psutil
        available=psutil.virtual_memory().available/1e9
        if available<min_ram_gb:raise ValueError(f'FIG7_INSUFFICIENT_FREE_MEMORY_{available:.2f}_GB_REQUIRE_{min_ram_gb}_GB')
    except ImportError:emit('FIG7_RAM_PREFLIGHT_UNAVAILABLE_INSTALL_PSUTIL')
    emit('FIG7_PREPROCESS_FULL_MESH',n)
    centers,vol=_geometry(points,tets,None,None,None)
    left,right,iface,bcell,bface=face_index(tets)
    del tets
    ia,flux,dist,skew=face_geometry(points,iface,left,right,centers,velo)
    del iface
    mincos=float(skew.min())
    maxangle=float(np.degrees(np.arccos(np.clip(mincos,-1,1))))
    if maxangle>max_nonorth:raise ValueError('FIG7_NONORTHOGONAL_EXCEEDS_CONFIGURED_LIMIT_DEG='+str(maxangle))
    ba,bflux,bdist=boundary_geometry(points,bface,bcell,centers,velo)
    labels,maxmatch=labels_from_reference(source_ref,bface,points,label_tolerance)
    present=set(map(int,np.unique(labels)));allids=set(groups[0]+groups[1]+groups[2])
    if present!=allids:raise ValueError('FIG7_PATCH_LABELS_NOT_EXHAUSTIVE '+str((present,allids)))
    kind=np.zeros(len(labels),dtype=np.uint8)
    kind[np.isin(labels,groups[0])]=1
    kind[np.isin(labels,groups[1])]=2
    kind[np.isin(labels,groups[2])]=3
    wall=kind==3
    wall_flow_abs=float(np.abs(bflux[wall]).sum());open_flow_abs=float(np.abs(bflux[~wall]).sum())
    max_wall_leakage_fraction=float(conf.get('max_wall_leakage_fraction',.005))
    if wall_flow_abs>max_wall_leakage_fraction*max(open_flow_abs,1e-300):
        raise ValueError('FIG7_WALL_NORMAL_FLOW_EXCEEDS_LIMIT ratio='+str(wall_flow_abs/max(open_flow_abs,1e-300)))
    in_flux=float(-np.minimum(bflux[kind==1],0).sum())
    out_flux=float(np.maximum(bflux[kind==2],0).sum())
    if in_flux<=0 or out_flux<=0:raise ValueError('FIG7_INLET_OR_OUTLET_FLUX_NONPOSITIVE')
    flux_imbalance=abs(in_flux-out_flux)/max(in_flux,out_flux)
    backflow=float((-np.minimum(bflux[kind==2],0)).sum())
    if backflow>float(conf.get('max_outlet_backflow_fraction',.002))*out_flux:
        raise ValueError('FIG7_OUTLET_BACKFLOW_UNMODELED_'+str(backflow/out_flux))
    if flux_imbalance>float(conf.get('max_volume_flux_imbalance',.02)):
        raise ValueError('FIG7_INLET_OUTLET_VOLUME_FLUX_IMBALANCE '+str(flux_imbalance))
    dt=float(conf['dt_s']);end=float(conf['time_end_s']);nsteps=int(math.ceil(end/dt))
    if nsteps>int(conf.get('max_steps',20000)):raise ValueError('FIG7_TOO_MANY_TIME_STEPS')
    emit('FIG7_BUILDING_GLOBAL_CONSERVATIVE_SPARSE_OPERATOR',len(vol),'cells',len(flux),'interior_faces')
    M,inlet_rhs,wall_coeff_by_cell,wall_coeff,diag_extra=_csr_operator(vol,dt,left,right,flux,ia/dist,
                     bcell,bflux,ba,bdist,kind,D,kw)
    del ia,dist,skew,left,right,flux,diag_extra
    gc.collect()
    c=np.zeros(len(vol),dtype=np.float64)
    diag=M.diagonal();P=LinearOperator(M.shape,matvec=lambda x: x/diag)
    report=[];cum=0.0
    for step in range(1,nsteps+1):
        t=min(step*dt,end);concin=float(np.interp(t,schedule[:,0],schedule[:,1]))
        rhs=vol/dt*c+concin*inlet_rhs
        new,info=bicgstab(M,rhs,x0=c,rtol=tolerance,atol=0.0,maxiter=maxiter,M=P)
        if info!=0:raise ValueError('FIG7_LINEAR_SOLVER_NOT_CONVERGED_STEP_'+str(step)+'_INFO_'+str(info))
        if not np.isfinite(new).all() or new.min()<-1e-7*max(concin,1):
            raise ValueError('FIG7_NONPHYSICAL_CONCENTRATION_STEP_'+str(step))
        residual=float(np.linalg.norm(M@new-rhs)/max(np.linalg.norm(rhs),1e-300))
        if residual>max(20*tolerance,1e-9):raise ValueError('FIG7_LINEAR_RESIDUAL_FAILED_'+str(residual))
        mass_old=float(vol@c);mass_new=float(vol@new)
        wall_loss=float(wall_coeff_by_cell@new)
        is_in=kind==1;is_out=kind==2
        inlet_adv_rate=float(concin*(-np.minimum(bflux[is_in],0)).sum())
        inlet_diff_rate=float(np.sum(D*ba[is_in]/bdist[is_in]*(concin-new[bcell[is_in]])))
        outlet_adv_rate=float(np.dot(np.maximum(bflux[is_out],0),new[bcell[is_out]]))
        inlet_backflow_rate=float(np.dot(np.maximum(bflux[is_in],0),new[bcell[is_in]]))
        # Independent face-by-face boundary budget, NOT a tautological sparse-matrix check.
        dmassdt=(mass_new-mass_old)/dt
        net_boundary=inlet_adv_rate+inlet_diff_rate-outlet_adv_rate-inlet_backflow_rate-wall_loss
        mb_err=abs(dmassdt-net_boundary)
        mb_scale=max(abs(inlet_adv_rate)+abs(inlet_diff_rate)+abs(outlet_adv_rate)+abs(inlet_backflow_rate)+abs(wall_loss),abs(dmassdt),1e-14)
        rel_balance=mb_err/mb_scale
        if rel_balance>max(1e-5,20*tolerance):
            raise ValueError('FIG7_INDEPENDENT_FACE_MASS_BUDGET_FAILED_STEP_'+str(step)+'_REL_'+str(rel_balance))
        cum+=wall_loss*dt
        report.append({'step':step,'time_s':t,'inlet_concentration':concin,'mass_native_concentration_times_volume':mass_new,
           'c_min':float(new.min()),'c_max':float(new.max()),'wall_uptake_rate':wall_loss,
           'cumulative_wall_uptake':cum,'relative_linear_residual':residual,'relative_mass_budget_error':rel_balance,
           'inlet_advective_input':inlet_adv_rate,'inlet_diffusive_input':inlet_diff_rate,
           'outlet_advective_loss':outlet_adv_rate,'wall_uptake_loss':wall_loss})
        c=new
        if step==1 or step%max(1,int(conf.get('progress_every_steps',10)))==0 or step==nsteps:
            emit('FIG7_PDE_SOLVE',step,'/',nsteps,'mass',mass_new,'wall_cumulative',cum,'balance',rel_balance)
        if step in {1,nsteps} or (conf.get('save_every_steps',0) and step%int(conf['save_every_steps'])==0):
            np.save(out/f'FIG7_3D_CELL_CONCENTRATION_STEP_{step:06d}.npy',c.astype(np.float32))
    import pandas as pd
    pd.DataFrame(report).to_csv(out/'FIG7_3D_TRANSPORT_MASS_BUDGET.csv',index=False)
    summary={'status':'NUMERICAL_3D_TRANSPORT_COMPLETED_'+scope,
       'source_vtu':str(source_vtu),'source_vtu_sha256':digest(source_vtu),
       'source_reference':str(source_ref),'source_reference_sha256':digest(source_ref),
       'config_path':str(conf_path),'config_sha256':digest(conf_path),
       'method':'cell-centered tetrahedron finite volume, implicit Euler, upwind face advection, two-point face diffusion, Robin wall',
       'limitations':['stationary imported flow field; NOT a flow solver','two-point diffusion on nonorthogonal tetrahedra has nonorthogonality error',
                      'adult reference geometry; not patient-specific','permeability is a model input, not inferred from scRNA','no treatment-vs-control hemodynamic efficacy'],
       'max_nonorth_degrees':maxangle,'volume_flux_imbalance':flux_imbalance,
       'max_boundary_centroid_match_m':maxmatch,'steps':nsteps,'tetra_cells':len(vol),'mass_budget_final':report[-1],
       'physical_parameter_provenance':scope,'units':'m, m/s, m2/s, m/s, seconds; concentration arbitrary specified units'}
    jd(out/'FIG7_3D_TRANSPORT_VERIFIED_NUMERICS.json',summary)
    return output_report(out,'FIG7','3D_CONVECTION_DIFFUSION_DONE_REFERENCE_GEOMETRY',
      solver='tetraFVM',scope=scope,n_tetra=len(vol),steps=nsteps)
