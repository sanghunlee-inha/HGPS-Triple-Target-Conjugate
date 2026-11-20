"""Read EVERY tetrahedron in archived V150 3D fields; no CFD rerun or treatment assertion."""
from pathlib import Path
import numpy as np
import pandas as pd
from core import emit, digest, jd, hold, output_report, REF_SHA, WSS_KNOWN_SHA

def _vtk_read(path):
    import vtk
    ext=Path(path).suffix.lower()
    reader=(vtk.vtkXMLUnstructuredGridReader() if ext=='.vtu' else vtk.vtkXMLPolyDataReader())
    reader.SetFileName(str(path));reader.Update()
    g=reader.GetOutput()
    if g is None or not g.GetNumberOfCells():raise ValueError('EMPTY_VTK '+str(path))
    return g

def _tetra_arrays(g):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    types=vtk_to_numpy(g.GetCellTypesArray())
    if not np.all(types==vtk.VTK_TETRA):
        u,c=np.unique(types,return_counts=True)
        raise ValueError('NOT_ALL_TETRA_CELL_TYPES '+str(dict(zip(u.tolist(),c.tolist()))))
    conn=vtk_to_numpy(g.GetCells().GetConnectivityArray())
    offsets=vtk_to_numpy(g.GetCells().GetOffsetsArray())
    if not np.all(np.diff(offsets)==4):raise ValueError('NON_TETRA_OFFSETS')
    return vtk_to_numpy(g.GetPoints().GetData()),conn.reshape(-1,4)

def _series_stats(vals,weights,kind):
    vals=np.asarray(vals,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    mask=np.isfinite(vals)&np.isfinite(weights)&(weights>0)
    if not mask.all():raise ValueError('NONFINITE_OR_INVALID '+kind+' count='+str((~mask).sum()))
    from core import wquantile
    return dict(min=float(vals.min()),max=float(vals.max()),mean_volume_weighted=float(np.average(vals,weights=weights)),
       q05_q50_q95_volume_weighted=wquantile(vals,weights,[.05,.5,.95]),
       sum_field_times_volume=float(np.dot(vals,weights)))

def run(snapshots,out,reference=None,wss=None,chunk=125000):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not snapshots:return hold(out,'FIG6','NO_ARCHIVED_V150_VTU_INPUT')
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    results=[]; volumes_cache=None; sig=None
    for si,p in enumerate(snapshots):
        p=Path(p)
        if not p.is_file():return hold(out,'FIG6','VTU_MISSING',{'path':str(p)})
        emit('FIG6_VTU_LOADING',str(p),'bytes',p.stat().st_size)
        grid=_vtk_read(p)
        pts,tets=_tetra_arrays(grid)
        n=len(tets);emit('FIG6_TETRA',n,'points',len(pts),'source',str(p))
        vfield=grid.GetPointData().GetArray('Velocity')
        pfield=grid.GetPointData().GetArray('Pressure')
        if vfield is None or pfield is None:raise ValueError('MISSING_VELOCITY_OR_PRESSURE '+str(p))
        velocity=vtk_to_numpy(vfield)
        pressure=vtk_to_numpy(pfield)
        if velocity.shape!=(len(pts),3) or pressure.shape[0]!=len(pts):raise ValueError('FIELD_SHAPE_INVALID')
        div_field=grid.GetPointData().GetArray('Divergence')
        div=vtk_to_numpy(div_field) if div_field is not None else None
        vmag=np.empty(n,dtype=np.float64);pres=np.empty(n,dtype=np.float64)
        divcell=np.empty(n,dtype=np.float64) if div is not None else None
        vol=np.empty(n,dtype=np.float64)
        invalid=0
        for start in range(0,n,chunk):
            stop=min(start+chunk,n);ids=tets[start:stop]
            x=pts[ids].astype(np.float64,copy=False)
            e1=x[:,1]-x[:,0];e2=x[:,2]-x[:,0];e3=x[:,3]-x[:,0]
            v=np.abs(np.einsum('ij,ij->i',np.cross(e1,e2),e3))/6.0
            invalid+=int(np.sum((~np.isfinite(v))|(v<=0)))
            vol[start:stop]=v
            mean_u=np.mean(velocity[ids],axis=1,dtype=np.float64)
            vmag[start:stop]=np.linalg.norm(mean_u,axis=1)
            pres[start:stop]=np.mean(pressure[ids],axis=1,dtype=np.float64)
            if div is not None:divcell[start:stop]=np.mean(div[ids],axis=1,dtype=np.float64)
            if start//chunk % 10==0:emit('FIG6_VOLUME_PROGRESS',si+1,'/',len(snapshots),stop,'/',n)
        if invalid:raise ValueError(f'VOLUME_NONPOSITIVE_OR_NONFINITE_N={invalid}')
        if volumes_cache is not None:
            if n!=len(volumes_cache) or not np.allclose(vol,volumes_cache,rtol=2e-5,atol=1e-16):
                raise ValueError('SNAPSHOT_MESH_CHANGED; DO_NOT_COMPARE_AS_SAME_MESH')
        else:volumes_cache=vol
        info={'file':str(p),'sha256':digest(p),'point_count':len(pts),'tetra_count':n,
              'total_native_volume':float(vol.sum()),'velocity':_series_stats(vmag,vol,'velocity'),
              'pressure':_series_stats(pres,vol,'pressure'),
              'derived_from':'tetrahedron mean of point fields, weighted by exact tetra volume',
              'field_units':'UNVERIFIED_SOURCE_NATIVE_UNITS','is_treated':False}
        if divcell is not None:
            info['divergence']={'volume_integral_signed':float(np.dot(divcell,vol)),
               'volume_mean_absolute':float(np.average(np.abs(divcell),weights=vol)),
               'volume_mean_signed':float(np.average(divcell,weights=vol)),
               'basis':'original saved Divergence field, not an independent flux-conservation test'}
        results.append(info)
        jd(out/f'FIG6_SNAPSHOT_{si+1}_FULL_VOLUME_AUDIT.json',info)
        del grid,pts,tets,velocity,pressure,vmag,pres,divcell,vol
    pd.DataFrame([{'file':r['file'],'tetra':r['tetra_count'],'volume':r['total_native_volume'],
         'velocity_volume_mean':r['velocity']['mean_volume_weighted'],
         'pressure_volume_mean':r['pressure']['mean_volume_weighted']} for r in results]).to_csv(out/'FIG6_V150_FULL_VOLUME_TIMELINE.csv',index=False,encoding='utf-8-sig')
    wss_report=None
    if wss:
        wss_report=wall_area_analysis(wss,out)
    if reference:
        rr=digest(reference)
        jd(out/'FIG6_REFERENCE_GEOMETRY_GATE.json',{'reference':str(reference),'sha256':rr,'matches_reference':rr==REF_SHA})
        if rr!=REF_SHA:raise ValueError('REFERENCE_SURFACE_SHA_MISMATCH')
    jd(out/'FIG6_EXISTING_3D_FULL_DATA_REPORT.json',{'snapshots':results,'wall':wss_report,
       'scope':'ARCHIVED_3D_REFERENCE_FIELDS_NOT_TREATED_CFD',
       'cannot_claim':['patient-specific vessel','treatment response','validated SI units','FSI',
                       'mass conservation through inlets/outlets without solver BC metadata']})
    return output_report(out,'FIG6','FULL_ARCHIVED_3D_FIELD_POSTPROCESSING_NOT_TREATMENT_CFD',
             snapshots=len(results),tetra_per_snapshot=results[0]['tetra_count'])

def wall_area_analysis(path,out):
    from vtk.util.numpy_support import vtk_to_numpy
    p=Path(path);sha=digest(p)
    emit('FIG6_WALL_WSS_READ',p)
    g=_vtk_read(p);pts=vtk_to_numpy(g.GetPoints().GetData())
    conn=vtk_to_numpy(g.GetPolys().GetConnectivityArray()).reshape(-1,3)
    area=np.empty(len(conn),dtype=np.float64)
    x=pts[conn].astype(np.float64,copy=False)
    area[:]=np.linalg.norm(np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]),axis=1)/2
    if not np.all((area>0)&np.isfinite(area)):raise ValueError('WALL_NONPOSITIVE_AREA')
    raw=g.GetPointData().GetArray('WSS_Magnitude_Native')
    if raw is None:raise ValueError('WALL_WSS_MAGNITUDE_FIELD_MISSING')
    wss=vtk_to_numpy(raw)[conn].mean(axis=1)
    if not np.isfinite(wss).all():raise ValueError('WALL_WSS_NONFINITE')
    report={'path':str(p),'sha256':sha,'reference_sha_match':sha==WSS_KNOWN_SHA,
       'triangles':len(conn),'surface_area_native':float(area.sum()),
       'wss_native_magnitude':_series_stats(wss,area,'wall_wss'),
       'definition':'triangle area weighted average of 3 nodal WSS_Magnitude_Native values',
       'units':'UNVERIFIED_NATIVE', 'not_treated_wss':True}
    jd(Path(out)/'FIG6_WALL_WSS_AREA_FULL.json',report)
    return report
