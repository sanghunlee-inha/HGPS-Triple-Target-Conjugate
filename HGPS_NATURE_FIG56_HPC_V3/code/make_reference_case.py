#!/usr/bin/env python3
"""Construct an OpenFOAM14 steady rigid-wall reference case from supplied VERIFIED physical inputs.
No fictional patient flow, viscosity, drug effect, inlet/outlet pressures, FSI, or unit conversion.
Requires a complete, source-traceable manifest. Only reference fluid hemodynamics, NOT treatment recovery.
"""
import argparse, json, math, os, re, sys, hashlib
from pathlib import Path

def valid_source(p):
    path=Path(p)
    if not path.is_file() or path.stat().st_size<1:raise ValueError('Missing actual physical input source: '+str(p))
    return hashlib.sha256(path.read_bytes()).hexdigest()

def foam_header(obj,location,cls='dictionary'):
    return f'FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {cls};\n    location "{location}";\n    object {obj};\n}}\n'

def ensure_pos(x,name):
    if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) or x<=0:
        raise ValueError(f'HOLD: invalid {name}: {x}')
    return float(x)

def main():
    p=argparse.ArgumentParser();p.add_argument('--physics',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.physics).read_text(encoding='utf-8-sig'))
    if cfg.get('case_type')!='rigid_wall_steady_reference_only' or cfg.get('approved_input_sources') is not True:
        raise RuntimeError('HOLD: explicit reference-only model and source approval required')
    provenance=cfg.get('provenance',{})
    if not isinstance(provenance,dict) or not all(k in provenance for k in ('mra_coordinate_units','flow_boundary_measurement','blood_fluid_properties')):
        raise RuntimeError('HOLD: missing units, flow boundary and fluid source documents')
    digests={k:valid_source(v) for k,v in provenance.items()}
    scalar=ensure_pos(cfg.get('coordinate_unit_to_m'),'coordinate_unit_to_m')
    nu=ensure_pos(cfg.get('kinematic_viscosity_m2_s'),'kinematic_viscosity_m2_s')
    rho=ensure_pos(cfg.get('density_kg_m3'),'density_kg_m3')
    write_interval=int(ensure_pos(cfg.get('write_interval_steps'),'write_interval_steps'))
    iterations=int(ensure_pos(cfg.get('max_steady_iterations'),'max_steady_iterations'))
    inlet=cfg.get('inlet_flows_m3_s',{});outlet=cfg.get('outlet_kinematic_pressures_m2_s2',{})
    if not isinstance(inlet,dict) or not isinstance(outlet,dict) or not inlet or not outlet:raise ValueError('HOLD: measured inlet and outlet maps required')
    ports={f'port_{i}' for i in range(2,12)}
    if set(inlet)&set(outlet) or set(inlet)|set(outlet)!=ports:raise ValueError('HOLD: each of 10 ports must be explicitly inlet or outlet')
    if set(cfg.get('preserved_surface_patch_names',[]))!=ports|{'wall'}:raise ValueError('HOLD: names must match 11 original groups')
    for k,v in inlet.items():ensure_pos(v,'inlet '+k)
    for k,v in outlet.items():
        if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v):raise ValueError('HOLD: invalid outlet kinematic pressure '+k)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for n in ('0','constant','system'): (out/n).mkdir(exist_ok=True)
    patchU='';patchP=''
    for patch in ['wall']+sorted(ports,key=lambda x:int(x.split('_')[1])):
        if patch=='wall':
            U='type fixedValue; value uniform (0 0 0);';P='type zeroGradient;'
        elif patch in inlet:
            U='type flowRateInletVelocity; volumetricFlowRate constant %.17g; value uniform (0 0 0);'%inlet[patch]
            P='type zeroGradient;'
        else:
            U='type pressureInletOutletVelocity; value uniform (0 0 0);'
            P='type fixedValue; value uniform %.17g;'%outlet[patch]
        patchU+=f'  {patch}\n  {{ {U} }}\n';patchP+=f'  {patch}\n  {{ {P} }}\n'
    (out/'0/U').write_text(foam_header('U','0','volVectorField')+
      'dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\nboundaryField\n{\n'+patchU+'}\n',encoding='ascii')
    (out/'0/p').write_text(foam_header('p','0','volScalarField')+
      'dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n'+patchP+'}\n',encoding='ascii')
    (out/'constant/physicalProperties').write_text(foam_header('physicalProperties','constant')+
      'viscosityModel constant;\nnu %.17g;\n'%nu,encoding='ascii')
    (out/'constant/momentumTransport').write_text(foam_header('momentumTransport','constant')+
      'simulationType laminar;\n',encoding='ascii')
    (out/'system/fvSchemes').write_text(foam_header('fvSchemes','system')+
      '''ddtSchemes {default steadyState;}
gradSchemes {default Gauss linear;}
divSchemes {default none; div(phi,U) bounded Gauss linearUpwind grad(U); div((nuEff*dev2(T(grad(U))))) Gauss linear;}
laplacianSchemes {default Gauss linear corrected;}
interpolationSchemes {default linear;}
snGradSchemes {default corrected;}
wallDist {method meshWave;}
''',encoding='ascii')
    (out/'system/fvSolution').write_text(foam_header('fvSolution','system')+
      '''solvers {
  p {solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.1;}
  pcorr {solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0;}
  U {solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1;}
}
SIMPLE { nNonOrthogonalCorrectors 2; consistent yes; residualControl {p 1e-3; U 1e-4;} }
relaxationFactors {equations {U 0.8; ".*" 0.8;} }
''',encoding='ascii')
    (out/'system/controlDict').write_text(foam_header('controlDict','system')+
      f'''solver incompressibleFluid;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {iterations};
deltaT 1;
writeControl timeStep;
writeInterval {write_interval};
writeFormat binary;
runTimeModifiable false;
functions {{
  wallShearStressMonitor {{type wallShearStress; libs ("libfieldFunctionObjects.so"); patches (wall); writeControl writeTime;}}
'''+''.join(f'''  flow_{port} {{type surfaceFieldValue; libs ("libfieldFunctionObjects.so"); select patch; name {port}; operation sum; fields (phi); writeControl writeTime;}}
''' for port in sorted(ports,key=lambda x:int(x.split('_')[1])))+'}\n',encoding='ascii')
    report={'gate':'PREPARED_REFERENCE_ONLY_NOT_CFD_VALIDATED','native_coordinate_to_m':scalar,'kinematic_viscosity_m2_s':nu,'density_kg_m3':rho,
      'inlets_m3_s':inlet,'outlet_kinematic_pressures_m2_s2':outlet,'source_sha256':digests,
      'numerical_settings':{'max_steady_iterations':iterations,'write_interval_steps':write_interval},
      'treatment_comparison_not_performed':True,'fsi_not_performed':True}
    (out/'REFERENCE_PHYSICS_GATE.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('REFERENCE_CASE_PREPARED:',out)
    return 0
if __name__=='__main__':sys.exit(main())
