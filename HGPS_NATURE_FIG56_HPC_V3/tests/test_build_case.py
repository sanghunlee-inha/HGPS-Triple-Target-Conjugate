"""Test-only synthetic source documents; tests are never passed as real clinical input."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
CODE=Path(__file__).resolve().parents[1]/'code'
class TestBuild(unittest.TestCase):
 def test_case_builder_with_temporary_test_sources(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   for x in ('units','flow','fluid'):(p/x).write_text('SYNTHETIC_TEST_ONLY',encoding='utf-8')
   m={'case_type':'rigid_wall_steady_reference_only','approved_input_sources':True,
      'provenance':{'mra_coordinate_units':str(p/'units'),'flow_boundary_measurement':str(p/'flow'),'blood_fluid_properties':str(p/'fluid')},
      'coordinate_unit_to_m':1.,'kinematic_viscosity_m2_s':1e-6,'density_kg_m3':1000.,
      'preserved_surface_patch_names':['wall']+[f'port_{i}' for i in range(2,12)],
      'inlet_flows_m3_s':{'port_2':1e-6},'outlet_kinematic_pressures_m2_s2':{f'port_{i}':0.0 for i in range(3,12)},
      'max_steady_iterations':10,'write_interval_steps':5}
   f=p/'source.json';f.write_text(json.dumps(m))
   proc=subprocess.run([sys.executable,str(CODE/'make_reference_case.py'),'--physics',str(f),'--out',str(p/'out')],capture_output=True,text=True)
   self.assertEqual(proc.returncode,0,proc.stderr)
   self.assertIn('flowRateInletVelocity',(p/'out/0/U').read_text())
   self.assertEqual(json.loads((p/'out/REFERENCE_PHYSICS_GATE.json').read_text())['gate'],'PREPARED_REFERENCE_ONLY_NOT_CFD_VALIDATED')
if __name__=='__main__':unittest.main()
