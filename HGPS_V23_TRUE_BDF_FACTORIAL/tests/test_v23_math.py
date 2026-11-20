"""Unit and safety tests. They are not claims that PC-only V16–V20 model ran."""
import hashlib,importlib.util,pathlib,sqlite3,sys,tempfile,unittest,zlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_v23_true_bdf as v

class TestDeepSciencePipeline(unittest.TestCase):
 def test_source_hashes_strict(self):
  _,_,hashes=v.source_verify()
  self.assertEqual(hashes['v21_core.py'],v.REQUIRED_SHA['v21_core.py'])
  self.assertEqual(hashes['v22_core.py'],v.REQUIRED_SHA['v22_core.py'])

 def test_mask_mapping_complete(self):
  self.assertEqual(len({v.axis(i) for i in range(8)}),8)
  self.assertEqual(v.axis(0),(False,False,False))
  self.assertEqual(v.axis(7),(True,True,True))
  self.assertEqual(v.axis(4),(False,False,True))

 def test_threeway_exact_decomposition(self):
  f={m: 1+1.1*bool(m&1)+2.2*bool(m&2)+3.3*bool(m&4)+
     0.4*bool(m&3==3)+0.5*bool(m&5==5)+0.6*bool(m&6==6)+0.7*(m==7)
     for m in range(8)}
  p=v.factorial_terms(f)
  self.assertAlmostEqual(p['P1'],1.1)
  self.assertAlmostEqual(p['TBKI'],2.2)
  self.assertAlmostEqual(p['MIR4651'],3.3)
  self.assertAlmostEqual(p['P1xTBKI'],0.4)
  self.assertAlmostEqual(p['P1xMIR4651'],0.5)
  self.assertAlmostEqual(p['TBKIxMIR4651'],0.6)
  self.assertAlmostEqual(p['THREE_WAY'],0.7)
  self.assertAlmostEqual(sum(p[k] for k in ('P1','TBKI','MIR4651','P1xTBKI',
                                  'P1xMIR4651','TBKIxMIR4651','THREE_WAY')),
                         p['ALL_VS_NO_PAYLOAD'])

 def test_missing_factorial_input_holds(self):
  with self.assertRaises(v.Hold):v.factorial_terms({0:0,7:1})

 def test_original_design_subset_reproducible(self):
  for seed in v.SEEDS:
   x=v.select_indices(2048,seed)
   self.assertEqual(len(x),len(set(x)))
   self.assertGreaterEqual(x[0],0)
   self.assertLess(x[-1],65536)
   self.assertEqual(x,v.select_indices(2048,seed))

 def test_checkpoint_schema_and_resume_keys(self):
  with tempfile.TemporaryDirectory() as td:
   db=v.open_db(pathlib.Path(td)/'t.sqlite')
   with db:db.execute('INSERT INTO runs(seed,idx,scenario_sha,status,payload,nfev,seconds,error) VALUES(?,?,?,?,?,?,?,?)',
                       (11,32,'ABC','PASS',zlib.compress(b'{}'),42,1.1,''))
   self.assertEqual(db.execute('SELECT nfev FROM runs WHERE seed=11 AND idx=32').fetchone()[0],42)
   with self.assertRaises(sqlite3.IntegrityError):
    with db:db.execute('INSERT INTO runs(seed,idx,scenario_sha,status,payload,nfev,seconds,error) VALUES(?,?,?,?,?,?,?,?)',
                        (11,32,'OTHER','PASS',zlib.compress(b'{}'),43,1.1,''))
   db.close()

 def test_original_v22_solve_ivp_bdf_present(self):
  path=ROOT/'source_locked'/'v21_core.py'
  contents=path.read_text(encoding='utf-8')
  self.assertIn("method='BDF'",contents)
  self.assertIn("axis=(True,True,True)",contents)
  self.assertIn('run_conjugate',contents)

 def test_original_zero_d_unchanged_equilibrium(self):
  # Unit test only: identical cell signals must produce no modeled 0D contrast.
  from source_locked import v22_vascular
  timeline=[]
  for d in (0.,1.,2.,3.,4.):
   x={'horizon_days':d}
   for arm in ('untreated','lona','conj'):
    x.update({f'{arm}_ec_activation':1.,f'{arm}_vsmc_dysfunction':1.,f'{arm}_ecm_stiffness':1.})
   timeline.append(x)
  out=v22_vascular.paired_remodel(timeline,k_ec=.0001,k_vsmc=.0001,k_ecm=.0001)
  self.assertTrue(np.allclose(out['conj']['resistance_relative_fixed_flow'],1.))
  self.assertTrue(np.allclose(out['lona']['resistance_relative_fixed_flow'],1.))

if __name__=='__main__':unittest.main()
