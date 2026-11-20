import sys,unittest,tempfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import v11_main as v
class TestV11(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(v.SEEDS,(11,23,37,51,73))
        self.assertEqual(len(v.FACTORS),8)
        self.assertEqual(len(v.DOSES),3)
        self.assertEqual(len(v.TRACE_STATE_NAMES),22)
    def test_lock_determinism(self):
        self.assertEqual(v.canon({'x':1,'y':2}),v.canon({'y':2,'x':1}))
        self.assertNotEqual(v.canon({'y':2}),v.canon({'y':3}))
    def test_science_gate_requires_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,'MISSING_SOURCE'):
                v.verify_hash(Path(d)/'missing','0'*64)
    def test_observed_empty_gate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'inputs').mkdir()
            (p/'inputs/observed_pkpd.csv').write_text('study_id,observed_value\n',encoding='utf-8')
            self.assertEqual(v.observed_gate(p),'HOLD_NO_OBSERVED_PK_PD')
    def test_observed_nonempty_gate_never_passes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'inputs').mkdir()
            (p/'inputs/observed_pkpd.csv').write_text('study_id,observed_value\nA,0.03\n',encoding='utf-8')
            self.assertTrue(v.observed_gate(p).startswith('HOLD_'))
    def test_missing_results_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,'MISSING_SOURCE'):
                v.preflight(Path(d)/'a',Path(d)/'b')
if __name__=='__main__':unittest.main()
