"""Software-only tests. Synthetic inputs are created in a temporary directory, NOT research results."""
import csv, json, subprocess, sys, tempfile, unittest
from pathlib import Path
CODE=Path(__file__).resolve().parents[1]/'code'
class Tests(unittest.TestCase):
    def test_v22_real_logic_with_test_only_data(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); f=root/'synthetic.csv'
            with f.open('w',newline='') as o:
                w=csv.writer(o);w.writerow(['seed','release_mode','hypothetical_0D_rate','delta_R_C_minus_L'])
                for seed in (11,23,37,51,73):
                    for rate in ('low','high'):
                        for i in range(10):w.writerow([seed,'independent',rate,float(i) if rate=='high' else -float(i)])
            proc=subprocess.run([sys.executable,str(CODE/'v22_reanalysis.py'),'--input',str(f),'--out',str(root/'output')],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            result=json.loads((root/'output/V22_STUDY_AUDIT.json').read_text())
            self.assertEqual(result['gate'],'PASS_CONDITIONAL_0D_COMPARISON_ONLY')
            self.assertEqual(result['rows'],100)
            self.assertEqual(len(result['groups']),2)
            self.assertFalse(result['clinical_efficacy'])
    def test_ref_case_fails_without_measured_sources(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); f=root/'empty_config.json'
            f.write_text(json.dumps({'case_type':'rigid_wall_steady_reference_only','approved_input_sources':False}))
            proc=subprocess.run([sys.executable,str(CODE/'make_reference_case.py'),'--physics',str(f),'--out',str(root/'cases')],capture_output=True,text=True)
            self.assertNotEqual(proc.returncode,0)
            self.assertFalse((root/'cases/0/U').exists())
    def test_missing_real_v22_holds_and_writes_report(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            proc=subprocess.run([sys.executable,str(CODE/'run_deep_workflow.py'),'--research-root',str(root),
                '--stage-dir',str(root/'work'),'--only','v22'],capture_output=True,text=True)
            self.assertEqual(proc.returncode,2)
            self.assertEqual(json.loads((root/'work/STUDY_STATUS.json').read_text())['figure5_0d']['gate'],'HOLD_NO_V22_RAW')
if __name__=='__main__':unittest.main()
