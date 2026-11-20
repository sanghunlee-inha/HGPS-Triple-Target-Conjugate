"""Actual test invocations. Never label output as patient data."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from make_toy_fixture import main as fixture
from fig6 import run as fig6
from fig7 import run as fig7
from fig8 import run as fig8
from fig5 import run as fig5

def main(root):
 root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
 fixture(root/'fixture')
 g=root/'fixture'
 r6=fig6([g/'test_cfd.vtu'],root/'figure6_test',wss=g/'toy_surface.vtp',chunk=2)
 assert not r6['status'].startswith('HOLD'),r6
 r7=fig7(g/'transport_toy.json',g/'test_cfd.vtu',g/'toy_surface.vtp',root/'figure7_test',allow_exploratory=True)
 assert r7['status'].startswith('3D_CONVECTION_DIFFUSION_DONE'),r7
 r8=fig8(g/'sensitivity_toy.json',root/'figure8_test',workers=2)
 assert r8['status'].startswith('FORWARD_MODEL_GSA_COMPLETED'),r8
 d=json.loads((root/'figure8_test'/'FIG8_REPLICATED_GLOBAL_SENSITIVITY_REPORT.json').read_text())
 s={r['parameter']:r for r in d['summary']}
 assert abs(s['x']['S1_mean']-.2)<.15, s
 assert abs(s['y']['ST_mean']-.8)<.15, s
 raw=Path(__file__).resolve().parent.parent.parent.parent/'hgps_fig56_test_pkpd.csv'
 if raw.exists():
  r5=fig5(raw,root/'figure5_test',bootstrap=64,allow_other_hash=True)
  assert r5['status'].startswith('ANALYZED_VERIFIED_0D'),r5
 print('ALL_SYNTHETIC_NUMERICAL_FIXTURE_TESTS_PASS; NOT_RESEARCH_RESULTS')
if __name__=='__main__':main(sys.argv[1])
