#!/usr/bin/env python3
"""Never mistake WSL preflight failure for CFD mesh failure."""
import json, re, sys
from pathlib import Path

out = Path(sys.argv[1]); manifest_file = out / 'conversion_manifest.json'
if not manifest_file.exists():
    print('HOLD: conversion_manifest.json missing; cannot finalize')
    sys.exit(3)
manifest = json.loads(manifest_file.read_text(encoding='utf-8-sig'))
expected = manifest['patch_faces']

def read_text(name):
    p = out / name
    return p.read_text(encoding='utf-8-sig', errors='replace') if p.exists() else ''

preflight = {}
if read_text('WSL_PREFLIGHT_STATUS.json'):
    try: preflight = json.loads(read_text('WSL_PREFLIGHT_STATUS.json'))
    except Exception as exc: preflight = {'parse_error': repr(exc)}

status = {}
for line in read_text('WSL_EXIT_STATUS.txt').splitlines():
    if '=' in line:
        k,v = line.split('=',1)
        try: status[k] = int(v)
        except ValueError: status[k] = v

boundary = read_text('polyMesh/boundary')
actual = {}
for match in re.finditer(r'(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\n\s*\{([^{}]*)\}',boundary):
    name, body = match.groups()
    count = re.search(r'\bnFaces\s+(\d+)\s*;',body)
    if count: actual[name] = int(count.group(1))
missing = {k:v for k,v in expected.items() if k not in actual}
mismatched = {k:{'expected':v,'actual':actual.get(k)} for k,v in expected.items() if k in actual and actual[k]!=v}
unexpected = {k:v for k,v in actual.items() if k not in expected and v!=0}
patch_pass = bool(len(expected)==11 and not missing and not mismatched and not unexpected)

def log_audit(path):
    content = read_text(path)
    if not content:
        return {'present':False,'mesh_ok':False,'failed_checks':None}
    fails = [int(x) for x in re.findall(r'Failed\s+(\d+)\s+mesh\s+checks?', content, re.I)]
    mesh_ok = bool(re.search(r'\bMesh\s+OK\b', content))
    return {'present': True,'mesh_ok':mesh_ok,'failed_checks':max(fails) if fails else None,
            'tail':'\n'.join(content.splitlines()[-45:])}

basic = log_audit('checkMesh_allGeometry_allTopology.log')
quality = log_audit('checkMesh_meshQuality.log')
process_exit = None
try: process_exit = int(read_text('WSL_PROCESS_EXIT_CODE.txt').strip())
except ValueError: pass
convert_pass = status.get('convert_rc') == 0
basic_pass = status.get('basic_rc') == 0 and basic['mesh_ok'] and basic['failed_checks'] is None
quality_pass = (status.get('quality_rc') == 0 and quality['mesh_ok'] and quality['failed_checks'] is None
                and status.get('quality_config') not in ('NOT_FOUND', None))
gate = 'PASS_OPENFOAM_MESH_QUALITY' if (convert_pass and patch_pass and basic_pass and quality_pass and process_exit==0) else 'HOLD'
report = {
    'gate':gate,
    'execution_stage':status.get('stage') or preflight.get('stage','UNKNOWN'),
    'environment_reason':preflight.get('reason','unknown'),
    'wsl_process_exit_code':process_exit,
    'original_surface_groups_expected':expected,
    'openfoam_actual_patch_faces':actual,
    'patch_counts_exact':patch_pass,
    'missing_patches':missing,'patch_count_mismatches':mismatched,
    'unexpected_nonempty_patches':unexpected,
    'run_exit_codes':status,
    'checkMesh_all_geometry_topology':basic,
    'checkMesh_mesh_quality':quality,
    'quality_dictionary_source':status.get('quality_config','NOT_RUN'),
    'wsl_environment_hold':read_text('WSL_ENVIRONMENT_HOLD.txt').strip(),
    'wsl_stderr_tail':'\n'.join(read_text('WSL_RUN_STDERR.log').splitlines()[-25:]),
    'wsl_stdout_tail':'\n'.join(read_text('WSL_RUN_STDOUT.log').splitlines()[-25:]),
    'note':'OpenFOAM mesh quality is not CFD physics, flow/FSI, or patient-specific response validation.'
}
path = out / 'OPENFOAM_MESH_GATE_REPORT.json'
path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
print('OPENFOAM_MESH_GATE:',gate)
print('STAGE:',report['execution_stage'])
print('REASON:',report['environment_reason'])
print('EXIT:',process_exit,'CONVERT:',status.get('convert_rc'),'BASIC:',status.get('basic_rc'),'QUALITY:',status.get('quality_rc'))
print('QUALITY_DICTIONARY:',report['quality_dictionary_source'])
print('PATCH_COUNTS_EXACT:',patch_pass)
print('BASIC_MESH_OK:',basic_pass,'QUALITY_MESH_OK:',quality_pass)
print('REPORT:',path)
if report['wsl_environment_hold']:
    print('ENVIRONMENT_HOLD:',report['wsl_environment_hold'])
for logname in ('gmshToFoam.log','checkMesh_allGeometry_allTopology.log','checkMesh_meshQuality.log'):
    data = read_text(logname)
    if data: print('\n---',logname,'(last 15 lines) ---\n'+'\n'.join(data.splitlines()[-15:]))
if report['wsl_stderr_tail']:
    print('\n--- WSL STDERR ---\n'+report['wsl_stderr_tail'])
sys.exit(0 if gate.startswith('PASS') else 2)
