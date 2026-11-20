#!/usr/bin/env bash
# Run baseline rigid-wall patient-geometry reference CFD, not therapy/FSI.
set -eo pipefail
source /opt/openfoam14/etc/bashrc
if [ "$#" -ne 4 ]; then echo 'USAGE: CASE_PATH TEMPLATE_STAGE_PATH MPI_RANKS OUTPUT_STAGE_PATH' >&2; exit 2;fi
SRC="$1"; TEMPL="$2"; NP="$3"; OUT="$4"
[[ "$NP" =~ ^[0-9]+$ ]] && [ "$NP" -gt 0 ] && [ "$NP" -le 64 ] || { echo HOLD_BAD_RANKS;exit 3; }
[ -d "$SRC/constant/polyMesh" ] || { echo HOLD_MISSING_MESH;exit 4; }
for f in 0/U 0/p constant/physicalProperties constant/momentumTransport system/controlDict system/fvSchemes system/fvSolution REFERENCE_PHYSICS_GATE.json;do
    [ -s "$TEMPL/$f" ] || { echo "HOLD_MISSING_TEMPLATE $f";exit 5; }
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DEST="$HOME/HGPS_REF_CFD_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEST" "$OUT"
cp -a "$SRC/constant" "$DEST/"
for d in 0 system;do mkdir -p "$DEST/$d";cp "$TEMPL/$d/"* "$DEST/$d/";done
cp "$TEMPL/constant/physicalProperties" "$TEMPL/constant/momentumTransport" "$DEST/constant/"
cp "$TEMPL/REFERENCE_PHYSICS_GATE.json" "$DEST/"
SCALE=$(python3 - "$DEST/REFERENCE_PHYSICS_GATE.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));print(r['native_coordinate_to_m'])
PY
)
if [ "$SCALE" != '1.0' ] && [ "$SCALE" != '1' ];then
  transformPoints -case "$DEST" -scale "($SCALE $SCALE $SCALE)" > "$OUT/transformPoints.log" 2>&1
fi
# Do not silently alter official mesh quality thresholds.
if [ ! -f "$DEST/system/meshQualityDict" ];then
  cat > "$DEST/system/meshQualityDict" <<'EOF'
FoamFile {version 2.0; format ascii; class dictionary; object meshQualityDict;}
#includeEtc "caseDicts/mesh/generation/meshQualityDict.cfg"
EOF
fi
checkMesh -case "$DEST" -allGeometry -allTopology -meshQuality > "$OUT/REFERENCE_CHECKMESH.log" 2>&1
if ! grep -Eq 'Mesh OK\.' "$OUT/REFERENCE_CHECKMESH.log" || grep -Eq 'Failed [1-9][0-9]* mesh checks' "$OUT/REFERENCE_CHECKMESH.log"; then
    echo 'HOLD: scaled mesh did not pass strict OpenFOAM quality';exit 6
fi
cat > "$DEST/system/decomposeParDict" <<EOF
FoamFile {version 2.0; format ascii; class dictionary; object decomposeParDict;}
numberOfSubdomains $NP;
method scotch;
EOF
decomposePar -case "$DEST" -force > "$OUT/decomposePar.log" 2>&1
mpirun -np "$NP" foamRun -case "$DEST" -parallel > "$OUT/foamRun.log" 2>&1
reconstructPar -case "$DEST" -latestTime > "$OUT/reconstructPar.log" 2>&1
printf '%s\n' "$DEST" > "$OUT/REFERENCE_WSL_CASE_PATH.txt"
echo 'REFERENCE_SOLVER_FINISHED_NOT_INDEPENDENTLY_VALIDATED'
# Source outputs are preserved for numerical and physiological audits.
if [ -d "$DEST/postProcessing" ]; then
    mkdir -p "$OUT/postProcessing"
    cp -a "$DEST/postProcessing/." "$OUT/postProcessing/"
fi
