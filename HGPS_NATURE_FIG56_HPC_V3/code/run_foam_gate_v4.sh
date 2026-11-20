#!/usr/bin/env bash
# Robust diagnostic OpenFOAM mesh gate. Never starts CFD solver.
set -eo pipefail
winout="$1"
status="$winout/WSL_EXIT_STATUS.txt"
stage="BEGIN"
convert_rc=99
basic_rc=99
quality_rc=99
quality_config="NOT_FOUND"
linuxcase=""
finish() {
  local process_rc="$?"
  set +e
  {
    printf 'stage=%s\n' "$stage"
    printf 'convert_rc=%s\n' "$convert_rc"
    printf 'basic_rc=%s\n' "$basic_rc"
    printf 'quality_rc=%s\n' "$quality_rc"
    printf 'quality_config=%s\n' "$quality_config"
    printf 'shell_exit_rc=%s\n' "$process_rc"
  } > "$status"
  printf 'WSL_FINAL_STAGE=%s SHELL_EXIT=%s\n' "$stage" "$process_rc"
}
trap finish EXIT

stage="ENVIRONMENT"
{
  echo "Date: $(date -Is)"
  echo "User: $(whoami)"
  echo "Distro: ${WSL_DISTRO_NAME:-unknown}"
  echo "HOME: $HOME"
  echo "PATH: $PATH"
  echo "WM_PROJECT_DIR: ${WM_PROJECT_DIR:-unset}"
  echo "Linux output: $winout"
} > "$winout/WSL_LINUX_ENVIRONMENT.log"

if ! command -v gmshToFoam >/dev/null 2>&1 || ! command -v checkMesh >/dev/null 2>&1; then
  for rc in /opt/openfoam*/etc/bashrc /usr/lib/openfoam*/etc/bashrc /usr/share/openfoam/etc/bashrc; do
    if [[ -f "$rc" ]]; then
      set +e
      # shellcheck disable=SC1090
      source "$rc" >> "$winout/WSL_LINUX_ENVIRONMENT.log" 2>&1
      source_rc=$?
      set -e
      echo "Source attempt: $rc (exit=$source_rc)" >> "$winout/WSL_LINUX_ENVIRONMENT.log"
      if command -v gmshToFoam >/dev/null 2>&1 && command -v checkMesh >/dev/null 2>&1; then
        break
      fi
    fi
  done
fi
if ! command -v gmshToFoam >/dev/null 2>&1 || ! command -v checkMesh >/dev/null 2>&1; then
  echo 'HOLD: gmshToFoam/checkMesh not installed or not configured in this WSL distribution' | tee "$winout/WSL_ENVIRONMENT_HOLD.txt"
  echo 'Paths checked: /opt/openfoam*/etc/bashrc, /usr/lib/openfoam*/etc/bashrc, /usr/share/openfoam/etc/bashrc' >> "$winout/WSL_ENVIRONMENT_HOLD.txt"
  exit 4
fi
{
  echo "gmshToFoam=$(command -v gmshToFoam)"
  echo "checkMesh=$(command -v checkMesh)"
  echo "WM_PROJECT_DIR=${WM_PROJECT_DIR:-unset}"
} >> "$winout/WSL_LINUX_ENVIRONMENT.log"

stage="STAGING"
linuxcase="$HOME/HGPS_MMG3D_20261113_230655_gate_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$linuxcase/system" "$linuxcase/constant"
printf '%s\n' "$linuxcase" > "$winout/WSL_CASE_PATH.txt"
cp "$winout/vascular_mmg3d_v22.msh" "$linuxcase/vascular_mmg3d_v22.msh"
cat > "$linuxcase/system/controlDict" <<'EOF'
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object controlDict;
}
application checkMesh;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
EOF

# Copy an OpenFOAM-distributed quality dictionary, never invent thresholds.
for source_dict in \
  "${WM_PROJECT_DIR:-/nonexistent}/etc/caseDicts/mesh/generation/meshQualityDict" \
  "${WM_PROJECT_DIR:-/nonexistent}/etc/caseDicts/meshQualityDict" \
  "/usr/share/openfoam/etc/caseDicts/mesh/generation/meshQualityDict"; do
  if [[ -f "$source_dict" ]]; then
    cp "$source_dict" "$linuxcase/system/meshQualityDict"
    quality_config="$source_dict"
    echo "$source_dict" > "$winout/OPENFOAM_QUALITY_DICT_SOURCE.txt"
    break
  fi
done

# OpenFOAM 14 ships its official quality thresholds as meshQualityDict.cfg.
# Do NOT invent or relax thresholds: include the upstream configuration unmodified.
if [[ "$quality_config" == "NOT_FOUND" && -f "${WM_PROJECT_DIR:-/nonexistent}/etc/caseDicts/mesh/generation/meshQualityDict.cfg" ]]; then
  cat > "$linuxcase/system/meshQualityDict" <<'QUALITY_EOF'
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    location "system";
    object meshQualityDict;
}
#includeEtc "caseDicts/mesh/generation/meshQualityDict.cfg"
QUALITY_EOF
  quality_config="${WM_PROJECT_DIR}/etc/caseDicts/mesh/generation/meshQualityDict.cfg (upstream include)"
  printf '%s\n' "$quality_config" > "$winout/OPENFOAM_QUALITY_DICT_SOURCE.txt"
fi

stage="GMSH_TO_FOAM"
convert_rc=0
gmshToFoam -case "$linuxcase" "$linuxcase/vascular_mmg3d_v22.msh" > "$winout/gmshToFoam.log" 2>&1 || convert_rc=$?
if [[ $convert_rc -ne 0 ]]; then
  echo "HOLD: gmshToFoam exited with $convert_rc" >> "$winout/WSL_LINUX_ENVIRONMENT.log"
  exit 5
fi

stage="CHECKMESH_GEOMETRY_TOPOLOGY"
basic_rc=0
checkMesh -case "$linuxcase" -allGeometry -allTopology > "$winout/checkMesh_allGeometry_allTopology.log" 2>&1 || basic_rc=$?

stage="CHECKMESH_QUALITY"
if [[ -f "$linuxcase/system/meshQualityDict" ]]; then
  quality_rc=0
  checkMesh -case "$linuxcase" -allGeometry -allTopology -meshQuality > "$winout/checkMesh_meshQuality.log" 2>&1 || quality_rc=$?
else
  quality_rc=97
  echo 'HOLD: An official meshQualityDict was not found in this OpenFOAM installation. Thresholds were NOT invented.' > "$winout/checkMesh_meshQuality.log"
fi

stage="COPY_POLYMESH"
if [[ -f "$linuxcase/constant/polyMesh/boundary" ]]; then
  mkdir -p "$winout/polyMesh"
  cp -a "$linuxcase/constant/polyMesh/." "$winout/polyMesh/"
fi
stage="FINISHED"
echo "OPENFOAM_CASE=$linuxcase"
echo "CONVERT_EXIT=$convert_rc BASIC_EXIT=$basic_rc QUALITY_EXIT=$quality_rc QUALITY_DICT=$quality_config"
