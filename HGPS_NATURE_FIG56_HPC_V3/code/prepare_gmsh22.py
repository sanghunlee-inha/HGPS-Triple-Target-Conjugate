#!/usr/bin/env python3
"""Validated Medit->Gmsh 2.2 conversion for one HGPS MMG run. No solver is run."""
import json
import math
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import meshio


def fail(message):
    raise RuntimeError("HOLD: " + message)


def extract(mesh):
    data = {}
    if "medit:ref" not in mesh.cell_data:
        fail("No Medit references")
    for kind in ("line", "triangle", "tetra"):
        blocks = []
        tags = []
        for index, cb in enumerate(mesh.cells):
            if cb.type != kind:
                if cb.type not in ("line", "triangle", "tetra"):
                    fail("Unexpected cell type: " + cb.type)
                continue
            e = np.asarray(cb.data, dtype=np.int64)
            r = np.asarray(mesh.cell_data["medit:ref"][index], dtype=np.int64)
            if len(e) != len(r):
                fail(f"Reference count mismatch for {kind}")
            blocks.append(e)
            tags.append(r)
        if blocks:
            data[kind] = (np.concatenate(blocks), np.concatenate(tags))
    return data


def normalized_physical_names(baseline, tri_tags, tet_tags):
    raw = baseline.get("group_names")
    if not isinstance(raw, dict) or not raw:
        fail("No original group_names in baseline_audit.json; won't invent boundary names")
    info = {}
    used = set()
    for original_name, item in raw.items():
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            fail(f"Invalid group_names entry: {original_name!r}")
        tag, dim = int(item[0]), int(item[1])
        if (dim, tag) in info:
            fail(f"Duplicate physical group ({dim},{tag})")
        name = re.sub(r"[^A-Za-z0-9_]", "_", str(original_name)).strip("_")
        if not name:
            fail(f"Group name cannot be converted to OpenFOAM name: {original_name!r}")
        if name[0].isdigit():
            name = "group_" + name
        if name in used or name == "defaultFaces":
            fail(f"Group name collision/reserved name: {name}")
        used.add(name)
        info[(dim, tag)] = {"original_name": str(original_name), "foam_name": name}
    expected = {(2, int(v)) for v in tri_tags} | {(3, int(v)) for v in tet_tags}
    if set(info) != expected:
        fail(f"Names do not match observed Physical Groups. Missing={sorted(expected-set(info))}, Extra={sorted(set(info)-expected)}")
    return info


def main():
    src = Path(sys.argv[1])
    dest = Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)
    audit = json.loads((src / "BOUNDARY_ORIENTATION_AUDIT.json").read_text(encoding="utf-8-sig"))
    baseline = json.loads((src / "baseline_audit.json").read_text(encoding="utf-8-sig"))
    geo = json.loads((src / "STRICT_GEOMETRY_AUDIT.json").read_text(encoding="utf-8-sig"))

    required = {
        "preconversion_surface_gate": "PASS",
        "surface_group_ids_equal": True,
        "surface_group_counts_equal": True,
        "one_to_one_surface_vertices": True,
        "surface_coordinates_within_tolerance": True,
        "surface_connectivity_with_labels_equal": True,
        "candidate_triangles_equal_exterior_faces": True,
        "candidate_triangles_inward": 0,
        "candidate_triangles_winding_unknown": 0,
        "tetra_nonmanifold_faces": 0,
    }
    for key, expected in required.items():
        if audit.get(key) != expected:
            fail(f"Pre-conversion gate missing/failed: {key}={audit.get(key)!r}")
    if int(audit["candidate_triangles_outward"]) != int(audit["optimized_exterior_faces"]):
        fail("Not all boundary triangles are outward")

    vs = geo.get("optimized_volume_stats", {})
    if int(vs.get("negative", -1)) or int(vs.get("zero", -1)):
        fail("Invalid tetra volume sign in strict audit")

    print("Reading existing MMG mesh (no remeshing)...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mesh = meshio.read(src / "mmg_repaired.mesh")
    xyz = np.asarray(mesh.points[:, :3], dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        fail("Invalid 3D node coordinates")
    data = extract(mesh)
    if "triangle" not in data or "tetra" not in data:
        fail("Missing triangles or tetrahedra")
    tri, tritags = data["triangle"]
    tet, tettags = data["tetra"]
    if len(tet) != int(geo["optimized_tetra"]) or len(tri) != int(geo["optimized_triangle"]):
        fail("Changed triangle/tetra counts since strict audit")
    if len(tet) != int(vs.get("positive", -1)):
        fail("Positive tetra count mismatch")
    if np.any(tritags <= 0) or np.any(tettags <= 0):
        fail("Nonpositive physical tags on triangle or tetra")
    if set(map(int, np.unique(tritags))) != set(range(1, 12)) or set(map(int, np.unique(tettags))) != {100}:
        fail("Physical group ID mismatch")
    if "line" in data and np.any(data["line"][1] != 0):
        fail("Unexpected nonzero line tags: manual audit required")

    counts = Counter(map(int, tritags))
    reference_counts = {int(k): int(v) for k, v in audit["optimized_surface_tags"].items()} if "optimized_surface_tags" in audit else None
    if reference_counts is None:
        # Previous audits use either surface_group_count or other naming.
        original_counts = baseline.get("tag_counts", {}).get("triangle", {})
        if isinstance(original_counts, dict) and original_counts:
            reference_counts = {int(k): int(v) for k,v in original_counts.items()}
    if reference_counts is not None and dict(counts) != reference_counts:
        fail("Patch face counts do not match baseline")

    names = normalized_physical_names(baseline, np.unique(tritags), np.unique(tettags))
    manifest = {
        "source_medit": str(src / "mmg_repaired.mesh"),
        "msh_version": "2.2 ASCII",
        "vertices": int(len(xyz)),
        "triangles": int(len(tri)),
        "tetrahedra": int(len(tet)),
        "discarded_zero_tag_lines": int(len(data["line"][0])) if "line" in data else 0,
        "physical_groups": [
            {"dimension": d, "tag": t, **details, "faces": int(counts[t]) if d==2 else None}
            for (d,t), details in sorted(names.items())
        ],
        "patch_faces": {names[(2,tag)]["foam_name"]: int(n) for tag,n in sorted(counts.items())},
        "cfd_quality_gate": "HOLD_NOT_CHECKMESH_VALIDATED",
    }
    msh = dest / "vascular_mmg3d_v22.msh"
    print("Writing Gmsh 2.2 ASCII:", msh, flush=True)
    with msh.open("w", encoding="ascii", newline="\n") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$PhysicalNames\n{len(names)}\n")
        for (dim,tag), info in sorted(names.items()):
            f.write(f'{dim} {tag} "{info["foam_name"]}"\n')
        f.write("$EndPhysicalNames\n")
        f.write(f"$Nodes\n{len(xyz)}\n")
        for i, (x,y,z) in enumerate(xyz, start=1):
            f.write(f"{i} {x:.17g} {y:.17g} {z:.17g}\n")
        f.write("$EndNodes\n")
        f.write(f"$Elements\n{len(tri)+len(tet)}\n")
        elem_id = 1
        for elems, tags, etype in ((tri, tritags, 2), (tet, tettags, 4)):
            for ids, tag in zip(elems, tags):
                nodes = " ".join(str(int(q)+1) for q in ids)
                f.write(f"{elem_id} {etype} 2 {int(tag)} {int(tag)} {nodes}\n")
                elem_id += 1
        f.write("$EndElements\n")

    print("Read-back validation...", flush=True)
    check = meshio.read(msh)
    if check.points.shape != xyz.shape:
        fail("Gmsh read-back vertex count differs")
    maxdiff = float(np.max(np.abs(np.asarray(check.points)-xyz)))
    if maxdiff > 1e-14:
        fail(f"Gmsh read-back coordinate error {maxdiff}")
    for kind, elems, tags in (("triangle", tri, tritags), ("tetra", tet, tettags)):
        matching = [(i, b) for i,b in enumerate(check.cells) if b.type==kind]
        re = np.concatenate([b.data for i,b in matching])
        rt = np.concatenate([np.asarray(check.cell_data["gmsh:physical"][i]) for i,b in matching])
        if not np.array_equal(re, elems) or not np.array_equal(rt, tags):
            fail(f"Read-back changed {kind} connectivity or tag order")
    manifest["gmsh_readback_verified"] = True
    manifest["gmsh_max_coordinate_delta"] = maxdiff
    (dest / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("PASS: Gmsh read-back and physical groups. CFD quality remains HOLD.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"CONVERSION_HOLD: {exc!r}", file=sys.stderr, flush=True)
        sys.exit(2)
