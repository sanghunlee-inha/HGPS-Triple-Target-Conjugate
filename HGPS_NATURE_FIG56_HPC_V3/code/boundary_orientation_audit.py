"""Read-only MMG3D boundary topology, tag, and orientation audit.

Usage: python HGPS_MMG3D_BOUNDARY_ORIENTATION_AUDIT.py <run_directory>
Does not modify source .mesh or create a CFD mesh.
"""
import json
import sys
import warnings
from pathlib import Path

import meshio
import numpy as np
from scipy.spatial import cKDTree


def read(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return meshio.read(path)


def elements(mesh, kind):
    selected = [(i, b) for i, b in enumerate(mesh.cells) if b.type == kind]
    if not selected:
        raise RuntimeError('Missing mesh element type: ' + kind)
    key = 'medit:ref'
    if key not in mesh.cell_data:
        raise RuntimeError('Missing medit:ref data')
    return (np.concatenate([np.asarray(b.data, dtype=np.int64)
                            for _, b in selected]),
            np.concatenate([np.asarray(mesh.cell_data[key][i], dtype=np.int64)
                            for i, _ in selected]))


def counts(ref):
    u, n = np.unique(ref, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(u, n)}


def order_xyz(keys):
    return np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))


def order_with_tag(tri, tags):
    keys = np.sort(tri, axis=1)
    return np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0], tags))


def orientation(reference, candidate):
    """Parity of candidate triangular winding relative to reference.

    Reference and candidate have been paired by sorted node indices.
    """
    same = np.zeros(len(reference), dtype=bool)
    reverse = np.zeros(len(reference), dtype=bool)
    inv = reference[:, [0, 2, 1]]
    for k in range(3):
        same |= np.all(candidate == np.roll(reference, k, axis=1), axis=1)
        reverse |= np.all(candidate == np.roll(inv, k, axis=1), axis=1)
    return same, reverse


def exterior_faces(tets):
    # For positive-handed tetrahedra these oriented triangles face outward.
    faces = np.concatenate((tets[:, [1, 2, 3]], tets[:, [0, 3, 2]],
                            tets[:, [0, 1, 3]], tets[:, [0, 2, 1]]), axis=0)
    keys = np.sort(faces, axis=1)
    order = order_xyz(keys)
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)) + 1]
    multiplicity = np.diff(np.r_[starts, len(order)])
    external = faces[order[starts[multiplicity == 1]]]
    return external, int(np.count_nonzero(multiplicity > 2))


def main(root):
    baseline = json.loads((root / 'baseline_audit.json').read_text(encoding='utf-8'))
    tol = float(baseline['coord_tolerance'])
    original = read(root / 'input.mesh')
    optimized = read(root / 'mmg_repaired.mesh')
    t0, g0 = elements(original, 'triangle')
    t1, g1 = elements(optimized, 'triangle')
    tet0, v0 = elements(original, 'tetra')
    tet1, v1 = elements(optimized, 'tetra')

    surf0, surf1 = np.unique(t0), np.unique(t1)
    dist, near = cKDTree(original.points[surf0, :3]).query(optimized.points[surf1, :3])
    coord_delta = float(np.max(dist))
    unique_map = (len(surf0) == len(surf1) == len(np.unique(near)))
    coord_ok = unique_map and coord_delta <= tol

    report = {
        'original_surface_tags': counts(g0),
        'optimized_surface_tags': counts(g1),
        'surface_group_ids_equal': set(g0.tolist()) == set(g1.tolist()),
        'surface_group_counts_equal': counts(g0) == counts(g1),
        'original_volume_tags': counts(v0),
        'optimized_volume_tags': counts(v1),
        'volume_group_ids_equal': set(v0.tolist()) == set(v1.tolist()),
        'volume_element_counts_allowed_to_change': True,
        'max_surface_point_delta': coord_delta,
        'coordinate_tolerance': tol,
        'one_to_one_surface_vertices': bool(unique_map),
        'surface_coordinates_within_tolerance': bool(coord_ok),
        'surface_connectivity_with_labels_equal': False,
        'orientation_equal_count': None,
        'orientation_reversed_count': None,
        'orientation_unmatched_count': None,
        'orientation_reversed_by_group': None,
        'optimized_exterior_faces': None,
        'candidate_triangles_equal_exterior_faces': None,
        'tetra_nonmanifold_faces': None,
        'candidate_triangles_outward': None,
        'candidate_triangles_inward': None,
        'candidate_triangles_winding_unknown': None,
        'preconversion_surface_gate': 'HOLD',
        'openfoam_quality_gate': 'HOLD_NOT_RUN'
    }

    if coord_ok:
        mapping = np.full(len(optimized.points), -1, dtype=np.int64)
        mapping[surf1] = surf0[near]
        mapped = mapping[t1]
        ix0, ix1 = order_with_tag(t0, g0), order_with_tag(mapped, g1)
        keys0 = np.column_stack((g0[ix0], np.sort(t0[ix0], axis=1)))
        keys1 = np.column_stack((g1[ix1], np.sort(mapped[ix1], axis=1)))
        topology_ok = bool(np.array_equal(keys0, keys1))
        report['surface_connectivity_with_labels_equal'] = topology_ok
        if topology_ok:
            same, reverse = orientation(t0[ix0], mapped[ix1])
            report['orientation_equal_count'] = int(same.sum())
            report['orientation_reversed_count'] = int(reverse.sum())
            report['orientation_unmatched_count'] = int((~(same | reverse)).sum())
            report['orientation_reversed_by_group'] = {
                str(int(tag)): int(np.count_nonzero(reverse & (g0[ix0] == tag)))
                for tag in np.unique(g0)
            }

    # Independent assessment: do candidate surface triangles coincide with
    # unique exterior faces of tetrahedra, and follow outward tetra orientation?
    print('Extracting candidate exterior tetra faces...', flush=True)
    external, nonmanifold = exterior_faces(tet1)
    report['optimized_exterior_faces'] = int(len(external))
    report['tetra_nonmanifold_faces'] = nonmanifold
    candidate_order = order_xyz(np.sort(t1, axis=1))
    exterior_order = order_xyz(np.sort(external, axis=1))
    boundary_ok = bool(np.array_equal(np.sort(t1[candidate_order], axis=1),
                                      np.sort(external[exterior_order], axis=1)))
    report['candidate_triangles_equal_exterior_faces'] = boundary_ok
    if boundary_ok:
        same, reverse = orientation(external[exterior_order], t1[candidate_order])
        report['candidate_triangles_outward'] = int(same.sum())
        report['candidate_triangles_inward'] = int(reverse.sum())
        report['candidate_triangles_winding_unknown'] = int((~(same | reverse)).sum())

    accepted = (
        coord_ok and report['surface_group_counts_equal']
        and report['surface_connectivity_with_labels_equal']
        and report['volume_group_ids_equal']
        and nonmanifold == 0 and boundary_ok
        and report['candidate_triangles_outward'] == len(t1)
    )
    report['preconversion_surface_gate'] = 'PASS' if accepted else 'HOLD'

    path = root / 'BOUNDARY_ORIENTATION_AUDIT.json'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print('SAVED:', path)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python HGPS_MMG3D_BOUNDARY_ORIENTATION_AUDIT.py <run_directory>')
    main(Path(sys.argv[1]))
