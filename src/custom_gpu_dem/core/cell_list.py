"""
Cell-list accelerated force computation for custom GPU DEM (CuPy).

Hotpath rewrite (complete): Python `for c in range` + per-cell neighbor loops removed
from production path. All neighbor search now lives inside a single RawKernel launch
(compute_forces_cell_raw) for scale (N>>7k) and sustained GPU utilization (kernels
stay fed with no host per-cell work).

build_cell_list is now fully device-side (no .get() sync + no Python backward fill).

Use via:
  from cell_list import compute_forces_cell_list   # shim to fast Raw cell
or directly:
  from dem_kernels import compute_forces_cell_raw

Target: high-N sensitivities, 20k-100k+ particles, long Rung evidence runs at 0.14 bar.
"""

import cupy as cp
import numpy as np


def build_cell_list(pos, cell_size, box_size):
    """Build sorted cell list on GPU (device-only, delegates to dem_kernels for single source of truth).
    The backfill is now pure CuPy (no host .get() + Python for c in range) to keep hot path sync-free.
    """
    from dem_kernels import build_cell_list as _dk_build
    return _dk_build(pos, cell_size, box_size)


def compute_forces_cell_list(pos, vel, omega, radius, mat_type, dt,
                             cell_size=0.003, box_size=0.018,
                             **kwargs):
    """
    Cell-list accelerated contact forces (hot path).

    REWRITE COMPLETE: The old Python `for c in range(grid_size)` + nested neighbor loops
    (hundreds of small CuPy ops / potential syncs per timestep) have been removed from
    the production path.

    This is now a thin shim that routes to the fast single-RawKernel implementation
    (compute_forces_cell_raw in dem_kernels.py). The neighbor search (27 cells per
    particle) lives entirely inside one CUDA kernel launch — zero Python cell loops
    in the hot path, maximal sustained GPU utilization.

    Use for N > ~4k-5k or long evidence runs. Matches brute Raw on unit tests
    (within float tolerance from summation order).

    cell_size ~0.003 and box_size=0.018 are good defaults for current 0.14 bar
    evidence (BOX=0.016-0.018, particle radii 1e-4 to ~1.6e-3).
    """
    N = pos.shape[0]
    if N < 1200:
        from dem_kernels import compute_forces as brute
        return brute(pos, vel, omega, radius, mat_type, dt)

    from dem_kernels import compute_forces_cell_raw
    if cell_size is None or cell_size <= 0.0031:  # old default
        # conservative tuned default for typical 0.14 bar lid-clustered evidence particles
        cell_size = 0.004
    return compute_forces_cell_raw(pos, vel, omega, radius, mat_type, dt,
                                   cell_size=cell_size, box_size=box_size)


if __name__ == "__main__":
    print("Cell list module ready for testing.")