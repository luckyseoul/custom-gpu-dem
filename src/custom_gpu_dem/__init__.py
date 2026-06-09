"""
custom-gpu-dem
High-performance single-GPU Discrete Element Method (DEM) simulator using CuPy.

Key features (strengths vs many research codes):
- Device-only cell list + single RawKernel contact (scales well for N=5k-20k+)
- "Physical" (force-based, not mass-scaled acceleration) wall/floor boundaries for stability with stiff contacts
- Real gas drag (Stokes + quadratic) with local porosity correction and full drag_mult=1.0 support
- Reproducible high-N scaling via settled base + jittered addition (preserves containment)
- Physical lid + freeboard damping for contained fluidized bed simulations
- Clean separation of contact forces and integration step

This package was extracted and cleaned from the RCFX patent evidence campaign.
It is intended as a reusable tool for granular simulations (especially low-g, cohesive, low-pressure gas drag cases).

Example:
    from custom_gpu_dem import DEMSimulation, DEMConfig, load_config
    cfg = DEMConfig(n_particles=2000)  # or load_config("my_sim.yaml")
    sim = DEMSimulation(cfg)
    sim.initialize_particles()
    sim.run(steps=1000)
    sim.write_vtk("output/particles.vtu")

See docs/ and examples/ for full user guide and demos.
"""
from .core.dem_kernels import (
    compute_forces,
    compute_forces_cell_raw,
    build_cell_list,
    compute_drag,
)
from .core.optimized_step import (
    make_optimized_stepper,
    position_only_clips,
    make_lid_freeboard_damper,
)
from .simulation import DEMSimulation
from .config import DEMConfig, load_config

__version__ = "0.1.0"
__all__ = [
    "DEMSimulation",
    "DEMConfig",
    "load_config",
    
    "compute_forces",
    "compute_forces_cell_raw",
    "build_cell_list",
    "compute_drag",
    "make_optimized_stepper",
    "position_only_clips",
    "make_lid_freeboard_damper",
]
