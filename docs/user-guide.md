# custom-gpu-dem User Guide

High-performance single-GPU Discrete Element Method (DEM) using CuPy.

## Installation

```bash
# Ensure you have CuPy matching your CUDA version first
# pip install cupy-cuda12x   (or cupy-cuda11x, etc.)

pip install -e .
```

## Quickstart

```python
from custom_gpu_dem import DEMSimulation, DEMConfig

cfg = DEMConfig(
    n_particles=2000,
    iron_frac=0.07,
    physical_drag_only=True,   # recommended validated mode
    u_g=0.066,
    dt=6.5e-7,
    box=0.018,
)
sim = DEMSimulation(cfg)
sim.initialize_particles()

for _ in range(500):
    sim.step()

sim.write_vtk("particles.vtu")          # ParaView-ready
sim.save_checkpoint("final.npz")
```

## Configuration

Use `DEMConfig` or a YAML file.

Important fields for physical fidelity:
- `physical_drag_only=True` — use only real gas drag + contacts + physical clips (matches the clean good-variable evidence).
- `u_g`, `rho_g`, `mu_g` — gas properties.
- `lid.freeboard_z`, `lid.lid_z`, `lid.damp` — physical lid + freeboard.

## Output

- `write_vtk(path)` — lightweight .vtu with position, velocity, radius, material.
- `save_checkpoint` / `from_checkpoint` — full restart support.

## Key Features (what makes this useful)

- Device-side cell list + single RawKernel for good scaling on one GPU.
- Physical (force-based) boundaries instead of mass-scaled forces → much more stable with stiff contacts.
- Real gas drag (Stokes + quadratic) with porosity correction and full drag strength.
- Reproducible high-N scaling technique (settled base + controlled particle addition).
- Built-in physical lid + freeboard for contained fluidized beds.

## Running from Command Line

```bash
custom-gpu-dem run my_config.yaml --steps 1000 --vtk-every 100
```

## Extending

Edit `src/custom_gpu_dem/core/dem_kernels.py` for new contact/drag models.

The high-level `DEMSimulation.step()` is intentionally simple so you can subclass or replace the integration logic.

## Origin

Extracted and cleaned from the RCFX (PERRY-RCFX-004) low-pressure regolith + iron shot heat recovery modeling work. The physical_drag_only + physical lid + real-drag good-variable data (1.5 mm iron at 3.5 m/s, 34.47 mm iron lift, 3.58× EMI, 100% containment) is the primary validated result.

## License

MIT. See LICENSE.
