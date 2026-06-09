"""
Minimal example: random particles in a box settle under lunar gravity + contacts.

Run:
  python examples/simple_settling.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from custom_gpu_dem import DEMSimulation, DEMConfig

def main():
    cfg = DEMConfig(
        box=0.018,
        n_particles=2000,
        iron_frac=0.0,
        dt=1e-6,
        u_g=0.0,           # no gas
        physical_drag_only=True,
    )
    sim = DEMSimulation(cfg)
    sim.initialize_particles()
    print("Initial particles:", len(sim.pos))

    for i in range(300):
        sim.step()
        if i % 100 == 0:
            print(f"step {sim.step}  mean z = {float(sim.pos[:,2].mean()):.4f} m")

    out = Path("examples_output")
    out.mkdir(exist_ok=True)
    sim.write_vtk(out / "settled.vtu")
    sim.save_checkpoint(out / "settled.npz")
    print("Wrote examples_output/settled.vtu and .npz")

if __name__ == "__main__":
    main()
