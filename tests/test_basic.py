import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from custom_gpu_dem import DEMSimulation, DEMConfig

def test_init_and_step():
    cfg = DEMConfig(n_particles=200, iron_frac=0.0, physical_drag_only=True, u_g=0.0, dt=1e-6)
    sim = DEMSimulation(cfg)
    sim.initialize_particles()
    assert len(sim.pos) == 200
    for _ in range(10):
        sim.step()
    assert sim.current_step == 10
    assert float(sim.pos[:,2].mean()) >= 0.0   # didn't fall through floor

def test_vtk_export(tmp_path):
    cfg = DEMConfig(n_particles=50, physical_drag_only=True)
    sim = DEMSimulation(cfg)
    sim.initialize_particles()
    sim.step()
    out = tmp_path / "test.vtu"
    sim.write_vtk(out)
    assert out.exists()
    assert out.stat().st_size > 100
