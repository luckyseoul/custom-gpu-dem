from __future__ import annotations
from pathlib import Path
from typing import Optional
import cupy as cp
import numpy as np

from .core.optimized_step import position_only_clips
from .core.dem_kernels import compute_forces_cell_raw, compute_drag
from .config import DEMConfig
from .io import save_checkpoint, write_vtk_particles


class DEMSimulation:
    def __init__(self, config: DEMConfig | dict | str):
        if isinstance(config, str):
            self.cfg = DEMConfig.from_yaml(config) if Path(config).exists() else DEMConfig()
        elif isinstance(config, dict):
            self.cfg = DEMConfig.from_dict(config)
        else:
            self.cfg = config

        self.pos = None
        self.vel = None
        self.omega = None
        self.radius = None
        self.mat = None
        self.step = 0

        self.BOX = float(self.cfg.box)
        self.DT = float(self.cfg.dt)
        self.U_G = float(self.cfg.u_g)
        self._physical = bool(self.cfg.physical_drag_only)
        self._e_wall = 0.95

    def initialize_particles(self):
        n = int(self.cfg.n_particles)
        iron_frac = float(self.cfg.iron_frac)
        n_iron = int(n * iron_frac)
        n_reg = n - n_iron
        rng = np.random.default_rng(getattr(self.cfg, "seed", 42))

        pos_reg = rng.uniform(0, self.BOX, (n_reg, 3)).astype(np.float32)
        pos_iron = rng.uniform(0, self.BOX, (n_iron, 3)).astype(np.float32)
        pos_iron[:, 2] *= 0.5

        pos = np.vstack([pos_reg, pos_iron])
        r_reg = rng.uniform(1e-5, 2e-4, n_reg).astype(np.float32)
        r_iron = np.full(n_iron, 0.0015, dtype=np.float32)
        radius = np.concatenate([r_reg, r_iron])
        mat = np.array([0] * n_reg + [1] * n_iron, dtype=np.int32)

        idx = rng.permutation(n)
        pos, radius, mat = pos[idx], radius[idx], mat[idx]

        self.pos = cp.asarray(pos)
        self.vel = cp.zeros((n, 3), dtype=cp.float32)
        self.omega = cp.zeros((n, 3), dtype=cp.float32)
        self.radius = cp.asarray(radius)
        self.mat = cp.asarray(mat)
        self.step = 0

    def step(self):
        if self.pos is None:
            raise RuntimeError("initialize first")

        f, tq = compute_forces_cell_raw(
            self.pos, self.vel, self.omega, self.radius, self.mat, self.DT,
            cell_size=self.cfg.cell_size or 0.004, box_size=self.BOX
        )

        drag = compute_drag(self.vel, self.radius, self.mat, U_g=self.U_G,
                            rho_g=self.cfg.rho_g, mu_g=self.cfg.mu_g, drag_mult=self.cfg.drag_mult)
        f = f + drag

        if not self._physical:
            near = self.pos[:, 2] < 0.003
            f[near, 2] += 0.0006

        density = cp.asarray(self.cfg.materials.density, dtype=cp.float32)
        mass = density[self.mat] * (4.0/3.0) * cp.pi * (self.radius ** 3)
        acc = f / cp.clip(mass[:, None], 1e-12, None)
        self.vel = self.vel + acc * self.DT
        self.pos = self.pos + self.vel * self.DT

        self.pos, self.vel = position_only_clips(self.pos, self.vel, self.BOX, e_wall=self._e_wall)

        if self.cfg.lid:
            z = self.pos[:, 2]
            in_damp = (z > self.cfg.lid.freeboard_z) & (z < self.cfg.lid.lid_z)
            self.vel[in_damp, 2] *= (1.0 - self.cfg.lid.damp)
            over = z > self.cfg.lid.lid_z
            self.pos[over, 2] = float(self.cfg.lid.lid_z)
            self.vel[over, 2] = cp.minimum(self.vel[over, 2], 0.0)

        self.step += 1

    def run(self, steps: int, log_every: int = 100, vtk_every: Optional[int] = None, out_dir: str = "output"):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for s in range(steps):
            self.step()
            if log_every and (self.step % log_every == 0):
                print(f"step {self.step}  mean_z={float(self.pos[:,2].mean()):.5f}")
            if vtk_every and (self.step % vtk_every == 0):
                self.write_vtk(f"{out_dir}/p_{self.step:06d}.vtu")

    def write_vtk(self, path: str, extra: Optional[dict] = None):
        write_vtk_particles(path, self.pos, self.vel, self.radius, self.mat, self.step, extra or {})

    def save_checkpoint(self, path: str):
        save_checkpoint(path, self.pos, self.vel, self.omega, self.radius, self.mat, self.step,
                        box=self.BOX, u_g=self.U_G)
