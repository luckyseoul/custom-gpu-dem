"""
Configuration handling for custom-gpu-dem.

Supports dict or YAML (if pyyaml installed).

Example YAML:
box: 0.018
n_particles: 6500
iron_frac: 0.07
dt: 6.5e-7
u_g: 0.066
physical_drag_only: true
lid:
  freeboard_z: 0.040
  lid_z: 0.060
materials:
  young: [3e7, 2.1e11]
  ...
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import os

try:
    import yaml  # pyyaml or compatible
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

@dataclass
class MaterialProps:
    young: list[float] = field(default_factory=lambda: [3.0e7, 2.1e11])
    poisson: list[float] = field(default_factory=lambda: [0.25, 0.29])
    density: list[float] = field(default_factory=lambda: [3100.0, 7870.0])
    friction: list[float] = field(default_factory=lambda: [0.55, 0.35])
    rolling_friction: list[float] = field(default_factory=lambda: [0.08, 0.025])
    restitution: list[float] = field(default_factory=lambda: [0.25, 0.45])
    surface_energy: list[list[float]] = field(default_factory=lambda: [[0.00012, 0.0], [0.0, 0.0]])

@dataclass
class LidConfig:
    freeboard_z: float = 0.040
    lid_z: float = 0.060
    damp: float = 0.08

@dataclass
class DEMConfig:
    box: float = 0.018
    n_particles: int = 6500
    iron_frac: float = 0.07
    dt: float = 6.5e-7
    u_g: float = 0.066
    gravity: list[float] = field(default_factory=lambda: [0.0, 0.0, -1.625])
    physical_drag_only: bool = True
    drag_mult: float = 1.0
    rho_g: float = 0.0438
    mu_g: float = 2.28e-5
    cell_size: Optional[float] = None
    materials: MaterialProps = field(default_factory=MaterialProps)
    lid: LidConfig = field(default_factory=LidConfig)
    seed: int = 42

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DEMConfig":
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                if k == "materials":
                    setattr(cfg, k, MaterialProps(**v))
                elif k == "lid":
                    setattr(cfg, k, LidConfig(**v))
                else:
                    setattr(cfg, k, v)
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "DEMConfig":
        if not HAS_YAML:
            raise RuntimeError("pyyaml not installed. pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def save_yaml(self, path: str):
        if not HAS_YAML:
            raise RuntimeError("pyyaml required to save YAML")
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
