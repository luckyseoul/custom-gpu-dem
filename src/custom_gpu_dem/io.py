"""
I/O helpers for custom-gpu-dem.

- Checkpoint (npz) - full state
- Minimal pure-Python VTK point cloud writer (.vtu) for ParaView - no extra heavy deps required at runtime.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
import os

def save_checkpoint(path: str | Path, pos, vel, omega, radius, mat, step: int, **meta):
    """Save full simulation state for restart."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        pos=pos.get() if hasattr(pos, "get") else pos,
        vel=vel.get() if hasattr(vel, "get") else vel,
        omega=omega.get() if hasattr(omega, "get") else omega,
        radius=radius.get() if hasattr(radius, "get") else radius,
        mat=mat.get() if hasattr(mat, "get") else mat,
        step=step,
        **{k: (v.get() if hasattr(v, "get") else v) for k, v in meta.items()},
    )

def load_checkpoint(path: str | Path):
    """Load checkpoint. Returns numpy arrays + step + meta."""
    data = np.load(path)
    return {
        "pos": data["pos"],
        "vel": data["vel"],
        "omega": data["omega"],
        "radius": data["radius"],
        "mat": data["mat"],
        "step": int(data["step"]),
        "meta": {k: data[k] for k in data.files if k not in {"pos","vel","omega","radius","mat","step"}},
    }

def write_vtk_particles(path: str | Path, pos, vel, radius, mat, step: int = 0, extra: dict | None = None):
    """
    Write a simple UnstructuredGrid .vtu with point data (no cells needed for particle viz).

    Pure Python implementation - works without the 'vtk' package.
    ParaView can open these directly.

    pos, vel, radius, mat: numpy arrays (or cupy - will be .get())
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(pos, "get"):
        pos = pos.get()
        vel = vel.get()
        radius = radius.get()
        mat = mat.get()

    n = len(pos)
    extra = extra or {}

    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfCells="0">\n')
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        for p in pos:
            f.write(f"          {p[0]} {p[1]} {p[2]}\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")
        f.write("      <PointData Scalars=\"radius\">\n")

        # radius
        f.write('        <DataArray type="Float32" Name="radius" format="ascii">\n')
        f.write("          " + " ".join(f"{r:.6e}" for r in radius) + "\n")
        f.write("        </DataArray>\n")

        # material/type
        f.write('        <DataArray type="Int32" Name="material" format="ascii">\n')
        f.write("          " + " ".join(str(int(m)) for m in mat) + "\n")
        f.write("        </DataArray>\n")

        # velocity
        f.write('        <DataArray type="Float32" Name="velocity" NumberOfComponents="3" format="ascii">\n')
        for v in vel:
            f.write(f"          {v[0]} {v[1]} {v[2]}\n")
        f.write("        </DataArray>\n")

        for name, arr in extra.items():
            if hasattr(arr, "get"):
                arr = arr.get()
            arr = np.asarray(arr).ravel()
            f.write(f'        <DataArray type="Float32" Name="{name}" format="ascii">\n')
            f.write("          " + " ".join(f"{x:.6e}" for x in arr) + "\n")
            f.write("        </DataArray>\n")

        f.write("      </PointData>\n")
        f.write("      <Cells>\n")
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii"></DataArray>\n')
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii"></DataArray>\n')
        f.write('        <DataArray type="UInt8" Name="types" format="ascii"></DataArray>\n')
        f.write("      </Cells>\n")
        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")
