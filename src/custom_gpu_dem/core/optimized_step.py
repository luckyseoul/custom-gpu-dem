"""
Optimized DEM timestep primitives for CuPy.

Goals:
- Minimize Python loop overhead and host-device synchronizations in the hot path.
- Unconditional device-side clips (no cp.any + host branch inside per-step loop).
- Fuse simple body forces + integrate where easy.
- Support larger N (tens of k particles) to actually use VRAM and keep GPU busy.
- Cell-list friendly.

Usage in a runner:

from common.optimized_step import make_optimized_stepper, unconditional_clips

stepper = make_optimized_stepper(BOX, U_G, DAMP, add_lid_func=None)

for s in range(steps):
    f, tq = compute_forces(...)
    ... drag + distributor ...
    pos, vel, omega = stepper(pos, vel, omega, f, tq, radius, mat, DT)

# only do expensive reductions at log intervals
"""

import cupy as cp

# Bring names used inside low-level stepper / runner into module scope.
# These are implemented in dem_kernels (shared with high-level path).
from .dem_kernels import compute_drag, estimate_local_porosity, integrate


def unconditional_clips(pos, vel, BOX):
    """Device-only clips for walls + floor. No host sync, no 'if cp.any'."""
    # Floor z=0
    z = pos[:, 2]
    below = z < 0.0
    pos[below, 2] = 0.0
    vel[below, 2] = cp.abs(vel[below, 2]) * 0.80

    # Lateral walls x,y
    for ax in [0, 1]:
        p = pos[:, ax]
        below = p < 0.0
        pos[below, ax] = 0.0
        vel[below, ax] = cp.abs(vel[below, ax]) * 0.80
        over = p > BOX
        pos[over, ax] = float(BOX)
        vel[over, ax] = -cp.abs(vel[over, ax]) * 0.80
    return pos, vel


def position_only_clips(pos, vel, BOX, e_wall=0.95):
    """Device-only position containment + minimal physical bounce (restitution e_wall<1 for dissipation).
    Removes mass-scaled body forces (no k*pen*m pre-integrate acc).
    Uses post-integrate kinematic correction for floor/walls + vel reflection *e (physical wall restitution).
    This allows stable sim without the old artificial distributor etc, while providing floor support (vz = max(vz,0)*e) and wall bounces.
    e=0.95 gives mild loss per bounce (real materials); combined with real gas drag (z) + internal contact damping, velocities should relax to physical levels sustainable by U_G=0.066 drag.
    """
    # Floor: support + bounce (prevents sink-through, provides reaction without mass-scaled spring force)
    z = pos[:, 2]
    below = z < 0.0
    pos[below, 2] = 0.0
    # kill downward vel on floor (support), reflect up with e
    down = below & (vel[:, 2] < 0)
    vel[down, 2] = -vel[down, 2] * e_wall

    # Lateral walls: pos + normal vel reflection *e (no mass-scaled force)
    for ax in [0, 1]:
        p = pos[:, ax]
        v = vel[:, ax]
        # low
        below = p < 0.0
        pos[below, ax] = 0.0
        inward = below & (v < 0)
        vel[inward, ax] = -vel[inward, ax] * e_wall
        # high
        over = p > BOX
        pos[over, ax] = float(BOX)
        inward = over & (v > 0)
        vel[inward, ax] = -vel[inward, ax] * e_wall
    return pos, vel


def make_body_force_adder(add_distributor, add_walls, add_floor, add_lid=None):
    """Returns a function that applies all body forces in a few launches."""
    def apply_body_forces(f, pos, vel, radius, mat, BOX, DT):
        f = add_distributor(f, pos, radius, mat)
        f = add_walls(f, pos, radius, mat)
        f = add_floor(f, pos, vel, radius, mat)
        if add_lid is not None:
            f = add_lid(f, pos, vel, radius, mat)
        return f
    return apply_body_forces


def make_optimized_stepper(BOX, U_G, DAMP, add_lid_func=None, physical_drag_only=False):
    """
    Returns a stepper function that does the non-contact part of one step
    with minimal syncs.
    Contact forces (the expensive part) are still caller-provided.
    physical_drag_only=True: skip ALL mass-scaled body force adders (dist/wall/floor),
    use position_only_clips (no restitution-0.8 vel mods). Real drag (updated rho_g=0.0438, drag_mult=1.0)
    + gravity (from contact kernel) + particle contacts are the only physics.
    This is the drag-fix configuration per external review critique.
    """
    from .dem_kernels import compute_drag  # relative within package
    # We close over the common adders; caller still supplies contact f
    _BOX = BOX  # close over for physical boundary forces
    _DENSITY = None  # density now passed via materials or handled in kernels

    def step(pos, vel, omega, contact_force, contact_torque, radius, mat, dt,
             distributor_adder, wall_adder, floor_adder):
        # Drag (depends on porosity) -- always real now (rho default updated in dem_kernels)
        eps = estimate_local_porosity(pos, radius, BOX)
        drag = compute_drag(vel, radius, mat, U_g=U_G, local_porosity=eps)

        f = contact_force + drag
        if not physical_drag_only:
            f = distributor_adder(f, pos, radius, mat)
            f = wall_adder(f, pos, radius, mat)
            f = floor_adder(f, pos, vel, radius, mat)
        else:
            # Physical mode: no distributor (real drag only).
            # Use physical (F=k*pen, not mass-scaled) walls/floor for numerical stability
            # without the artificial mass-independent acceleration of the old adders.
            f = add_physical_wall_forces(f, pos, radius, mat, _BOX, _DENSITY)
            f = add_physical_floor_force(f, pos, vel, radius, mat, _DENSITY)

        if add_lid_func is not None:
            f = add_lid_func(f, pos, vel, radius, mat)

        # Integrate (mutates in place in current integrate)
        pos, vel, omega = integrate(pos, vel, omega, f, contact_torque, radius, mat, dt, damping=DAMP)

        # Clips - unconditional, device only
        if physical_drag_only:
            pos, vel = position_only_clips(pos, vel, BOX)
        else:
            pos, vel = unconditional_clips(pos, vel, BOX)

        return pos, vel, omega

    return step


def run_n_steps_optimized(pos, vel, omega, radius, mat,
                          compute_contact_forces_fn,
                          distributor_adder, wall_adder, floor_adder,
                          n_steps, dt, BOX, U_G, DAMP,
                          add_lid_func=None,
                          log_every=0, checkpoint_every=0, checkpoint_cb=None,
                          physical_drag_only=False):
    """
    High-level optimized runner for many steps.

    - Minimizes per-step Python work and sync points.
    - `compute_contact_forces_fn(pos, vel, omega, radius, mat, dt)` -> (f, tq)
    - Only does reductions / host work at log/checkpoint intervals.
    - Returns final (pos, vel, omega)
    - physical_drag_only: passed to stepper (skips mass body forces, uses pos-only clips)
    """
    stepper = make_optimized_stepper(BOX, U_G, DAMP, add_lid_func, physical_drag_only=physical_drag_only)

    for s in range(n_steps):
        f_contact, tq = compute_contact_forces_fn(pos, vel, omega, radius, mat, dt)
        pos, vel, omega = stepper(
            pos, vel, omega, f_contact, tq, radius, mat, dt,
            distributor_adder, wall_adder, floor_adder
        )

        if log_every and (s + 1) % log_every == 0:
            # These reductions will sync, but only rarely
            reg_mask = (mat == 0)
            reg_z = pos[reg_mask, 2]
            bed = float(cp.mean(reg_z) * 1000)
            print(f"  step {s+1}: bed={bed:.1f} mm (reg N={int(cp.sum(reg_mask))})")

        if checkpoint_every and (s + 1) % checkpoint_every == 0 and checkpoint_cb:
            checkpoint_cb(pos, vel, radius, mat, s + 1)

    return pos, vel, omega


# --- Sync-free body force adders (no cp.any host branches in hot path) ---
# These use pure boolean masking + in-place masked assign. CuPy launches the
# kernel without a device->host sync for the predicate. Use these (or
# equivalents) inside the stepper for best CPU/GPU overlap and to avoid GIL peg.
#
# NOTE (drag-fix): the mass-scaled body forces (distributor * mass, wall/floor k*pen*mass)
# were identified as primary source of non-physical velocities (18m/s mean when U_G=0.066).
# For physical runs, use the no_ versions below (or pass no-op adders) + position_only_clips.

def add_distributor_force_syncfree(force, pos, radius, mat, DENSITY):
    """Acceleration-style distributor support, unconditional masked update.
    This is a mass-scaled body force (acc * m) that injects energy independent of gas drag.
    Disabled in physical-drag-fix runs.
    """
    z = pos[:, 2]
    dist_strength = 2.8 * cp.exp(-z / 0.003)
    mass = DENSITY[mat] * (4.0 / 3.0 * cp.pi * radius**3)
    force[:, 2] += dist_strength * mass
    return force


def no_body_force(force, pos, radius, mat):
    """No-op for distributor/wall body forces. Real gas drag (in stepper) + gravity (in contact kernel) + contacts only."""
    return force


def no_floor_force(force, pos, vel, radius, mat):
    """No-op for floor body force (mass-scaled spring). Position clip still applied post-integrate."""
    return force


# Physical (non-mass-scaled) boundary forces for stability in drag-fix runs.
# These use F = k * pen (proper spring force in Newtons), added directly.
# Heavy particles (iron) get same F but smaller acc = F/m -- physical.
# Small k chosen for numerical stability without strong artificial "push".
# Distributor remains no-op (real drag only).

def add_physical_wall_forces(force, pos, radius, mat, BOX, DENSITY=None, k_wall=500.0):
    """Physical wall forces: F = k * pen (not acc*m). DENSITY unused (for sig compat).
    High k for numerical stability on initial settling/packing (required for explicit stiff Hertz + gravity + vel=0 start).
    Still physical force-based (heavy iron gets less acc) and no upward distributor."""
    for ax in [0, 1]:
        p = pos[:, ax]
        # low side
        pen = -p
        over = pen > 0.0
        f = k_wall * pen[over]
        force[over, ax] += f
        # high side
        pen = p - BOX
        over = pen > 0.0
        f = k_wall * pen[over]
        force[over, ax] -= f
    return force


def add_physical_floor_force(force, pos, vel, radius, mat, DENSITY=None, k_floor=1000.0):
    """Physical floor force: F = k * pen upward (not acc*m). High k for stability on drop/packing.
    This is still physical (force based) vs old acc-based mass-scaled. Required to prevent Hertz spikes on initial floor contact."""
    z0 = 0.0
    z = pos[:, 2]
    below = z < z0
    pen = z0 - z[below]
    f = k_floor * pen
    force[below, 2] += f
    vel[below, 2] = cp.maximum(vel[below, 2], 0.0)
    return force


def add_wall_forces_syncfree(force, pos, radius, mat, BOX, DENSITY):
    """Lateral walls, fully device-side, no 'if cp.any'."""
    k_wall = 120.0
    for ax in [0, 1]:
        p = pos[:, ax]
        # low side
        pen = -p
        over = pen > 0.0
        acc = k_wall * pen[over]
        m = DENSITY[mat[over]] * (4.0 / 3.0 * cp.pi * radius[over]**3)
        force[over, ax] += acc * m
        # high side
        pen = p - BOX
        over = pen > 0.0
        acc = k_wall * pen[over]
        m = DENSITY[mat[over]] * (4.0 / 3.0 * cp.pi * radius[over]**3)
        force[over, ax] -= acc * m
    return force


def no_wall_force(force, pos, radius, mat, BOX=None, DENSITY=None):
    """No-op for wall body forces (mass-scaled). Matches call sig wall_adder(f, pos, r, m) from stepper."""
    return force


def add_floor_force_syncfree(force, pos, vel, radius, mat, DENSITY):
    """Floor at z=0, unconditional device masked."""
    z0 = 0.0
    k_floor = 200.0
    z = pos[:, 2]
    below = z < z0
    pen = z0 - z[below]
    acc = k_floor * pen
    m = DENSITY[mat[below]] * (4.0 / 3.0 * cp.pi * radius[below]**3)
    force[below, 2] += acc * m
    vel[below, 2] = cp.maximum(vel[below, 2], 0.0)
    return force


def make_lid_freeboard_damper(BOX, DENSITY, freeboard_start=0.040, lid_z=0.060, damping=0.6):
    """
    Returns a callable add_lid_func(f, pos, vel, radius, mat) that applies
    soft freeboard damping + hard lid cap at physical ~60 mm.
    Used for the Rung1 lid+freeboard demo that keeps heights realistic while
    preserving the iron agitation differential (higher iron z, KE bias, lower dead%).
    """
    def add_lid_and_freeboard(f, pos, vel, radius, mat):
        z = pos[:, 2]
        # Soft damping in freeboard (40-60 mm)
        in_free = (z > freeboard_start) & (z < lid_z)
        if cp.any(in_free):  # rare branch, acceptable (not every particle every step)
            vel[in_free, 2] *= damping
            vel[in_free, :2] *= 0.7
        # Hard cap at lid (prevent escape, model vessel lid / freeboard limit)
        above = z >= lid_z
        if cp.any(above):
            pos[above, 2] = lid_z - 1e-6
            vel[above, 2] = -cp.abs(vel[above, 2]) * 0.3
            vel[above, :2] *= 0.4
        return f  # lid primarily kinematic; forces already integrated
    return add_lid_and_freeboard
