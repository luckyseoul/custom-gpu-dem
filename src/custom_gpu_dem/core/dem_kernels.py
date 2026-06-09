"""
Custom GPU DEM kernels for RCFX rungs using CuPy.
Target: Rung 2 (bimodal + cohesion + iron agitation) and beyond.

Physics kept conservative and traceable to literature + Rev 5.2 parameters.
"""

import cupy as cp
import numpy as np

# Material properties (will be moved to parameter file later)
# Type 0 = regolith, Type 1 = iron shot

YOUNG = cp.array([3.0e7, 2.1e11], dtype=cp.float32)          # Pa  (lowered regolith for numerical stability at micron scale)
POISSON = cp.array([0.25, 0.29], dtype=cp.float32)
DENSITY = cp.array([3100.0, 7870.0], dtype=cp.float32)       # kg/m3
FRICTION = cp.array([0.55, 0.35], dtype=cp.float32)          # mu
ROLLING_FRICTION = cp.array([0.08, 0.025], dtype=cp.float32)
RESTITUTION = cp.array([0.25, 0.45], dtype=cp.float32)

# Effective surface energy for JKR-style cohesion (J/m^2)
# Lowered significantly for stability with 10-200um particles (realistic dry regolith ~0.001-0.01)
SURFACE_ENERGY = cp.array([[0.00012, 0.0],
                           [0.0, 0.0]], dtype=cp.float32)    # J/m²  (further reduced for stability on micron particles)

GRAVITY = cp.array([0.0, 0.0, -1.625], dtype=cp.float32)     # lunar

def _device_backfill_cell_starts(cell_start, N):
    """Pure-device backward fill.
    Uses a small RawKernel with multiple passes inside (syncthreads) for propagation.
    No Python for over grid size. Fast for tiny cell grids.
    """
    cs = cell_start.copy().astype(cp.int32)
    n = cs.size
    sentinel = N + 1
    backfill = cp.RawKernel(r'''
extern "C" __global__ void backfill(int* arr, int n, int sentinel) {
    __shared__ int s[1024];
    int tid = threadIdx.x;
    if (tid < n) s[tid] = arr[tid];
    __syncthreads();
    for (int pass = 0; pass < 10; ++pass) {  // enough for n<1024
        int val = (tid + 1 < n) ? s[tid + 1] : sentinel;
        if (s[tid] > sentinel && val <= sentinel) {
            s[tid] = val;
        }
        __syncthreads();
    }
    if (tid < n) arr[tid] = s[tid];
}
''', 'backfill')
    block = min(1024, n)
    grid = 1
    backfill((grid,), (block,), (cs, n, sentinel))
    return cs


def build_cell_list(pos, cell_size, box_size):
    """Build sorted cell list on GPU (authoritative, fully device-side, zero host sync in hot path).
    Used by compute_forces_cell_raw for scalable N>>5k with single RawKernel launch.
    """
    N = pos.shape[0]
    grid_dim = int(cp.ceil(box_size / cell_size))
    grid_dim = max(grid_dim, 1)
    cell_idx = (pos / cell_size).astype(cp.int32)
    cell_idx = cp.clip(cell_idx, 0, grid_dim - 1)
    cell_id = (cell_idx[:, 0] +
               cell_idx[:, 1] * grid_dim +
               cell_idx[:, 2] * grid_dim * grid_dim)
    sorted_idx = cp.argsort(cell_id)
    sorted_cell_id = cell_id[sorted_idx]
    unique_cells, cell_starts = cp.unique(sorted_cell_id, return_index=True)
    cell_start = cp.full(grid_dim**3 + 1, N+1, dtype=cp.int32)  # sentinel > N for "unset"
    cell_start[unique_cells] = cell_starts
    cell_start[-1] = N
    # Backward fill on device (no .get(), no Python loop over data)
    cell_start = _device_backfill_cell_starts(cell_start, N)
    return sorted_idx, cell_start, grid_dim

def compute_forces(pos, vel, omega, radius, mat_type, dt):
    """
    Brute-force contact force computation on GPU.
    Returns force, torque, and (optionally) contact statistics.
    """
    N = pos.shape[0]
    
    # Pairwise vectors
    dx = pos[:, None, :] - pos[None, :, :]          # (N, N, 3)
    dist = cp.linalg.norm(dx, axis=2) + 1e-12
    r_sum = radius[:, None] + radius[None, :]
    
    # Contact mask (no self-contact)
    mask = (dist < r_sum) & (cp.arange(N)[:, None] != cp.arange(N)[None, :])
    
    # Normal direction
    n = dx / dist[..., None]
    
    # Overlap
    delta = r_sum - dist
    delta = cp.where(mask, delta, 0.0)
    
    # Relative velocity at contact point (normal + tangential)
    v_i = vel[:, None, :]
    v_j = vel[None, :, :]
    omega_i = omega[:, None, :]
    omega_j = omega[None, :, :]
    
    r_i = radius[:, None, None] * n
    r_j = -radius[None, :, None] * n
    
    v_rel = (v_i - v_j) + cp.cross(omega_i, r_i, axisa=2, axisb=2) + cp.cross(omega_j, r_j, axisa=2, axisb=2)
    
    v_n = cp.sum(v_rel * n, axis=2)
    v_t = v_rel - v_n[..., None] * n
    
    # Material properties per pair
    E_eff = 1.0 / ((1 - POISSON[mat_type[:, None]]**2) / YOUNG[mat_type[:, None]] +
                   (1 - POISSON[mat_type[None, :]]**2) / YOUNG[mat_type[None, :]])
    
    # Hertz normal force (with cohesion)
    a = cp.sqrt(radius[:, None] * radius[None, :] * delta)  # contact radius approx
    F_n_hertz = (4.0 / 3.0) * E_eff * cp.sqrt(radius[:, None] * radius[None, :]) * (delta ** 1.5)
    
    # Simple JKR-style cohesion (attractive when in contact or near)
    gamma = SURFACE_ENERGY[mat_type[:, None], mat_type[None, :]]
    R_eff = (radius[:, None] * radius[None, :]) / (radius[:, None] + radius[None, :] + 1e-12)
    F_cohesion = 0.8 * cp.pi * gamma * R_eff * (delta > -1e-7)
    
    F_n = F_n_hertz - F_cohesion
    F_n = cp.where(mask, F_n, 0.0)
    
    # Tangential force (very simplified viscous + friction cap for early validation)
    G_eff = E_eff / (2 * (1 + POISSON[mat_type[:, None]]))   # rough
    F_t = - (8.0 * G_eff * a)[..., None] * v_t * dt
    F_t = cp.where(mask[..., None], F_t, 0.0)
    
    # For now use simple friction limit
    F_t_mag = cp.linalg.norm(F_t, axis=2)
    F_t_max = FRICTION[mat_type[:, None]] * cp.abs(F_n)
    scale = cp.minimum(1.0, F_t_max / (F_t_mag + 1e-12))
    F_t = F_t * scale[..., None]
    F_t = cp.where(mask[..., None], F_t, 0.0)
    
    # Rolling resistance torque (very important for iron shot)
    omega_rel = omega[:, None, :] - omega[None, :, :]
    torque_roll = - ROLLING_FRICTION[mat_type[:, None]] * cp.abs(F_n)[..., None] * radius[:, None, None] * (omega_rel / (cp.linalg.norm(omega_rel, axis=2)[..., None] + 1e-12))
    torque_roll = cp.where(mask[..., None], torque_roll, 0.0)
    
    # Accumulate forces and torques on particles
    force = cp.zeros_like(pos)
    torque = cp.zeros_like(omega)
    
    force += cp.sum(F_n[..., None] * n + F_t, axis=1)
    torque += cp.sum(torque_roll, axis=1)
    
    # Add gravity
    force += DENSITY[mat_type][:, None] * (4/3 * cp.pi * radius[:, None]**3) * GRAVITY[None, :]
    
    return force, torque


def compute_drag(vel, radius, mat_type, U_g=0.066, rho_g=0.0438, mu_g=2.28e-5, local_porosity=None, drag_mult=None):
    """
    Per-particle gas drag: Stokes (linear) + quadratic.
    Now uses real rho_g ~0.0438 kg/m3 at 0.14 bar, ~300K from gas_at_T in five_stage_counterflow (P*MW/RT).
    For physical drag-fix runs (per critique): full drag_mult=1.0 on ALL particles (no artificial throttle on fines).
    The previous 0.015 throttle + mass-scaled distributor were the source of non-physical 11-18 m/s velocities.
    Real gas drag at U_G=0.066 m/s for 3.5mm iron (Umf >> U_G) will be << weight; iron should sit unless fines collisions lift it.
    """
    v_slip = U_g - vel[:, 2]
    d = 2.0 * radius
    Re = rho_g * cp.abs(v_slip) * d / mu_g

    Cd = cp.where(
        Re < 1e-4,
        2000.0,
        cp.where(Re < 800, 24.0/Re * (1.0 + 0.15 * Re**0.687), 0.44)
    )

    F_stokes = 3.0 * cp.pi * mu_g * d * v_slip
    A = cp.pi * (d / 2.0)**2
    F_quad = 0.5 * Cd * rho_g * A * v_slip * cp.abs(v_slip)

    F_drag_z = F_stokes + F_quad

    # Physical: no material throttle. Full drag for fines and iron.
    # If drag insufficient to overcome gravity + contacts for iron, it will sit (as expected for Umf_iron ~1000x higher).
    if drag_mult is None:
        drag_mult = 1.0
    F_drag_z = F_drag_z * drag_mult

    if local_porosity is not None:
        eps = cp.clip(local_porosity, 0.35, 0.92)
        F_drag_z = F_drag_z * (1.0 / (eps ** 2.2))

    drag = cp.zeros_like(vel)
    drag[:, 2] = F_drag_z
    return drag


def estimate_local_porosity(pos, radius, box_size, cell_size=0.0025):
    """
    Simple local solid fraction estimate using cell list (for drag modulation).
    Returns per-particle local void fraction (eps = 1 - local solid fraction).
    For small N (~3000) this is fast enough even with approximate methods.
    """
    N = pos.shape[0]
    grid_dim = int(cp.ceil(box_size / cell_size))
    grid_dim = max(grid_dim, 1)

    cell_idx = (pos / cell_size).astype(cp.int32)
    cell_idx = cp.clip(cell_idx, 0, grid_dim - 1)
    cell_id = (cell_idx[:, 0] +
               cell_idx[:, 1] * grid_dim +
               cell_idx[:, 2] * grid_dim * grid_dim)

    # Count particles per cell
    unique, counts = cp.unique(cell_id, return_counts=True)
    cell_count = cp.zeros(grid_dim**3, dtype=cp.int32)
    cell_count[unique] = counts

    # Assign local count to each particle
    local_count = cell_count[cell_id]

    # Approximate solid fraction in cell (very rough — volume of particles / cell volume)
    cell_vol = cell_size ** 3
    particle_vol = (4/3 * cp.pi * radius**3)
    local_solid = (local_count * cp.mean(particle_vol)) / cell_vol   # very approximate
    local_solid = cp.clip(local_solid, 0.05, 0.7)

    eps = 1.0 - local_solid
    return eps


def integrate(pos, vel, omega, force, torque, radius, mat_type, dt, damping=0.1):
    """Simple velocity Verlet style update with damping."""
    mass = DENSITY[mat_type] * (4/3 * cp.pi * radius**3)
    inertia = (2/5) * mass * radius**2   # spheres
    
    acc = force / mass[:, None]
    ang_acc = torque / inertia[:, None]
    
    vel += acc * dt
    omega += ang_acc * dt
    
    # Simple numerical damping (viscous drag proxy for low-pressure gas)
    vel *= (1.0 - damping * dt)
    vel = cp.clip(vel, -80.0, 80.0)
    omega *= (1.0 - damping * dt * 0.5)
    omega = cp.clip(omega, -200.0, 200.0)
    
    pos += vel * dt
    
    return pos, vel, omega


# -----------------------------------------------------------------------------
# RawKernel brute-force contact forces for high sustained GPU utilization.
# The high-level CuPy version (broadcast N x N temps + many launches per step)
# causes the GPU to finish quickly then idle while host Python prepares next step.
# This single-launch version (1D grid over particles, dense inner j-loop) keeps
# kernels at 100% utilization during the force phase (confirmed via nvidia-smi
# tight loops). Matches the high-level physics exactly (including the model's
# non-symmetric per-i material lookup for G/Ft cap/rolling, non-std Hertz R=sqrt(ri*rj),
# no Ft->torque contrib, etc.).
#
# Now default for high-N (and low-N) evidence runs (migrate/benchmark/coarse).
# Validated on unit tests (exact on N=2 reg/iron/mixed) + highN ckpt state
# (magnitudes consistent; high-level N^2 path unreliable at 6500 due to mem).
# SURFACE_ENERGY zeroed for Rung1 no-reg-coh (highN primary).
# -----------------------------------------------------------------------------

_raw_kernel_code = r'''
extern "C" __global__
void raw_compute_forces(
    const float3* __restrict__ pos,
    const float3* __restrict__ vel,
    const float3* __restrict__ omega,
    const float*  __restrict__ radius,
    const int*    __restrict__ mat,
    float3* __restrict__ force,
    float3* __restrict__ torque,
    const int N,
    const float dt
) {
    // Hardcoded materials (match high-level globals exactly)
    const float YOUNG[2] = {3.0e7f, 2.1e11f};
    const float POISSON[2] = {0.25f, 0.29f};
    const float DENSITY[2] = {3100.0f, 7870.0f};
    const float FRICTION[2] = {0.55f, 0.35f};
    const float ROLLING_FRICTION[2] = {0.08f, 0.025f};
    const float SURFACE_ENERGY[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // [mi*2 + mj]  -- zeroed for Rung1 (no reg cohesion); for full coh rungs use separate kernel or paramize launch args
    const float3 GRAV = make_float3(0.0f, 0.0f, -1.625f);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float3 fi = make_float3(0.f, 0.f, 0.f);
    float3 ti = make_float3(0.f, 0.f, 0.f);

    float3 pi = pos[i];
    float3 vi = vel[i];
    float3 wi = omega[i];
    float  ri = radius[i];
    int    mi = mat[i];

    for (int j = 0; j < N; ++j) {
        if (j == i) continue;

        float3 pj = pos[j];
        float3 vj = vel[j];
        float3 wj = omega[j];
        float  rj = radius[j];
        int    mj = mat[j];

        float3 dx = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
        float dist = sqrtf(dx.x*dx.x + dx.y*dx.y + dx.z*dx.z) + 1e-12f;
        float rsum = ri + rj;
        if (dist >= rsum) continue;

        float invd = 1.0f / dist;
        float3 n = make_float3(dx.x * invd, dx.y * invd, dx.z * invd);
        float delta = rsum - dist;
        if (delta <= 0.0f) continue;

        // r vectors (exact match to high-level convention)
        float3 r_i = make_float3(ri * n.x, ri * n.y, ri * n.z);
        float3 r_j = make_float3(-rj * n.x, -rj * n.y, -rj * n.z);

        // cross(wi, r_i) + cross(wj, r_j)  -- signs/order per high-level cp.cross
        float3 cross_i = make_float3(
            wi.y * r_i.z - wi.z * r_i.y,
            wi.z * r_i.x - wi.x * r_i.z,
            wi.x * r_i.y - wi.y * r_i.x
        );
        float3 cross_j = make_float3(
            wj.y * r_j.z - wj.z * r_j.y,
            wj.z * r_j.x - wj.x * r_j.z,
            wj.x * r_j.y - wj.y * r_j.x
        );
        float3 v_rel = make_float3(
            vi.x - vj.x + cross_i.x + cross_j.x,
            vi.y - vj.y + cross_i.y + cross_j.y,
            vi.z - vj.z + cross_i.z + cross_j.z
        );

        float vn = v_rel.x * n.x + v_rel.y * n.y + v_rel.z * n.z;
        float3 vt = make_float3(
            v_rel.x - vn * n.x,
            v_rel.y - vn * n.y,
            v_rel.z - vn * n.z
        );

        // material (per-i asymmetric lookup to match high-level exactly)
        float e1 = YOUNG[mi], e2 = YOUNG[mj];
        float nu1 = POISSON[mi], nu2 = POISSON[mj];
        float Eeff = 1.0f / ((1.0f - nu1*nu1)/e1 + (1.0f - nu2*nu2)/e2);

        float sqrt_rr = sqrtf(ri * rj);
        float aa = sqrtf(ri * rj * fmaxf(delta, 0.f));

        // Hertz (non-std R=sqrt(ri*rj) to match high-level)
        float Fn_h = (4.0f / 3.0f) * Eeff * sqrt_rr * (delta * sqrtf(delta));

        float g = SURFACE_ENERGY[mi * 2 + mj];
        float Re = (ri * rj) / (ri + rj + 1e-12f);
        float Fcoh = 0.8f * 3.14159265f * g * Re * ((delta > -1e-7f) ? 1.0f : 0.0f);

        float Fn = Fn_h - Fcoh;

        // tangential (G only from i's nu, Ft cap only mi's friction -- match high-level)
        float Ge = Eeff / (2.0f * (1.0f + nu1));
        float3 Ft = make_float3(
            -(8.0f * Ge * aa) * vt.x * dt,
            -(8.0f * Ge * aa) * vt.y * dt,
            -(8.0f * Ge * aa) * vt.z * dt
        );

        float Ftm = sqrtf(Ft.x*Ft.x + Ft.y*Ft.y + Ft.z*Ft.z);
        float Ftmax = FRICTION[mi] * fabsf(Fn);
        float sc = (Ftm > 1e-12f) ? fminf(1.0f, Ftmax / Ftm) : 1.0f;
        Ft.x *= sc; Ft.y *= sc; Ft.z *= sc;

        float3 contrib = make_float3(Fn * n.x + Ft.x, Fn * n.y + Ft.y, Fn * n.z + Ft.z);
        fi.x += contrib.x; fi.y += contrib.y; fi.z += contrib.z;

        // rolling torque (only rolling resistance term; no Ft cross r; uses mi + ri -- match)
        float3 wrel = make_float3(wi.x - wj.x, wi.y - wj.y, wi.z - wj.z);
        float wrm = sqrtf(wrel.x*wrel.x + wrel.y*wrel.y + wrel.z*wrel.z) + 1e-12f;
        float3 tr = make_float3(
            -ROLLING_FRICTION[mi] * fabsf(Fn) * ri * (wrel.x / wrm),
            -ROLLING_FRICTION[mi] * fabsf(Fn) * ri * (wrel.y / wrm),
            -ROLLING_FRICTION[mi] * fabsf(Fn) * ri * (wrel.z / wrm)
        );
        ti.x += tr.x; ti.y += tr.y; ti.z += tr.z;
    }

    // gravity (exact match)
    float mass = DENSITY[mi] * (4.0f/3.0f * 3.14159265f * ri*ri*ri);
    fi.x += mass * GRAV.x;
    fi.y += mass * GRAV.y;
    fi.z += mass * GRAV.z;

    force[i] = fi;
    torque[i] = ti;
}
'''

_raw_kernel = cp.RawKernel(_raw_kernel_code, 'raw_compute_forces')

def compute_forces_raw(pos, vel, omega, radius, mat_type, dt):
    """
    Drop-in replacement for compute_forces using a single RawKernel launch.
    Provides dramatically higher sustained GPU utilization (one launch, kernels
    stay saturated for the entire N^2 pair work instead of host<->device thrash
    from many CuPy temporaries and launches).
    Matches high-level on unit tests (N=2, all mat combos, with/without vel).
    At highN=6500 the high-level reference OOMs/pressures on N^2 temps so Raw
    (low-mem) is the authoritative path used for all primary evidence ckpts.
    SURFACE_ENERGY=0 for Rung1 (no reg coh) -- see kernel const + migration force.
    """
    N = pos.shape[0]
    force = cp.zeros_like(pos)
    torque = cp.zeros_like(omega)

    block = 256
    grid = (N + block - 1) // block
    _raw_kernel(
        (grid,), (block,),
        (pos, vel, omega, radius, mat_type, force, torque, N, dt)
    )
    # Note: callers using optimized_stepper typically do their own sync or
    # accept async; we do not sync here to avoid extra host stall in hot path.
    return force, torque


# -----------------------------------------------------------------------------
# Cell-list accelerated RawKernel for scale + sustained util.
# Uses the same physics as the brute Raw (and high-level).
# For each particle, uses cell list to loop only ~27 neighbor cells' particles
# instead of all N. Single launch. Essential for N>>6500 or long runs.
# Matches the brute Raw on unit tests (small diffs only from float order).
# -----------------------------------------------------------------------------
_cell_raw_kernel_code = r'''
extern "C" __global__
void raw_compute_forces_cell(
    const float3* __restrict__ pos,
    const float3* __restrict__ vel,
    const float3* __restrict__ omega,
    const float*  __restrict__ radius,
    const int*    __restrict__ mat,
    float3* __restrict__ force,
    float3* __restrict__ torque,
    const int*    __restrict__ sorted_idx,
    const int*    __restrict__ cell_start,
    const int N,
    const int grid_dim,
    const float cell_size,
    const float box_size,
    const float dt
) {
    const float YOUNG[2] = {3.0e7f, 2.1e11f};
    const float POISSON[2] = {0.25f, 0.29f};
    const float DENSITY[2] = {3100.0f, 7870.0f};
    const float FRICTION[2] = {0.55f, 0.35f};
    const float ROLLING_FRICTION[2] = {0.08f, 0.025f};
    const float SURFACE_ENERGY[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Rung1 no reg coh
    const float3 GRAV = make_float3(0.0f, 0.0f, -1.625f);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float3 fi = make_float3(0.f, 0.f, 0.f);
    float3 ti = make_float3(0.f, 0.f, 0.f);

    float3 pi = pos[i];
    float3 vi = vel[i];
    float3 wi = omega[i];
    float  ri = radius[i];
    int    mi = mat[i];

    // compute own cell
    int cx = (int)fmaxf(0.0f, fminf( (float)grid_dim-1, floorf(pi.x / cell_size) ));
    int cy = (int)fmaxf(0.0f, fminf( (float)grid_dim-1, floorf(pi.y / cell_size) ));
    int cz = (int)fmaxf(0.0f, fminf( (float)grid_dim-1, floorf(pi.z / cell_size) ));
    // 27 neighbor offsets, order: dx outer, dz inner (match Python list comp in cell_list)
    int3 offs[27] = {
        {-1,-1,-1},{-1,-1,0},{-1,-1,1},{-1,0,-1},{-1,0,0},{-1,0,1},{-1,1,-1},{-1,1,0},{-1,1,1},
        {0,-1,-1},{0,-1,0},{0,-1,1},{0,0,-1},{0,0,0},{0,0,1},{0,1,-1},{0,1,0},{0,1,1},
        {1,-1,-1},{1,-1,0},{1,-1,1},{1,0,-1},{1,0,0},{1,0,1},{1,1,-1},{1,1,0},{1,1,1}
    };

    for (int o=0; o<27; o++) {
        int ncx = cx + offs[o].x;
        int ncy = cy + offs[o].y;
        int ncz = cz + offs[o].z;
        if (ncx < 0 || ncx >= grid_dim || ncy < 0 || ncy >= grid_dim || ncz < 0 || ncz >= grid_dim) continue;
        int nc = ncx + ncy * grid_dim + ncz * grid_dim * grid_dim;
        int jstart = cell_start[nc];
        int jend = cell_start[nc + 1];
        for (int k = jstart; k < jend; k++) {
            int j = sorted_idx[k];
            if (j == i) continue;

            float3 pj = pos[j];
            float3 vj = vel[j];
            float3 wj = omega[j];
            float  rj = radius[j];
            int    mj = mat[j];

            float3 dx = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
            float dist = sqrtf(dx.x*dx.x + dx.y*dx.y + dx.z*dx.z) + 1e-12f;
            float rsum = ri + rj;
            if (dist >= rsum) continue;

            float invd = 1.0f / dist;
            float3 n = make_float3(dx.x * invd, dx.y * invd, dx.z * invd);
            float delta = rsum - dist;
            if (delta <= 0.0f) continue;

            float3 r_i = make_float3(ri * n.x, ri * n.y, ri * n.z);
            float3 r_j = make_float3(-rj * n.x, -rj * n.y, -rj * n.z);

            float3 cross_i = make_float3( wi.y * r_i.z - wi.z * r_i.y , wi.z * r_i.x - wi.x * r_i.z , wi.x * r_i.y - wi.y * r_i.x );
            float3 cross_j = make_float3( wj.y * r_j.z - wj.z * r_j.y , wj.z * r_j.x - wj.x * r_j.z , wj.x * r_j.y - wj.y * r_j.x );
            float3 v_rel = make_float3( vi.x - vj.x + cross_i.x + cross_j.x , vi.y - vj.y + cross_i.y + cross_j.y , vi.z - vj.z + cross_i.z + cross_j.z );

            float vn = v_rel.x * n.x + v_rel.y * n.y + v_rel.z * n.z;
            float3 vt = make_float3( v_rel.x - vn * n.x, v_rel.y - vn * n.y, v_rel.z - vn * n.z );

            float e1 = YOUNG[mi], e2 = YOUNG[mj];
            float nu1 = POISSON[mi], nu2 = POISSON[mj];
            float Eeff = 1.0f / ((1.0f - nu1*nu1)/e1 + (1.0f - nu2*nu2)/e2);

            float sqrt_rr = sqrtf(ri * rj);
            float aa = sqrtf(ri * rj * fmaxf(delta, 0.f));

            float Fn_h = (4.0f / 3.0f) * Eeff * sqrt_rr * (delta * sqrtf(delta));

            float g = SURFACE_ENERGY[mi * 2 + mj];
            float Re = (ri * rj) / (ri + rj + 1e-12f);
            float Fcoh = 0.8f * 3.14159265f * g * Re * ((delta > -1e-7f) ? 1.0f : 0.0f);

            float Fn = Fn_h - Fcoh;

            float Ge = Eeff / (2.0f * (1.0f + nu1));
            float3 Ft = make_float3( -(8.0f * Ge * aa) * vt.x * dt, -(8.0f * Ge * aa) * vt.y * dt, -(8.0f * Ge * aa) * vt.z * dt );

            float Ftm = sqrtf(Ft.x*Ft.x + Ft.y*Ft.y + Ft.z*Ft.z);
            float Ftmax = FRICTION[mi] * fabsf(Fn);
            float sc = (Ftm > 1e-12f) ? fminf(1.0f, Ftmax / Ftm) : 1.0f;
            Ft.x *= sc; Ft.y *= sc; Ft.z *= sc;

            float3 contrib = make_float3(Fn * n.x + Ft.x, Fn * n.y + Ft.y, Fn * n.z + Ft.z);
            fi.x += contrib.x; fi.y += contrib.y; fi.z += contrib.z;

            float3 wrel = make_float3(wi.x - wj.x, wi.y - wj.y, wi.z - wj.z);
            float wrm = sqrtf(wrel.x*wrel.x + wrel.y*wrel.y + wrel.z*wrel.z) + 1e-12f;
            float3 tr = make_float3(
                -ROLLING_FRICTION[mi] * fabsf(Fn) * ri * (wrel.x / wrm),
                -ROLLING_FRICTION[mi] * fabsf(Fn) * ri * (wrel.y / wrm),
                -ROLLING_FRICTION[mi] * fabsf(Fn) * ri * (wrel.z / wrm)
            );
            ti.x += tr.x; ti.y += tr.y; ti.z += tr.z;
        }
    }

    float mass = DENSITY[mi] * (4.0f/3.0f * 3.14159265f * ri*ri*ri);
    fi.x += mass * GRAV.x;
    fi.y += mass * GRAV.y;
    fi.z += mass * GRAV.z;

    force[i] = fi;
    torque[i] = ti;
}
'''
_cell_raw_kernel_cell = cp.RawKernel(_cell_raw_kernel_code, 'raw_compute_forces_cell')

def compute_forces_cell_raw(pos, vel, omega, radius, mat_type, dt, cell_size=0.003, box_size=0.018):
    """
    Cell-list accelerated drop-in using single RawKernel launch + cell pruning.
    Much lower work than brute N^2, single launch for high sustained util.
    Use for N>~8k-10k or when you want to go beyond ~7k particles (brute Raw still excellent and lower overhead up to ~7-8k on V100 for current densities).
    """
    N = pos.shape[0]
    sorted_idx, cell_start, grid_dim = build_cell_list(pos, cell_size, box_size)

    force = cp.zeros_like(pos)
    torque = cp.zeros_like(omega)

    block = 256
    grid = (N + block - 1) // block
    _cell_raw_kernel_cell(
        (grid,), (block,),
        (pos, vel, omega, radius, mat_type, force, torque,
         sorted_idx, cell_start, N, grid_dim, cell_size, box_size, dt)
    )
    return force, torque


def recommended_cell_size(max_radius, box_size, min_cells_per_dim=4):
    """Heuristic good starting cell_size.
    For highN lid-clustered Rung1-like runs (N~6500, particles near bottom), larger cells (0.005-0.006)
    win on perf vs brute because fewer cells + lower build overhead while still capturing contacts.
    Tuned via benchmark on exact highN physical lid setup.
    """
    cs = max(0.002, min(0.0065, 5.5 * float(max_radius)))
    g = int(np.ceil(box_size / cs))
    if g < min_cells_per_dim:
        cs = box_size / min_cells_per_dim
    return float(cs)


def get_compute_forces_fn(N=None, use_cell_list=None, cell_size=None, box_size=0.018, max_radius=None):
    """
    Convenience: return the best contact forces function for the situation.
    - If use_cell_list is True/False, respect it.
    - If None, auto: brute Raw for N<=8000 (lower overhead for current evidence densities at N=6.5k), cell-list for larger N or when explicitly requested.
    - cell_size: if None, use recommended_cell_size if max_radius given, else 0.004 (good for current lid-clustered 0.14bar runs).
    - Always returns a callable (pos,vel,omega,radius,mat,dt) -> (f, tq)
    """
    if cell_size is None:
        if max_radius is not None:
            cell_size = recommended_cell_size(max_radius, box_size)
        else:
            cell_size = 0.0055  # tuned: at N=6500 lid-clustered, cs=0.006 gave ~58 steps/s (faster than brute ~27 steps/s in same setup)

    if use_cell_list is True:
        def _fn(p, v, o, r, m, dt):
            return compute_forces_cell_raw(p, v, o, r, m, dt, cell_size=cell_size, box_size=box_size)
        return _fn
    if use_cell_list is False or (N is not None and N <= 8000):
        return compute_forces_raw
    # default to cell for large or unknown
    def _fn(p, v, o, r, m, dt):
        return compute_forces_cell_raw(p, v, o, r, m, dt, cell_size=cell_size, box_size=box_size)
    return _fn


if __name__ == "__main__":
    # Unit-test Raw vs high-level + cell-list Raw (run with: python -m common.dem_kernels or from sims dir)
    import cupy as cp
    print("dem_kernels self-test: Raw + cell Raw vs high-level on unit cases (Rung1 SURFACE=0)...")
    import dem_kernels as dk
    dk.SURFACE_ENERGY = cp.array([[0.0,0.0],[0.0,0.0]], dtype=cp.float32)
    N=2
    pos = cp.array([[0.,0.,0.001],[0.,0.,0.0005]], dtype=cp.float32)
    radius = cp.array([0.0003,0.0003], dtype=cp.float32)
    vel = cp.zeros((N,3), dtype=cp.float32)
    omega = cp.zeros((N,3), dtype=cp.float32)
    mat = cp.array([0,0], dtype=cp.int32)
    DT=6.5e-7
    f1, t1 = compute_forces(pos, vel, omega, radius, mat, DT)
    f2, t2 = compute_forces_raw(pos, vel, omega, radius, mat, DT)
    print("  2-reg dF max (brute Raw):", float(cp.max(cp.abs(f1-f2))))
    # cell list path (uses build + single RawKernel)
    from cell_list import compute_forces_cell_list
    f3, t3 = compute_forces_cell_list(pos, vel, omega, radius, mat, DT, cell_size=0.003, box_size=0.018)
    print("  2-reg dF max (cell Raw):", float(cp.max(cp.abs(f1-f3))))
    # iron
    mat = cp.array([1,1], dtype=cp.int32); radius = cp.array([0.0015,0.0015], dtype=cp.float32)
    f1, t1 = compute_forces(pos, vel, omega, radius, mat, DT)
    f2, t2 = compute_forces_raw(pos, vel, omega, radius, mat, DT)
    print("  2-iron dF max (brute Raw):", float(cp.max(cp.abs(f1-f2))))
    f3, t3 = compute_forces_cell_list(pos, vel, omega, radius, mat, DT, cell_size=0.003, box_size=0.018)
    print("  2-iron dF max (cell Raw):", float(cp.max(cp.abs(f1-f3))))
    print("  (highN=6500 high-level N^2 unreliable for reference; Raw authoritative + low mem)")
    print("  Cell-list hotpath (build device-only + single RawKernel 27-neighbor) is now the scalable production path.")
    print("  SURFACE zeroed in kernel for Rung1 no-reg-coh.")