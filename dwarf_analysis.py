#!/usr/bin/env python3
"""
dwarf_analysis.py
=================
All analysis helpers from the notebooks in one file, usable both as a module
(``import dwarf_analysis as da``) and as a command line tool on the HPC.

Command line example (run it inside a Slurm job, not on the login node):

    python dwarf_analysis.py /lustre/hinke/123456_100kpc --dt 100 --t-end 40 --boxlength 1000 --plot

Outputs (prefix = --out, default <run_dir>/dwarf_analysis):
    <prefix>.pkl          everything plotting_dwarf() returns (dict)
    <prefix>_series.npz   numeric time series (times, r, mass, energies, ...)
    <prefix>_summary.png  r(t), bound mass(t), orbital energy(t)   (with --plot)
    <run_dir>/com_check.mp4                                          (with --movie)

Units: positions kpc, velocities km/s, masses Msun, times Myr.
`boxlength` must be the box length of the simulation in kpc (namelist value).

Requires: numpy, scipy, matplotlib, pynbody (+ ffmpeg for --movie).
"""

import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")                      # no display on the cluster
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import pynbody as pn
import pynbody.analysis.halo               # noqa: F401  (pn.analysis.halo.*)
import pynbody.analysis.profile            # noqa: F401  (pn.analysis.profile.*)

import scipy.integrate as integrate
import scipy.special as sc
from scipy import interpolate
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.spatial import cKDTree
from scipy.special import gamma
from scipy.stats import gaussian_kde


# =============================================================================
# Constants (one consistent set for the whole file)
# =============================================================================
G = 4.30091e-3            # pc (km/s)^2 / Msun
a0_kms_pc = 3.703         # (km/s)^2 / pc   (= 1.2e-10 m/s^2)
G_kpc = G * 1e-3          # kpc (km/s)^2 / Msun
a0_kpc = a0_kms_pc * 1e3  # (km/s)^2 / kpc

# The host potential is anchored to phi = 0 at PHI_BOXSIZE/2 kpc (a stand-in for
# infinity).  This is NOT the simulation box: it is independent of --boxlength.
PHI_BOXSIZE = 2 * 1e6


# =============================================================================
# MOND helpers (pc based)
# =============================================================================
def nu_mond(y):
    y = np.maximum(np.asarray(y, dtype=float), 1e-30)
    return 0.5 + np.sqrt(0.25 + 1.0 / y)


nu = nu_mond


def v_inf(M):
    # returns km/s
    return (G * M * a0_kms_pc) ** (1 / 4)


def g_ext(R_kpc, M):
    R_pc = R_kpc * 1e3
    gN = G * M / R_pc**2
    return nu_mond(gN / a0_kms_pc) * gN


# --- Sersic profile -----------------------------------------------------------
def sersic_a(r_e, n):
    return r_e * np.exp(0.1789 - 0.6950 * n - n * np.log(n))


def sersic_p(n):
    return 1 - 0.6097 / n + 0.05563 / n**2


def sersic_mass(r, M, n, r_e):
    p = sersic_p(n)
    a = sersic_a(r_e, n)
    s = 3 * n - n * p
    ra = (r / a) ** (1 / n)
    return M * sc.gammainc(s, ra)


def sersic_rho_0(M, n, r_e):
    gam = gamma(3 * n - n * sersic_p(n))
    return M / (4 * np.pi * n * sersic_a(r_e, n) ** 3 * gam)


def sersic_density(r, M, n, r_e):
    ra = r / sersic_a(r_e, n)
    return sersic_rho_0(M, n, r_e) * (ra) ** (-sersic_p(n)) * np.exp(-(ra) ** (1 / n))


# --- accelerations ------------------------------------------------------------
def g_newton(r_pc, M_enc):
    """Newtonian g = G M / r^2  [(km/s)^2 pc^-1].  r in pc, M in Msun."""
    return G * np.asarray(M_enc, float) / np.asarray(r_pc, float) ** 2


def g_mond(r_pc, R_kpc, M_enc, M_host, efe=True):
    """
    MOND internal acceleration |g(r)| [(km/s)^2 pc^-1].

    r_pc  : internal radii array  [pc]
    R_kpc : galactocentric distance of satellite COM  [kpc]  (scalar)
    M_enc : enclosed satellite mass at each r_pc  [Msun]
    M_host: host mass enclosed within R_kpc  [Msun]  (scalar)
    efe   : toggle External Field Effect on/off
    """
    r_pc = np.asarray(r_pc, float)
    M_enc = np.asarray(M_enc, float)
    gN = g_newton(r_pc, M_enc)

    if not efe:
        # isolated MOND - no external field
        return nu_mond(gN / a0_kms_pc) * gN

    # EFE branch (QUMOND approximation)
    R_pc = R_kpc * 1e3
    gNe = g_newton(R_pc, M_host)   # scalar external Newtonian field

    strong = gN > gNe              # internal > external
    nu_arr = np.empty_like(gN)

    if strong.any():
        # internal-dominated: g close to isolated MOND, external is a correction
        y = gN[strong] / a0_kms_pc + gNe**2 / (3.0 * gN[strong] * a0_kms_pc)
        nu_arr[strong] = nu_mond(y)

    if (~strong).any():
        # external-dominated (deep EFE): Newtonian-like internally
        y = gNe / a0_kms_pc + gN[~strong] ** 2 / (3.0 * gNe * a0_kms_pc)
        nu_arr[~strong] = nu_mond(y)

    return nu_arr * gN


def g_host(R_kpc, M_host):
    """
    MOND acceleration of the host *alone* at galactocentric radius R.
    This is what sets the tidal field - it does not depend on efe.
    """
    R_pc = R_kpc * 1e3
    gNe = g_newton(R_pc, M_host)
    return nu_mond(gNe / a0_kms_pc) * gNe


def host_phi(pos_sat_part, M_host, n=2000, r_min=1e-3, r_max=1e6):
    """MOND potential of a point-mass host, evaluated at given positions."""
    r_kpc = np.linalg.norm(pos_sat_part, axis=1)

    r_grid_kpc = np.geomspace(r_min, r_max, n)
    r_grid_pc = r_grid_kpc * 1e3
    g_grid = g_host(r_grid_kpc, M_host)
    phi_cumul = cumulative_trapezoid(g_grid, r_grid_pc, initial=0)
    phi_grid = phi_cumul - phi_cumul[-1]

    return np.interp(r_kpc, r_grid_kpc, phi_grid, left=phi_grid[0], right=0.0)


def mond_potential_bound(r_array, mass_enc_array):
    """
    Potential for bound mass calculation (isolated MOND, r in pc).
    phi(r_max) = 0 boundary condition - only differences matter for KE+PE < 0.
    """
    r_array = np.maximum(np.asarray(r_array, dtype=float), 1.0)
    mass_enc_array = np.asarray(mass_enc_array, dtype=float)

    # efe=False -> R_kpc and M_host are not used
    g_arr = g_mond(r_array, 1.0, mass_enc_array, 1.0, efe=False)
    phi = cumulative_trapezoid(g_arr[::-1], r_array[::-1], initial=0)[::-1]
    return phi


# --- velocity dispersion from the Jeans equation (Sersic satellite) -----------
def sigma_integrand_newton(r_prime, M, n, r_e):
    r_arr = np.atleast_1d(float(r_prime))
    m_arr = np.atleast_1d(float(sersic_mass(r_prime, M, n, r_e)))
    return float(sersic_density(r_prime, M, n, r_e) * g_newton(r_arr, m_arr)[0])


def sigma_integrand_mond(r_prime, M, n, r_e, R, M_host):
    r_arr = np.atleast_1d(float(r_prime))
    m_arr = np.atleast_1d(float(sersic_mass(r_prime, M, n, r_e)))
    g = g_mond(r_arr, R_kpc=float(R), M_enc=m_arr, M_host=float(M_host), efe=False)[0]
    return float(sersic_density(r_prime, M, n, r_e) * g)


def sigma_integrand_efe(r_prime, M, n, r_e, R, M_host):
    r_arr = np.atleast_1d(float(r_prime))
    m_arr = np.atleast_1d(float(sersic_mass(r_prime, M, n, r_e)))
    g = g_mond(r_arr, R_kpc=float(R), M_enc=m_arr, M_host=float(M_host), efe=True)[0]
    return float(sersic_density(r_prime, M, n, r_e) * g)


def get_sigma(r, M, a_p, n, r_e, profile, R, M_host, r_cut=None, mond=True,
              n_table=10000, efe=False):
    if profile == "sersic":
        if r_cut is None:
            r_max_int = 50 * r_e
        else:
            r_max_int = r_cut

        # the grid must cover the integration range to avoid extrapolation errors
        r_grid_end = max(20 * r_e, r_max_int)
        r_grid = np.geomspace(1e-3 * r_e, r_grid_end, n_table)

        def sigma_sq_at_r(r0):
            if mond and efe:
                integral, _ = integrate.quad(
                    sigma_integrand_efe, r0, r_max_int,
                    args=(M, n, r_e, R, M_host), limit=200)
            elif mond:
                integral, _ = integrate.quad(
                    sigma_integrand_mond, r0, r_max_int,
                    args=(M, n, r_e, R, M_host), limit=200)
            else:
                integral, _ = integrate.quad(
                    sigma_integrand_newton, r0, r_max_int,
                    args=(M, n, r_e), limit=200)
            return integral / sersic_density(r0, M, n, r_e)

        sigma_grid = np.sqrt([sigma_sq_at_r(r0) for r0 in r_grid])
        sigma_interp = interpolate.interp1d(
            r_grid, sigma_grid, bounds_error=False,
            fill_value=(sigma_grid[0], sigma_grid[-1]))
        return sigma_interp(r)


def get_beta(v, r, R, M):
    # Brada 2000
    v = float(v)
    r = float(r) * 1e3
    R = float(R) * 1e3
    M = float(M)
    return (v**2 * R) / (v_inf(M) ** 2 * r)


def get_alpha(v, r, R, M):
    # Brada 2000
    v = float(v)
    r = float(r)
    R = float(R)
    M = float(M)
    return (v * R / (r * v_inf(M))) ** (2 / 3)


# =============================================================================
# Snapshot helpers
# =============================================================================
def mask_bound(snapshot, center, r_cut):
    pos = np.array(snapshot["pos"], dtype=float)
    center = np.asarray(center, dtype=float).flatten()
    r = np.sqrt(((pos - center) ** 2).sum(axis=1))
    return r < r_cut


def lagrangian_radius(snapshot, center, fraction=0.99):
    pos = snapshot["pos"]
    mass = snapshot["mass"]
    center = np.asarray(center, dtype=float).flatten()
    r = np.linalg.norm(pos - center, axis=1)
    sort_idx = np.argsort(r)
    m_cumul = np.cumsum(mass[sort_idx])
    return np.interp(fraction * m_cumul[-1], m_cumul, r[sort_idx])


def get_density_profile(snapshot, center, r_cut, nbins=300):
    center = np.asarray(center, dtype=float).flatten()
    snapshot['pos'] -= center
    p = pn.analysis.profile.Profile(snapshot, min=0.1, max=r_cut, ndim=3, nbins=nbins)
    return np.array(p['rbins']), np.array(p['density'])


def sigma_1d(v):
    return np.std(v)


def get_bound_center(snapshot, M_host, seed_pos, mask_ids, host_pos, r_J_init, efe=False,
                     r_edge_kpc=20, min_particles=50, max_iter=20, tol=0.01):
    """Iteratively find bound satellite particles."""
    pos_all = np.array(snapshot["pos"], dtype=float)                          # kpc
    mass_all = np.array(snapshot["mass"], dtype=float)                        # Msun
    vel_all = np.array(snapshot["vel"].in_units("km s^-1"), dtype=float)      # km/s

    pos_sat = pos_all[mask_ids]
    mass_sat = mass_all[mask_ids]
    vel_sat = vel_all[mask_ids]
    sat_global_idx = np.where(mask_ids)[0]

    if seed_pos is not None:
        com_pos = np.array(seed_pos, dtype=float)
    else:
        com_pos = np.average(pos_sat, weights=mass_sat, axis=0)

    com_vel = np.average(vel_sat, weights=mass_sat, axis=0)

    r_from_com = np.linalg.norm(pos_sat - com_pos, axis=1)   # kpc
    mask_b = r_from_com < r_J_init   # initial tidal radius as search region

    if mask_b.sum() < min_particles:
        print(f"  Too few seed particles ({mask_b.sum()}) "
              f"within r_J = {r_J_init:.2f} kpc")
        return None, None, None, None, None, None, False

    converged = False
    r_b_pc = None
    phi_parts = None

    for i in range(max_iter):
        if mask_b.sum() < min_particles:
            break

        pos_b = pos_sat[mask_b]
        vel_b = vel_sat[mask_b]
        mass_b = mass_sat[mask_b]

        R_sat_cur = float(np.linalg.norm(com_pos - host_pos))   # kpc

        rel_pos = pos_b - com_pos          # kpc
        rel_vel = vel_b - com_vel          # km/s
        r_b_pc = np.linalg.norm(rel_pos, axis=1) * 1e3   # pc

        sort_idx = np.argsort(r_b_pc)
        r_sorted = r_b_pc[sort_idx]
        m_cumul = np.cumsum(mass_b[sort_idx])

        r_min_pc = max(r_sorted[0], 1.0)
        r_max_pc = r_edge_kpc * 1e3
        r_grid = np.geomspace(r_min_pc, r_max_pc, 500)           # [pc]
        m_grid = np.interp(r_grid, r_sorted, m_cumul)

        g_arr = g_mond(r_grid, R_sat_cur, m_grid, M_host, efe)   # (km/s)^2/pc

        phi_cumul = cumulative_trapezoid(g_arr, r_grid, initial=0.0)
        phi_grid = phi_cumul - phi_cumul[-1]   # phi = 0 at outer boundary

        phi_parts = np.interp(r_b_pc, r_grid, phi_grid, left=phi_grid[0], right=0.0)

        KE = 0.5 * np.sum(rel_vel**2, axis=1)   # (km/s)^2
        new_bound = (KE + phi_parts) < 0.0

        m_new = float(np.sum(mass_b[new_bound]))
        m_old = float(np.sum(mass_b))

        bound_indices = np.where(mask_b)[0]
        mask_b[:] = False
        mask_b[bound_indices[new_bound]] = True

        if mask_b.sum() > 0:
            com_pos = np.average(pos_sat[mask_b], weights=mass_sat[mask_b], axis=0)
            com_vel = np.average(vel_sat[mask_b], weights=mass_sat[mask_b], axis=0)

        delta = abs(m_new - m_old) / (m_old + 1e-10)
        print(f"  [efe={efe}] iter {i+1:2d}: N={mask_b.sum():5d}  M={m_new:.3e} Msun")

        if delta < tol:
            converged = True
            break

    if mask_b.sum() >= min_particles:
        full_mask = np.zeros(len(pos_all), dtype=bool)
        full_mask[sat_global_idx[mask_b]] = True
        return (com_pos, com_vel, full_mask, float(np.sum(mass_all[full_mask])),
                r_b_pc, phi_parts, converged)
    return None, None, None, None, None, None, False


def find_remnants(snapshot, seed_pos, mask_ids, M_host, host_pos, r_J_init,
                  search_radius=10.0, min_particles=50, n_peaks=5, efe=False):
    """Find all locally bound structures in the tidal debris."""
    pos_all = np.array(snapshot["pos"], dtype=float)
    mass_all = np.array(snapshot["mass"], dtype=float)

    sat_global_idx = np.where(mask_ids)[0]   # indices into full array
    pos_sat = pos_all[mask_ids]
    mass_sat = mass_all[mask_ids]

    # density peak search on satellite particles only
    tree = cKDTree(pos_sat)
    counts = tree.query_ball_point(pos_sat, r=search_radius, return_length=True)

    remnants = []
    used_mask = np.zeros(len(pos_sat), dtype=bool)   # satellite-local (N_sat,)

    for _ in range(n_peaks):
        counts_masked = counts.copy().astype(float)
        counts_masked[used_mask] = 0

        if counts_masked.max() < min_particles:
            break

        peak_idx = np.argmax(counts_masked * mass_sat)
        seed = pos_sat[peak_idx]

        r_to_seed = np.linalg.norm(pos_sat - seed, axis=1)
        near_peak_local = r_to_seed < search_radius

        candidate_local = near_peak_local & ~used_mask
        if candidate_local.sum() < min_particles:
            used_mask[peak_idx] = True
            continue

        candidate_global = np.zeros(len(pos_all), dtype=bool)
        candidate_global[sat_global_idx[candidate_local]] = True

        com_pos, com_vel, full_mask, mass_bound, r_b, phi_parts, converged = get_bound_center(
            snapshot,
            M_host=M_host,
            seed_pos=seed_pos,
            mask_ids=candidate_global,
            host_pos=host_pos,
            r_J_init=r_J_init,
            min_particles=min_particles,
            efe=efe)

        if com_pos is None:
            used_mask[peak_idx] = True   # skip this peak, try next
            continue

        local_bound = full_mask[sat_global_idx]   # (N_sat,) bool
        used_mask |= local_bound

        remnants.append({
            "com_pos": com_pos,
            "com_vel": com_vel,
            "mask": full_mask,
            "mass": mass_bound,
            "converged": converged,
            "N_bound": full_mask.sum(),
        })
        print(f"  Remnant {len(remnants)}: M={mass_bound:.3e}, "
              f"N={full_mask.sum()}, pos={com_pos[:2]}")

    return remnants, used_mask


def velocity_dispersion_profile(pos, vel, com_pos, com_vel, rbins, min_particles=20,
                                los_dir=None, L_orb=None):
    """
    Radial/tangential/LOS velocity dispersion profiles about (com_pos, com_vel).

    L_orb : orbital angular momentum vector of the satellite ABOUT THE HOST
            (np.cross(r_host->sat, v_host->sat)).  It defines the in-plane /
            out-of-plane tangential directions.  If None it falls back to
            np.cross(com_pos, com_vel), which is only meaningful when com_pos and
            com_vel are host-centred - not when they are internal offsets.
    """
    if los_dir is None:
        los_dir = np.array([0.0, 0.0, 1.0])
    else:
        los_dir = los_dir / np.linalg.norm(los_dir)

    rel_pos = pos - com_pos
    rel_vel = vel - com_vel
    r = np.linalg.norm(rel_pos, axis=1)

    if L_orb is None:
        L_orb = np.cross(com_pos, com_vel)
    L_norm = np.linalg.norm(L_orb)
    if L_norm > 0:
        L_hat = L_orb / L_norm
    else:
        L_hat = np.array([0.0, 0.0, 1.0])   # degenerate: fall back to z axis

    results = {'r_centers': [], 'sigma_los': [], 'sigma_r': [],
               'sigma_t1': [], 'sigma_t2': [], 'beta': [], 'n_particles': []}

    for j in range(len(rbins) - 1):
        r_in = rbins[j]
        r_out = rbins[j + 1]
        shell = (r >= r_in) & (r < r_out)
        count = shell.sum()

        if count < min_particles:
            results['r_centers'].append(np.sqrt(r_in * r_out))
            for k in ['sigma_los', 'sigma_r', 'sigma_t1', 'sigma_t2', 'beta']:
                results[k].append(np.nan)
            results['n_particles'].append(count)
            continue

        r_shell = r[shell]
        rel_pos_shell = rel_pos[shell]
        rel_vel_shell = rel_vel[shell]

        r_hat = rel_pos_shell / np.maximum(r_shell, 1e-10)[:, None]
        v_r = np.sum(rel_vel_shell * r_hat, axis=1)
        sigma_r = np.std(v_r)

        L_along_r = np.sum(L_hat * r_hat, axis=1)[:, None] * r_hat   # L_hat component along r_hat
        t2_hat = L_hat - L_along_r                                   # perpendicular to r_hat
        t2_norm = np.linalg.norm(t2_hat, axis=1, keepdims=True)
        t2_hat = t2_hat / np.maximum(t2_norm, 1e-10)

        v_t2 = np.sum(rel_vel_shell * t2_hat, axis=1)                # out-of-plane
        sigma_t2 = np.std(v_t2)

        t1_hat = np.cross(t2_hat, r_hat)                             # perp to L_hat and r_hat
        t1_norm = np.linalg.norm(t1_hat, axis=1, keepdims=True)
        t1_hat = t1_hat / np.maximum(t1_norm, 1e-10)

        v_t1 = np.sum(rel_vel_shell * t1_hat, axis=1)                # in-plane azimuthal
        sigma_t1 = np.std(v_t1)
        beta_vel = 1.0 - (sigma_t1**2 + sigma_t2**2) / (2 * sigma_r**2 + 1e-10)

        sigma_los = np.std(rel_vel_shell @ los_dir)

        results['r_centers'].append(np.sqrt(r_in * r_out))
        results['sigma_los'].append(sigma_los)
        results['sigma_r'].append(sigma_r)
        results['sigma_t1'].append(sigma_t1)
        results['sigma_t2'].append(sigma_t2)
        results['beta'].append(beta_vel)
        results['n_particles'].append(count)

    return (np.array(results['r_centers']),
            np.array(results['sigma_los']),
            np.array(results['sigma_r']),
            np.array(results['sigma_t1']),
            np.array(results['sigma_t2']),
            np.array(results['beta']),
            np.array(results['n_particles']))


def tidal_radius_mond(R_kpc, m_sat, M_host, efe=True):
    """Zhao 2006 Roche/tidal radius:  r1 / D0 = (m / zeta1 M)^(1/3),  zeta1 = 1 + zeta."""
    drel = 1e-3
    R_lo, R_hi = R_kpc * (1 - drel), R_kpc * (1 + drel)

    g_lo = g_host(R_lo, M_host)
    g_hi = g_host(R_hi, M_host)
    dlng_dlnr = (np.log(g_hi) - np.log(g_lo)) / (np.log(R_hi) - np.log(R_lo))

    zeta = -dlng_dlnr
    zeta1 = 1.0 + zeta
    r_J = R_kpc * (m_sat / (zeta1 * M_host)) ** (1.0 / 3.0)
    return float(r_J)


def tidal_force(M_host, m_sat, r_sat_size, D_orbit):
    """
    Bellazzini et al. 1996 dimensionless tidal force
    M_host     : host mass enclosed within D [Msun]
    m_sat      : satellite mass [Msun]
    r_sat_size : satellite size (e.g. half-mass radius) [kpc]
    D_orbit    : orbital radius [kpc]
    """
    return (M_host / m_sat) * (r_sat_size / D_orbit) ** 3


def get_angular_momentum(pos_part, vel_part, mass_part, phi_MW):
    """Specific angular momentum of each particle in the host frame."""
    rel_pos = pos_part    # kpc
    rel_vel = vel_part    # km/s

    L = np.cross(rel_pos, rel_vel)          # (N, 3) kpc km/s
    L_mag = np.linalg.norm(L, axis=1)       # (N,)
    Lz = L[:, 2]

    E_orb = 0.5 * np.linalg.norm(rel_vel, axis=1) ** 2 + phi_MW
    return L, L_mag, Lz, E_orb


def find_streams(Lz_sat, E_sat, bandwidth=0.15, smooth_sigma=3, min_height=0.1,
                 min_distance=20):
    """Find peaks in 2D (Lz, E) space.  Returns peaks, densities, particle masks."""
    n_grid = 100
    Lz_min, Lz_max = Lz_sat.min(), Lz_sat.max()
    E_min, E_max = E_sat.min(), E_sat.max()

    Lz_pad = 0.1 * (Lz_max - Lz_min)
    E_pad = 0.1 * (E_max - E_min)

    Lz_grid = np.linspace(Lz_min - Lz_pad, Lz_max + Lz_pad, n_grid)
    E_grid = np.linspace(E_min - E_pad, E_max + E_pad, n_grid)
    Lz_mesh, E_mesh = np.meshgrid(Lz_grid, E_grid)

    points = np.vstack([Lz_sat, E_sat])
    kde = gaussian_kde(points, bw_method=0.3)

    density_2d = kde(np.vstack([Lz_mesh.ravel(), E_mesh.ravel()])).reshape(n_grid, n_grid)
    density_smooth = gaussian_filter(density_2d, sigma=smooth_sigma)

    local_max_mask = ((density_smooth == maximum_filter(density_smooth, size=5))
                      & (density_smooth > min_height * density_smooth.max()))
    peak_idx = np.argwhere(local_max_mask)

    bw_L = kde.factor * np.std(Lz_sat)
    bw_E = kde.factor * np.std(E_sat)

    if len(peak_idx) > 0:
        peak_e_values = E_grid[peak_idx[:, 0]]     # row = E axis
        peak_lz_values = Lz_grid[peak_idx[:, 1]]   # col = Lz axis
        peak_densities = density_smooth[peak_idx[:, 0], peak_idx[:, 1]]

        dist_Lz = np.abs(Lz_sat[:, np.newaxis] - peak_lz_values[np.newaxis, :])
        dist_E = np.abs(E_sat[:, np.newaxis] - peak_e_values[np.newaxis, :])
        dist_to_peaks = np.sqrt((dist_Lz / bw_L) ** 2 + (dist_E / bw_E) ** 2)

        nearest_peak = np.argmin(dist_to_peaks, axis=1)
        min_dist = dist_to_peaks[np.arange(len(Lz_sat)), nearest_peak]

        assigned = min_dist < 5.0   # only assign if within 5*bw of a peak

        masks = []
        for i in range(len(peak_idx)):
            masks.append((nearest_peak == i) & assigned)
    else:
        peak_lz_values = np.array([])
        peak_e_values = np.array([])
        peak_densities = np.array([])
        masks = []

    return (peak_lz_values, peak_e_values, peak_densities, masks,
            Lz_grid, E_grid, density_smooth)


# =============================================================================
# Milky Way (host) set-up  (kpc based)
# =============================================================================
def f(r, rd):
    return 1 - np.exp(-r / rd) * (1 + r / rd)


def df_dr(r, rd):
    return r / (rd**2) * np.exp(-r / rd)


def alpha_ana(M_tot, r):
    return G_kpc * M_tot / (a0_kpc * r**2)


def nu_mond_ana(M_tot, r, rd):
    y = alpha_ana(M_tot, r) * f(r, rd)
    return 0.5 + np.sqrt(0.25 + 1.0 / y)


def dnu_dy(M_tot, r, rd):
    y = alpha_ana(M_tot, r) * f(r, rd)
    return -1 / (2 * y**2 * np.sqrt(1 / 4 + 1 / y))


def dnu_dr(M_tot, r, rd):
    dndy = dnu_dy(M_tot, r, rd)
    a = alpha_ana(M_tot, r)
    dfdr = df_dr(r, rd)
    return dndy * a * (dfdr - 2 * f(r, rd) / r)


def dMc_dr(M_tot, r, rd):
    dfdr = df_dr(r, rd)
    return M_tot * (f(r, rd) * dnu_dr(M_tot, r, rd) + nu_mond_ana(M_tot, r, rd) * dfdr)


def rho_ana_MW(M_tot, r, rd):
    dM_dr = dMc_dr(M_tot, r, rd)
    return 1 / (4 * np.pi * r**2) * dM_dr


def mw_bary_mass_encl(r, M_MW=9.15e10, f_inner=0.8236, f_outer=0.1764,
                      Rd_inner=1.29 / 0.6, Rd_outer=4.20 / 0.6):
    """Baryonic enclosed MW mass (Banik et al. 2022) of host at radius r [kpc]."""
    M_inner = f_inner * M_MW      # 7.54e10 Msun
    M_outer = f_outer * M_MW      # 1.61e10 Msun

    M_b_inner = M_inner * f(r, Rd_inner)
    M_b_outer = M_outer * f(r, Rd_outer)
    return M_b_inner + M_b_outer


def g_mond_host(R_kpc, M_host_enc, a0=a0_kpc):
    """MOND acceleration of host at orbital radius R [kpc]."""
    gN = G_kpc * M_host_enc / R_kpc**2
    return gN * nu(gN / a0)   # (km/s)^2/kpc


def G_eff(g_c, a0=a0_kpc):
    """Effective G in EFE-dominated regime [same units as G_kpc]"""
    mu_gc = g_c / (g_c + a0)
    return G_kpc / mu_gc   # = G*(g_c+a0)/g_c


def sqrt_dlng_dlngN(g_c, a0=a0_kpc):
    """Simplified: sqrt((g_c+a0)/(g_c+2*a0))."""
    return np.sqrt((g_c + a0) / (g_c + 2 * a0))


def alpha_slope(R_kpc, M_host_func, dR=None):
    """
    Logarithmic slope alpha = -d ln g_c / d ln R (finite differences of the host
    acceleration profile).  M_host_func(R) returns the enclosed host mass [Msun].
    """
    if dR is None:
        dR = R_kpc * 0.01
    R1, R2 = R_kpc - dR, R_kpc + dR
    g1 = g_mond_host(R1, M_host_func(R1))
    g2 = g_mond_host(R2, M_host_func(R2))
    dlng = np.log(g2) - np.log(g1)
    dlnR = np.log(R2) - np.log(R1)
    return -dlng / dlnR   # positive for declining acceleration


def dg_dR(R_kpc, M_host_func, dR=None):
    """Tidal field Delta g_c / Delta R [(km/s)^2/kpc^2]."""
    if dR is None:
        dR = R_kpc * 0.01
    g1 = g_mond_host(R_kpc - dR, M_host_func(R_kpc - dR))
    g2 = g_mond_host(R_kpc + dR, M_host_func(R_kpc + dR))
    return (g2 - g1) / (2 * dR)   # negative (acceleration decreases outward)


def tidal_radius_asencio(R_kpc, M_dwarf, M_host_func, host=True, a0=a0_kpc):
    """
    MOND tidal radius from Asencio et al. 2022 / Zhao & Tian 2006.

    R_kpc      : orbital radius of dwarf [kpc]
    M_dwarf    : dwarf total mass [Msun]
    M_host_func: callable, M_host_enc(R) in Msun
    host=False : return a fixed 50 kpc (isolated dwarf, no host)
    """
    g_c = g_mond_host(R_kpc, M_host_func(R_kpc), a0)
    sqrt_factor = sqrt_dlng_dlngN(g_c, a0)
    Geff = G_eff(g_c, a0)
    alpha = alpha_slope(R_kpc, M_host_func)
    tidal_field = abs(dg_dR(R_kpc, M_host_func))   # (km/s)^2/kpc^2

    bracket = (2 - alpha) / (3 - alpha) * Geff * M_dwarf / tidal_field
    r_tid = (2 / 3) * sqrt_factor * bracket ** (1 / 3)
    if host:
        return r_tid
    return 50.0


def host_potential(M_host_func, boxsize, a0=a0_kpc):
    """
    MOND potential of the Milky Way at radius r [kpc], phi(boxsize/2) = 0.
    M_host_func: callable, returns enclosed host mass at r [kpc]
    Returns an interpolator: phi_host(R) with R in kpc.
    """
    r_grid = np.geomspace(1e-3, boxsize / 2, 5000)   # kpc
    g_arr = g_mond_host(r_grid, M_host_func(r_grid), a0)

    phi_cumul = cumulative_trapezoid(g_arr, r_grid, initial=0)
    phi_total = phi_cumul[-1]
    phi_arr = phi_cumul - phi_total   # negative everywhere, 0 at r_max

    return UnivariateSpline(r_grid, phi_arr, s=0, k=3, ext=3)


# =============================================================================
# Main analysis loop over the pynbody snapshots
# =============================================================================
RESULT_NAMES = [
    "times", "all_pos", "all_vel", "all_mass", "coms_sat", "rs", "sigmas",
    "masses_sat", "rhos_sat", "bins_sat", "hmr_sat", "lag_sat", "all_remnants",
    "alphas", "betas", "r_centers", "tot_sigmas_bin", "rb_array", "tidals",
    "jacobs", "anisotropy", "orbital_energies", "ekin", "epot", "l", "lz",
]


def plotting_dwarf(filename, dt, t_end, r_e, x0, y0, vx, vy, boxlength, grid_length,
                   host=True, movie=True, cutoff_sat=20, phi_boxsize=PHI_BOXSIZE,
                   ffmpeg_path=None, verbose=True):
    """
    Read outputs 1..t_end from directory `filename` and measure the dwarf.

    filename   : run directory that contains output_00001, output_00002, ...
    dt         : time between outputs [Myr]
    boxlength  : simulation box length [kpc]  (host sits at boxlength/2)
    r_e, y0, vx: unused, kept so existing calls keep working
    x0, vy     : only used in the movie title

    Returns the 26-tuple listed in RESULT_NAMES.  All per-snapshot lists have
    exactly one entry per snapshot (NaN entries for snapshots without bound
    particles).
    """
    log = print if verbose else (lambda *a, **k: None)

    coms_sat = []
    l = []
    lz = []
    all_pos = []
    all_mass = []
    all_vel = []
    pos_bound = []
    times = []
    t = 0
    bins_sat = []
    rhos_sat = []
    masses_sat = []
    tot_sigmas_bin = []
    lag_sat = []
    hmr_sat = []
    alphas = []
    betas = []
    tidals = []
    jacobs = []
    fs = []
    rs = []
    r_efe = []
    sigmas = []
    rb_array = []
    v0 = vy
    sep = x0
    all_remnants = []
    anisotropy = []
    orbital_energies = []
    ekin = []
    epot = []
    galacto_centre = []
    r_centers = None

    # host potential, computed ONCE for the whole run
    phi_host = host_potential(mw_bary_mass_encl, phi_boxsize)

    for i in range(1, t_end + 1):
        path = os.path.join(filename, f"output_{i:05d}")
        s = pn.load(path)

        # ---- galactocentric quantities BEFORE centering ------------------------
        pos_raw = np.array(s.dm['pos'], dtype=float)
        vel_raw = np.array(s.dm['vel'].in_units('km s^-1'), dtype=float)
        mass_raw = np.array(s.dm['mass'], dtype=float)
        box_centre = np.array([boxlength / 2, boxlength / 2, boxlength / 2])

        all_pos.append(pos_raw - box_centre)

        # robust dwarf centre (shrinking sphere) - not biased by tidal debris
        com_box = np.array(pn.analysis.halo.shrink_sphere_center(s.dm, shrink_factor=0.9))
        r_galacto = float(np.linalg.norm(com_box - box_centre))
        galacto_centre.append(np.array(com_box - box_centre))

        # velocity of the dwarf: average over the core only
        r_from_com = np.linalg.norm(pos_raw - com_box, axis=1)
        core_mask = r_from_com < 1
        if core_mask.sum() == 0:
            core_mask = r_from_com < np.percentile(r_from_com, 5)   # fallback: nearest 5%
        com_vel = np.average(vel_raw[core_mask], weights=mass_raw[core_mask], axis=0)
        v_galacto = float(np.linalg.norm(com_vel))

        log(f"  snap {i}: r_galacto = {r_galacto:.2f} kpc, v_galacto = {v_galacto:.2f} km/s")

        encl_mass_host = mw_bary_mass_encl(r_galacto)
        r_J = tidal_radius_asencio(r_galacto, np.sum(mass_raw), mw_bary_mass_encl, host=host)

        # dwarf-centred coordinates / velocities
        pos_current = pos_raw - com_box
        vel_kms = vel_raw - com_vel
        mass_current = mass_raw

        # orbital angular momentum about the host, defines the tangential directions
        L_orbit = np.cross(com_box - box_centre, com_vel)

        # anisotropy profile on all particles (already dwarf-centred -> zero offsets)
        rbins_prof = np.logspace(np.log10(0.05), np.log10(cutoff_sat), 25)
        _, _, _, _, _, beta, _ = velocity_dispersion_profile(
            pos_current, vel_kms, np.zeros(3), np.zeros(3), rbins_prof, L_orb=L_orbit)
        anisotropy.append(beta)

        # ---- particles within the tidal radius ---------------------------------
        r_all = np.linalg.norm(pos_current, axis=1)
        sat_mask_local = r_all < r_J
        log(f"  {sat_mask_local.sum()} particles within r_J = {r_J:.2f} kpc")

        pos_sat = pos_current[sat_mask_local].copy()
        vel_sat = vel_kms[sat_mask_local].copy()
        mass_sat = mass_current[sat_mask_local].copy()
        mass_bound = float(np.sum(mass_sat))

        jacobs.append(r_J)
        rs.append(r_galacto)

        # ---- empty branch: keep every list aligned with `times` -----------------
        if len(pos_sat) == 0:
            log(f"  i={i}: No bound particles")
            pos_bound.append(pos_sat)
            coms_sat.append(np.full(3, np.nan))
            masses_sat.append(0.0)
            r_efe.append(np.nan)
            lag_sat.append(np.nan)
            hmr_sat.append(np.nan)
            fs.append(np.nan)
            sigmas.append(np.nan)
            bins_sat.append(np.array([]))
            rhos_sat.append(np.array([]))
            tot_sigmas_bin.append({"los": np.nan, "r": np.nan, "t1": np.nan, "t2": np.nan})
            alphas.append(np.nan)
            betas.append(np.nan)
            l.append(np.full(3, np.nan))
            lz.append(np.nan)
            ekin.append(np.nan)
            epot.append(np.nan)
            orbital_energies.append(np.nan)
            times.append(t)
            t += dt
            continue

        # ---- bound particle quantities ------------------------------------------
        com_sat = np.average(pos_sat, weights=mass_sat, axis=0)
        com_vel_sat = np.average(vel_sat, weights=mass_sat, axis=0)
        masses_sat.append(mass_bound)
        r_orbit = com_sat + com_box - box_centre   # bound COM, relative to host
        coms_sat.append(r_orbit)
        v_orbit = com_vel
        L = np.cross(r_orbit, v_orbit)             # 3d vector
        Lz = L[2]
        l.append(L)
        lz.append(Lz)

        # radii of the bound particles only
        r_bound = np.linalg.norm(pos_sat, axis=1)
        sort_idx = np.argsort(r_bound)
        m_cumul = np.cumsum(mass_sat[sort_idx])
        m_total = m_cumul[-1]
        r50 = np.interp(0.50 * m_total, m_cumul, r_bound[sort_idx])
        r99 = np.interp(0.99 * m_total, m_cumul, r_bound[sort_idx])

        lag_sat.append(r99)
        hmr_sat.append(r50)
        pos_bound.append(pos_sat)

        r_efe.append(r_galacto * np.sqrt(mass_bound / encl_mass_host))

        bin_h2, dens_h2 = get_density_profile(
            s.dm[sat_mask_local], center=com_sat + com_box, r_cut=cutoff_sat)
        bins_sat.append(bin_h2)
        rhos_sat.append(dens_h2)

        vx_sat, vy_sat, vz_sat = vel_sat[:, 0], vel_sat[:, 1], vel_sat[:, 2]
        sig_tot = float(np.sqrt(sigma_1d(vx_sat) ** 2 +
                                sigma_1d(vy_sat) ** 2 +
                                sigma_1d(vz_sat) ** 2))
        sigmas.append(sig_tot)

        f_tid = tidal_force(encl_mass_host, mass_bound, r99, r_galacto)
        fs.append(f_tid)

        r_centers, sigma_los, sig_r, sig_t1, sig_t2, beta, n_prof = velocity_dispersion_profile(
            pos_sat, vel_sat, com_sat, com_vel_sat, rbins_prof, L_orb=L_orbit)
        tot_sigmas_bin.append({"los": sigma_los, "r": sig_r,
                               "t1": sig_t1, "t2": sig_t2, "beta": beta})

        alpha = get_alpha(sig_tot, r99, r_galacto, encl_mass_host)
        beta = get_beta(sig_tot, r99, r_galacto, encl_mass_host)
        alphas.append(float(alpha))
        betas.append(float(beta))

        # ---- orbital energy per unit mass ---------------------------------------
        KE_orb = 0.5 * v_galacto**2                       # (km/s)^2
        phi_orb = phi_host(np.linalg.norm(r_orbit))
        ekin.append(KE_orb)
        epot.append(float(phi_orb))
        orbital_energies.append(KE_orb + float(phi_orb))

        times.append(t)
        t += dt

    if movie:
        _make_movie(filename, times, all_pos, coms_sat, pos_bound, galacto_centre,
                    masses_sat, grid_length, v0, sep, ffmpeg_path, log)

    return (
        np.array(times), all_pos, all_vel, all_mass, np.array(coms_sat),
        np.array(rs), np.array(sigmas), masses_sat, rhos_sat, bins_sat, hmr_sat,
        lag_sat, all_remnants, alphas, betas, r_centers, tot_sigmas_bin, rb_array,
        tidals, jacobs, anisotropy, orbital_energies, ekin, epot, l, lz,
    )


def _make_movie(filename, times, all_pos, coms_sat, pos_bound, galacto_centre,
                masses_sat, grid_length, v0, sep, ffmpeg_path, log):
    if ffmpeg_path:
        matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg_path
    log("Setting up movie...")
    fig, ax = plt.subplots(figsize=(10, 10))
    scat_s_particles = ax.scatter([], [], s=1, alpha=0.2, c='blue', label='Dwarf particles')
    scat_particles = ax.scatter([], [], s=1, alpha=0.2, c='gray', label='All particles')
    scat_com_sat = ax.scatter([], [], s=30, marker='x', c='blue', label='COM Satellite',
                              zorder=10)

    ax.set_xlim(-grid_length / 2, grid_length / 2)
    ax.set_ylim(-grid_length / 2, grid_length / 2)
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Y [kpc]')
    ax.grid(True, axis='x', alpha=0.3)

    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, color='black',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.4))

    def update(frame):
        scat_particles.set_offsets(all_pos[frame][:, :2])

        if len(pos_bound[frame]) > 0:
            # pos_bound is dwarf-centred -> shift into the galactocentric frame
            xy = np.array(pos_bound[frame])[:, :2] + np.asarray(galacto_centre[frame])[:2]
            scat_s_particles.set_offsets(xy)
        else:
            scat_s_particles.set_offsets(np.empty((0, 2)))

        com_s_xy = np.array(coms_sat[frame], dtype=float)[:2].reshape(1, -1)
        scat_com_sat.set_offsets(com_s_xy)

        time_text.set_text(f't = {times[frame]:.0f} Myr')
        ax.set_title(f'Dwarf mass: {masses_sat[frame]:.2e} M_sol, $v_y$={v0} $v_c$, sep={sep} kpc')
        return (scat_s_particles, scat_particles, scat_com_sat)

    ax.legend(loc='upper right')
    anim = FuncAnimation(fig, update, frames=len(times), interval=500, blit=True)
    out = os.path.join(filename, "com_check.mp4")
    anim.save(out, fps=2, dpi=150, extra_args=['-vcodec', 'libx264'])
    log(f"Movie saved as {out}")
    plt.close(fig)


# =============================================================================
# Convenience: results dict, saving, summary plot
# =============================================================================
def results_to_dict(res):
    return dict(zip(RESULT_NAMES, res))


def save_results(res, prefix, save_particles=False):
    d = results_to_dict(res)
    if not save_particles:
        d["all_pos"] = None    # can be several GB; use --save-particles to keep it
    with open(prefix + ".pkl", "wb") as fh:
        pickle.dump(d, fh)

    series = {
        "times": np.asarray(d["times"], float),
        "r": np.asarray(d["rs"], float),
        "mass_bound": np.asarray(d["masses_sat"], float),
        "sigma": np.asarray(d["sigmas"], float),
        "r50": np.asarray(d["hmr_sat"], float),
        "r99": np.asarray(d["lag_sat"], float),
        "r_tidal": np.asarray(d["jacobs"], float),
        "alpha": np.asarray(d["alphas"], float),
        "beta": np.asarray(d["betas"], float),
        "ekin": np.asarray(d["ekin"], float),
        "epot": np.asarray(d["epot"], float),
        "etot": np.asarray(d["orbital_energies"], float),
        "L": np.asarray(d["l"], float),
        "Lz": np.asarray(d["lz"], float),
        "pos_com": np.asarray(d["coms_sat"], float),
    }
    np.savez(prefix + "_series.npz", **series)
    return prefix + ".pkl", prefix + "_series.npz"


def make_summary_plot(res, path):
    d = results_to_dict(res)
    t = np.asarray(d["times"], float)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t, d["rs"])
    axes[0].set_ylabel("r [kpc]")
    axes[1].plot(t, d["masses_sat"])
    axes[1].set_ylabel("Mass [M_sol]")
    axes[2].plot(t, d["epot"], label="PE")
    axes[2].plot(t, d["ekin"], label="KE")
    axes[2].plot(t, d["orbital_energies"], color="red", label="KE + PE")
    axes[2].axhline(0, ls="--", alpha=0.3)
    axes[2].set_ylabel("E [(km/s)$^2$]")
    axes[2].set_xlabel("t [Myr]")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# =============================================================================
# Command line interface
# =============================================================================
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Analyse a dwarf-galaxy run (orbit, bound mass, dispersions, energy).")
    p.add_argument("run_dir", help="directory containing output_00001, output_00002, ...")
    p.add_argument("--dt", type=float, required=True, help="time between outputs [Myr]")
    p.add_argument("--t-end", type=int, required=True, help="last output number to read")
    p.add_argument("--boxlength", type=float, required=True,
                   help="simulation box length [kpc] (namelist value)")
    p.add_argument("--grid-length", type=float, default=400.0,
                   help="plot window of the movie [kpc] (default 400)")
    p.add_argument("--cutoff-sat", type=float, default=20.0,
                   help="outer radius of the satellite profiles [kpc] (default 20)")
    p.add_argument("--no-host", action="store_true",
                   help="dwarf without host: fixed 50 kpc tidal radius")
    p.add_argument("--x0", type=float, default=None, help="initial separation (movie title only)")
    p.add_argument("--vy", type=float, default=None, help="initial v_y / v_c (movie title only)")
    p.add_argument("--movie", action="store_true", help="also write com_check.mp4 (needs ffmpeg)")
    p.add_argument("--ffmpeg", default=None, help="path to the ffmpeg executable")
    p.add_argument("--plot", action="store_true", help="write the r / mass / energy summary png")
    p.add_argument("--save-particles", action="store_true",
                   help="keep all particle positions in the pickle (large!)")
    p.add_argument("--out", default=None,
                   help="output prefix (default <run_dir>/dwarf_analysis)")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    prefix = a.out or os.path.join(a.run_dir, "dwarf_analysis")

    res = plotting_dwarf(
        a.run_dir, a.dt, a.t_end, None, a.x0, None, None, a.vy,
        a.boxlength, a.grid_length, host=not a.no_host, movie=a.movie,
        cutoff_sat=a.cutoff_sat, ffmpeg_path=a.ffmpeg, verbose=not a.quiet)

    pkl, npz = save_results(res, prefix, save_particles=a.save_particles)
    print(f"Saved {pkl}")
    print(f"Saved {npz}")
    if a.plot:
        print(f"Saved {make_summary_plot(res, prefix + '_summary.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
