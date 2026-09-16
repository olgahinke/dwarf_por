"""
Minimal pure-Python reader for Phantom-of-RAMSES amr_*/grav_* output files.

Translated DIRECTLY, record-for-record, from grav2ascii.f90 (the Fortran
source you've been debugging this session) -- not from generic RAMSES
documentation. This matters because your grav_* files have a non-standard,
MOND-specific layout (each cell stores a Newtonian potential+force block
*and* a MOND potential+force block), which generic tools like yt/pynbody
don't know how to parse.

ASSUMPTIONS -- read before trusting the output:
1. Single-CPU output (ncpu == 1). Your file naming (*.out00001 only) suggests
   this is true for your test runs. If you ever have ncpu > 1, this script
   as written only reads CPU-file #1's own grids and will silently miss
   grids owned by other CPUs -- do NOT use it on a multi-CPU run without
   first re-adding the Hilbert domain-decomposition logic from the original
   Fortran (the `ordering == 'hilbert'` block).
2. `ordering` (from info_*.txt) is NOT 'bisection'. If yours is, the
   AMR-header skip count differs (5 records instead of 1) -- see the
   NOTE comment at that line below.
3. Integer = 4-byte, boxlen (legacy scalar) = 4-byte real, everything else
   physical = 8-byte real -- matching the explicit Fortran declarations.

VALIDATE BEFORE TRUSTING: compare this script's output for one cell against
your (now-fixed) Fortran grav2ascii for the same output, at the same
position, before relying on this for real analysis. Record-skip counts in
undocumented legacy binary formats are exactly the kind of thing that looks
plausible but is subtly wrong.
"""

from scipy.io import FortranFile
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):   # no-op fallback if tqdm isn't installed
        return iterable


def _cell_offsets(ind, dx):
    """Reproduce xc(ind,1:3) from the Fortran ind -> (ix,iy,iz) unpacking."""
    iz = ind // 4
    iy = (ind - 4 * iz) // 2
    ix = ind - 2 * iy - 4 * iz
    return np.array([(ix - 0.5) * dx, (iy - 0.5) * dx, (iz - 0.5) * dx])


def read_amr_grav(repository, mond=True, icpu=1, ordering_is_bisection=False):
    """
    repository: path to an output_XXXXX directory
    mond: True if grav files contain the doubled Newton+MOND blocks
    Returns an (N,13) array matching grav_out from the Fortran:
      [x, y, z, dx, potential, fx, fy, fz, ilevel,
       potential_other, fx_other, fy_other, fz_other]
    where "_other" is the Newtonian pair when mond=True, else a repeat.
    Units: same code units as the raw files -- physical-unit conversion
    (trans_pot/trans_acce, boxlength) is NOT applied here; see
    `apply_units()` below.
    """
    nchar = repository.rstrip('/').split('_')[-1]

    # ---------- quick header pass ----------
    amr_path = f"{repository}/amr_{nchar}.out{icpu:05d}"
    with FortranFile(amr_path, 'r') as f:
        ncpu          = int(f.read_ints(np.int32)[0])
        ndim          = int(f.read_ints(np.int32)[0])
        nx, ny, nz    = f.read_ints(np.int32)
        nlevelmax     = int(f.read_ints(np.int32)[0])
        ngridmax      = int(f.read_ints(np.int32)[0])
        nboundary     = int(f.read_ints(np.int32)[0])
        ngrid_current = int(f.read_ints(np.int32)[0])
        boxlen        = float(f.read_reals(np.float32)[0])

    if ncpu != 1:
        raise NotImplementedError(
            f"ncpu={ncpu} -- this script only supports ncpu=1. "
            "You need the Hilbert domain-decomposition logic for multi-CPU runs."
        )

    twotondim = 2 ** ndim
    xbound = np.array([nx // 2, ny // 2, nz // 2], dtype=np.float64)
    lmax = nlevelmax

    # ---------- full AMR structural pass ----------
    levels = {}
    with FortranFile(amr_path, 'r') as f:
        for _ in range(21):
            f.read_record('i')  # 21 unused header records, matches "do i=1,21" skip

        ngridlevel = f.read_ints(np.int32).reshape((ncpu, nlevelmax), order='F')
        ngridfile = np.zeros((ncpu + nboundary, nlevelmax), dtype=np.int64)
        ngridfile[:ncpu, :] = ngridlevel
        f.read_record('i')  # matches bare read(10) right after ngridlevel

        if nboundary > 0:
            f.read_record('i')
            f.read_record('i')
            ngridbound = f.read_ints(np.int32).reshape((nboundary, nlevelmax), order='F')
            ngridfile[ncpu:, :] = ngridbound

        f.read_record('i')  # read(10)
        f.read_record('i')  # read(10)  ("ROM: comment..." line)

        # NOTE: ordering-dependent skip count -- 5 records for 'bisection', else 1
        n_ordering_skips = 5 if ordering_is_bisection else 1
        for _ in range(n_ordering_skips):
            f.read_record('i')

        f.read_record('i')
        f.read_record('i')
        f.read_record('i')

        for ilevel in tqdm(range(1, lmax + 1), desc="AMR pass: levels"):
            dx = 0.5 ** ilevel

            for j in range(1, nboundary + ncpu + 1):
                ncache = int(ngridfile[j - 1, ilevel - 1])
                if ncache <= 0:
                    continue

                f.read_record('i')  # grid index
                f.read_record('i')  # next index
                f.read_record('i')  # prev index

                xg_this = np.zeros((ncache, ndim)) if j == icpu else None
                for idim in range(ndim):
                    rec = f.read_reals(np.float64)
                    if j == icpu:
                        xg_this[:, idim] = rec

                f.read_record('i')  # father index
                for _ in range(2 * ndim):
                    f.read_record('i')  # nbor index

                son_this = np.zeros((ncache, twotondim), dtype=np.int64) if j == icpu else None
                for ind in range(twotondim):
                    rec = f.read_ints(np.int32)
                    if j == icpu:
                        son_this[:, ind] = rec

                for _ in range(twotondim):
                    f.read_record('i')  # cpu map
                for _ in range(twotondim):
                    f.read_record('i')  # refinement map

                if j == icpu:
                    levels[ilevel] = {'xg': xg_this, 'son': son_this, 'ngrid': ncache}

    # ---------- grav file pass ----------
    grav_path = f"{repository}/grav_{nchar}.out{icpu:05d}"
    rows = []
    ndim2 = ndim  # force has ndim components, matches ivar=1,ndim2

    def read_pot_force(g, ncache, twotondim, ndim2):
        pot = np.zeros((ncache, twotondim))
        force = np.zeros((ncache, twotondim, ndim2))
        for ind in range(twotondim):
            pot[:, ind] = g.read_reals(np.float64)
            for ivar in range(ndim2):
                force[:, ind, ivar] = g.read_reals(np.float64)
        return pot, force

    with FortranFile(grav_path, 'r') as g:
        g.read_ints(np.int32)  # ncpu2
        g.read_ints(np.int32)  # ndim2
        g.read_ints(np.int32)  # nlevelmax2
        g.read_ints(np.int32)  # nboundary2

        for ilevel in tqdm(range(1, lmax + 1), desc="Grav pass: levels"):
            dx = 0.5 ** ilevel
            level_data = levels.get(ilevel)
            xc = np.array([_cell_offsets(ind, dx) for ind in range(twotondim)])

            for j in range(1, nboundary + ncpu + 1):
                ncache = int(ngridfile[j - 1, ilevel - 1])

                g.read_record('i')  # per-(level,domain) skip #1
                g.read_record('i')  # per-(level,domain) skip #2

                if ncache <= 0:
                    continue

                if mond:
                    potential_N, force_N = read_pot_force(g, ncache, twotondim, ndim2)
                    potential, force = read_pot_force(g, ncache, twotondim, ndim2)
                else:
                    potential, force = read_pot_force(g, ncache, twotondim, ndim2)
                    potential_N, force_N = potential, force

                if j == icpu and level_data is not None:
                    xg = level_data['xg']
                    son = level_data['son']
                    for ind in range(twotondim):
                        x = xg + xc[ind] - xbound  # (ncache, 3)
                        ref = (son[:, ind] > 0) & (ilevel < lmax)
                        nonzero = (potential[:, ind] != 0.0) | np.any(force[:, ind, :] != 0.0, axis=1)
                        keep = (~ref) & nonzero
                        n_keep = int(keep.sum())
                        if n_keep == 0:
                            continue
                        block = np.empty((n_keep, 13))
                        block[:, 0:3] = x[keep]
                        block[:, 3] = dx
                        block[:, 4] = potential[keep, ind]
                        block[:, 5:8] = force[keep, ind, :]
                        block[:, 8] = ilevel
                        block[:, 9] = potential_N[keep, ind]
                        block[:, 10:13] = force_N[keep, ind, :]
                        rows.append(block)

    return np.concatenate(rows, axis=0) if rows else np.empty((0, 13))


def apply_units(raw, info_path):
    """
    Convert the code-unit output of read_amr_grav() into physical units,
    matching the Fortran's trans_pot / trans_acce / boxlength scaling.
    info_path: path to info_XXXXX.txt for this output.
    """
    boxlength = unit_l = unit_t = None
    with open(info_path) as fh:
        for line in fh:
            if line.strip().startswith('boxlen'):
                boxlength = float(line.split('=')[1])
            elif line.strip().startswith('unit_l'):
                unit_l = float(line.split('=')[1])
            elif line.strip().startswith('unit_t'):
                unit_t = float(line.split('=')[1])
    if boxlength is None or unit_l is None or unit_t is None:
        raise ValueError(f"Could not parse boxlen/unit_l/unit_t from {info_path}")

    trans_pot = (unit_l / 100.0) ** 2 / unit_t ** 2   # cm -> m
    trans_acce = (unit_l / 100.0) / unit_t ** 2

    out = raw.copy()
    out[:, 0:3] = (raw[:, 0:3] - 0.5) * boxlength   # position -> physical
    out[:, 3] = raw[:, 3] * boxlength               # dx -> physical
    out[:, 4] *= trans_pot                          # potential
    out[:, 5:8] *= trans_acce                       # force
    out[:, 9] *= trans_pot                          # potential_N / potential (non-mond)
    out[:, 10:13] *= trans_acce                     # force_N / force (non-mond)
    return out


if __name__ == "__main__":
    import sys
    repo = sys.argv[1] if len(sys.argv) > 1 else "output_00002"
    raw = read_amr_grav(repo, mond=True)
    print(f"Read {len(raw)} cells")
    phys = apply_units(raw, f"{repo}/info_{repo.split('_')[-1]}.txt")
    print(phys[:5])
