# PG0844 PyROA delay_dist run — note for the server Claude session

**Context for a Claude Code session on the igm server.** A parallel session on
the user's Mac prepared this. Read it before doing anything.

## What we are doing

PyROA continuum reverberation mapping of the quasar **PG0844**. This task: the
**`delay_dist=True` emcee fit on the raw light curves** — each band modelled as
a delay *distribution* (not a single sharp lag) of one common driving curve.

Background: an earlier fit (`run_12`, `delay_dist=False`) had trouble with the
i/z-band lags. This `delay_dist` fit on the raw data is the cleaner standard
fit. It is also the **validation reference** for a JAX/numpyro HMC
reimplementation being built in parallel on the Mac — so getting it converged
matters.

## The code

This repo is `jiamuH/PyROA`, a fork of `FergusDonnan/PyROA`, branch
**`optimize-emcee`**. On top of upstream that branch adds:
- numba `cache=True` + `fastmath=True` on the jitted ROA functions (speed);
- `skip_initial_state_check=True` in `FullFit`'s `run_mcmc` (lets a contracted
  walker ensemble resume — emcee otherwise refuses with a condition-number error);
- drops the unused `generated_jit` import (numba >=0.59 compatibility).

Run script: **`pg0844_run/run_pg0844_delaydist.py`**
- `delay_dist=True`, `gridsize=1000`, `add_var=True`, `AccDisc=False`;
- priors `[[0.5,2],[0.5,2],[-50,50],[0.01,30],[0,10]]` = [A, B, tau, delta, sigma];
- `Nsamples=5000` per launch; auto-resumes from `Fit.h5` (re-run to add more).

## Please verify first (the user wasn't 100% sure)

1. Data is at `/data2/jhuang/lightcurves/pyroa_pg0844/lc_dat/` — 8 files
   `PG0844_{B1,u,B,g,V,r,i,z}.dat`, each (MJD, flux, err).
2. A working PyROA conda env exists (the one used for `run_12`). It needs
   numpy, scipy, matplotlib, emcee, h5py, numba, tqdm, tabulate, corner,
   astropy, pandas. Point PyROA at THIS checkout on the `optimize-emcee`
   branch — from the repo root: `python3 -m pip install -e .`

## How to run

1. Activate the PyROA conda env.
2. `cd` to a directory for the outputs (PyROA writes `Fit.h5`, `samples.obj`,
   `X_t.obj`, etc. to the current directory).
3. `python3 <repo>/pg0844_run/run_pg0844_delaydist.py`
4. The first launch does 5000 steps. **Re-run the same command to resume**
   (adds another `Nsamples`). For convergence you will want many more —
   `run_12`'s tau chains needed ~100k+ steps; bump `Nsamples` in the script
   for bigger chunks.
5. Check convergence with `PyROA.Chains` / `PyROA.Convergence`; resume until
   the tau chains settle. Do **not** delete `Fit.h5` unless you want to restart
   from scratch.

## Notes

- The `if __name__ == '__main__'` guard in the run script is a macOS
  requirement; harmless on Linux.
- numba `cache=True` writes a compiled-function cache next to `PyROA.py`.
- When the tau chains have converged, report the per-band delay (tau) values
  back to the user.
