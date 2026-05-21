"""PyROA delay-distribution fit to the raw PG0844 light curves -- SERVER copy.

delay_dist=True: each band gets a delay DISTRIBUTION (driver convolved with a
smearing kernel) instead of a single sharp lag. Fits the raw light curves in
/data2/jhuang/lightcurves/pyroa_pg0844/lc_dat/ directly.

This is the server twin of the Mac script. See SERVER_NOTE.md (same folder)
for the full context -- what we are doing and why.

Run on the server (Linux): activate the PyROA conda env, cd to the directory
where you want PyROA's outputs (Fit.h5 etc. are written to the current dir):
    python3 /path/to/PyROA/pg0844_run/run_pg0844_delaydist.py
Re-run the same command to resume from Fit.h5 (adds Nsamples more steps).
"""
import os

import emcee
import PyROA

datadir = '/data2/jhuang/lightcurves/pyroa_pg0844/lc_dat/'
objName = 'PG0844'
filters = ['B1', 'u', 'B', 'g', 'V', 'r', 'i', 'z']

# PyROA priors for this version: [A, B, tau, delta, sigma] -- 5 entries, the
# per-band delay-distribution width is handled internally by PyROA.
#   tau    mean lag, days
#   delta  ROA window width, days (upper bound 30 -- run_12's delta wanted ~15)
priors = [[0.5, 2.0], [0.5, 2.0], [-50.0, 50.0], [0.01, 30.0], [0.0, 10.0]]

GRIDSIZE = 1000   # ROA model grid points; must be an int (None breaks numba)


def setup_resume():
    """Resume if Fit.h5 holds completed steps; drop a stale empty backend
    left by a failed run so the next attempt starts cleanly."""
    if not os.path.exists('Fit.h5'):
        print('STARTING a fresh run')
        return False
    try:
        n = emcee.backends.HDFBackend('Fit.h5').iteration
    except Exception:
        n = 0
    if n > 0:
        print(f'RESUMING from Fit.h5 ({n} steps already done)')
        return True
    os.remove('Fit.h5')
    print('removed a stale empty Fit.h5 -> STARTING a fresh run')
    return False


def main():
    # Shared-server courtesy: lower scheduling priority before the multi-day
    # MCMC so the run is a good neighbour. Pool workers fork after this call
    # and inherit the niceness, so no 'nice' prefix is needed on the command.
    print('process niceness set to', os.nice(10))
    PyROA.Fit(datadir, objName, filters, priors, init_tau=None,
              Nsamples=50000, Nburnin=0, add_var=True, delay_dist=True,
              AccDisc=False, use_backend=True, resume_progress=setup_resume(),
              gridsize=GRIDSIZE)


# __main__ guard: needed on macOS (multiprocessing 'spawn'); harmless on Linux.
if __name__ == '__main__':
    main()
