"""Remark in Section 'reduction to abelian groups': for n = 6, 10, 14, among ALL mixed connection sets S in Z_n \\ {0}
with zeta > 0 (at least one undirected element), the perfect matching S = {n/2} is the only one with PST.
Every set is tested (4 options per pair {t,-t}, 2 for n/2).  PST is decided without using the structure theorem:
the transfer matrix column U(t)e_0 is computed from the spectrum on a fine grid of t in (0, 4*pi], and every grid
cell where some |U(t)_{z0}| > 0.9 is refined by a bounded optimiser; PST is declared iff the refined maximum exceeds 1-1e-7.  Usage: python3 check_zn_mixed.py   (exit 0 iff the claim holds)"""
import sys, itertools
import numpy as np

def spectrum(n, w):                       # w[s] in {0,1,1j,-1j}: H[g,h] = w[h-g]
    return np.array([sum(w[s] * np.exp(2j * np.pi * j * s / n) for s in range(1, n)) for j in range(n)])

def pst_exists(n, lam):
    """max over t of |U(t)_{z,0}| over z != 0 on a fine grid up to T, then refine candidates."""
    T = 4 * np.pi                          # all integral spectra considered below have gamma>=1: the least PST time <= 2*pi/gamma*...
    ts = np.linspace(1e-3, T, 20000)
    ph = np.exp(-1j * np.outer(ts, lam.real))                  # (t, j)
    F = np.exp(2j * np.pi * np.outer(np.arange(n), np.arange(n)) / n) / n   # column 0 of eigenbasis: u_{z} = (1/n) sum_j e^{2 pi i j z/n} ph_j
    u = ph @ F.T                                               # (t, z)
    m = np.abs(u[:, 1:]).max(axis=1)
    idx = np.where(m > 0.9)[0]
    from scipy.optimize import minimize_scalar
    for i in idx:
        lo, hi = ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]
        f = lambda t: -np.abs((np.exp(-1j * lam.real * t) @ F.T)[1:]).max()
        r = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
        if -r.fun > 1 - 1e-7:
            return True
    return False

ok = True
for n in (6, 10, 14):
    pairs = [(t, n - t) for t in range(1, (n + 1) // 2) if 2 * t != n]
    half = n // 2
    found, total = [], 0
    for opts in itertools.product(range(4), repeat=len(pairs)):
        for h in (0, 1):
            w = {s: 0 for s in range(1, n)}
            for (t, mt), o in zip(pairs, opts):
                if o == 1:   w[t] = 1;  w[mt] = 1
                elif o == 2: w[t] = 1j; w[mt] = -1j
                elif o == 3: w[t] = -1j; w[mt] = 1j
            if h: w[half] = 1
            zeta = sum(1 for s in range(1, n) if w[s] == 1)
            if zeta == 0: continue
            total += 1
            lam = spectrum(n, w)
            if np.max(np.abs(lam.imag)) > 1e-9: raise SystemExit("spectrum should be real")
            if pst_exists(n, lam): found.append(sorted(s for s in range(1, n) if w[s] != 0))
    # w[s]=1 for BOTH t and -t is an undirected edge; options 2 and 3 are arcs t->... (equivalent up to the relabelling t -> -t, kept both)
    good = (found == [[half]])
    print(f"n={n}: {total} sets with zeta>0 tested; sets with PST: {found}; claim {'holds' if good else 'FAILS'}", flush=True)
    ok &= good
sys.exit(0 if ok else 1)
