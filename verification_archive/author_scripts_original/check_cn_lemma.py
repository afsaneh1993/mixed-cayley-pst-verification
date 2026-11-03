"""Lemma S3.1' (Lemma lem:Cn of the supplementary note), n = 5..8:
(a) C_n equals {x^{N/4}} u {x^{N/8}s, x^{5N/8}s} u {x^a : a = 1 mod 4} and is oriented;
(b) spectrum of H is {0, +-2, +-4, +-N/2};
(c) exp(-i pi/4 H) = P_z with z = x^{N/4}, i.e. e_h -> e_{h z^{-1}}.
Floating point (tolerance 1e-7) for (b), (c).  Exit status 0 iff all hold."""
import sys
import numpy as np
from scipy.linalg import expm
from common import *

ok = True
for n in range(5, 9):
    N = 2 ** (n - 1)
    G = Group([N], lambda a, N=N: (((N // 2 + 1) * a[0]) % N,), (0,))
    X = lambda a: ((a % N,), 0)
    xs = lambda a: G.mul(X(a), ((0,), 1))
    gens = [X(N // 4), xs(N // 8), xs(N // 2 + N // 8)] + [X(4 * k + 1) for k in range(N // 4)]
    C = set()
    for g in gens:
        C |= {G.mul(G.mul(h, g), G.inv(h)) for h in G.els}
    pred = {X(N // 4), xs(N // 8), xs(5 * N // 8)} | {X(a) for a in range(1, N, 4)}
    a_ok = (C == pred) and not any(G.inv(s) in C for s in C)
    H = hermitian(G, list(C))
    vals = sorted(set(int(round(e)) for e in np.linalg.eigvalsh(H)))
    b_ok = vals == sorted({0, 2, -2, 4, -4, N // 2, -N // 2})
    z = X(N // 4)
    P = np.zeros((len(G.els),) * 2)
    for h in G.els:
        P[G.idx[G.mul(h, G.inv(z))], G.idx[h]] = 1
    c_ok = np.allclose(expm(-1j * np.pi / 4 * H), P, atol=1e-7)
    print(f"{'PASS' if a_ok and b_ok and c_ok else 'FAIL'}  n={n}: (a) {a_ok}  (b) spectrum {vals}  (c) {c_ok}", flush=True)
    ok &= a_ok and b_ok and c_ok
print("\nALL CHECKS PASSED" if ok else "\nCHECK(S) FAILED")
sys.exit(0 if ok else 1)
