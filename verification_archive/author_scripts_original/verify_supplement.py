"""Checks of the supplementary note: Proposition S1.4 (m=2..10, including the m=2 remark),
|Aut| of the SD_{2^k} family (k=4,5,6) and of the Dic_n family (n=3..10), and Proposition S3.2
(multiple state transfer on M_{2^n}, n=5..8).  About 1 minute."""
import sys
import numpy as np
from scipy.linalg import expm
from common import *

fails = []
def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name, flush=True)
    if not cond:
        fails.append(name)

# --- Proposition S1.4 -------------------------------------------------------
for m in range(2, 11):
    q = 4 * m
    G = generalized_dihedral([4, q]); c = ((0, 2 * m), 0)
    R = [(0, 0), (0, 2), (0, 2 * m), (0, 2 * m + 2)]
    T = [(0, 1), (0, m), (0, (2 * m + 1) % q), (1, 0), (3, 2 * m), (2, m)]
    S = [(r, 1) for r in R] + [(t, 0) for t in T]
    check(f"S1.4 m={m}: admissible, K^2=16I (exact), connected",
          len(set(S)) == 10 and c not in S and square_is_scalar(G, S, 4) and generates(G, S + [c]))
    check(f"S1.4 m={m}: abelian-Cayley obstruction {'absent (m=2, as the remark says)' if m == 2 else 'holds'}",
          abelian_test_passes(G, S) if m == 2 else not abelian_test_passes(G, S))
    if m <= 6:
        H = hermitian(G, S + [c])
        ev = sorted(set(int(round(e)) for e in np.linalg.eigvalsh(H)))
        U = expm(-1j * np.pi / 2 * H)
        check(f"S1.4 m={m}: spectrum {{-5,-3,3,5}} and U(pi/2) = -i R_c",
              ev == [-5, -3, 3, 5] and np.allclose(U, -1j * right_translation(G, c)))
    if m == 2:
        a = aut_order(colour_matrix(hermitian(G, S + [c])))
        check(f"S1.4 m=2: |G|=64, |Aut|=6144 (got {a})", len(G.els) == 64 and a == 6144)

# --- Remark S1.2 ------------------------------------------------------------
for n in range(3, 11):
    G = dicyclic(n); c = ((n,), 0)
    a = aut_order(colour_matrix(hermitian(G, [((0,), 1), ((n,), 1), ((1,), 1), c])))
    check(f"S1.2 Dic_{n}: |Aut| = n 2^(n+1) = {n * 2 ** (n + 1)} (got {a})", a == n * 2 ** (n + 1))
for k, e in ((4, 7), (5, 12), (6, 21)):
    G = semidihedral(k); c = ((2 ** (k - 2),), 0)
    a = aut_order(colour_matrix(hermitian(G, [((0,), 1), ((2 ** (k - 2),), 1), ((1,), 1), c])))
    check(f"S1.2 SD_{2 ** k}: |Aut| = 2^{e} (got {a})", a == 2 ** e)

# --- Proposition S3.2 -------------------------------------------------------
def order(G, g):
    k, h = 1, g
    while h != G.one:
        h = G.mul(h, g); k += 1
    return k
for n in range(5, 9):
    N = 2 ** (n - 1)
    G = Group([N], lambda a, N=N: (((N // 2 + 1) * a[0]) % N,), (0,))
    X = lambda a: ((a % N,), 0)
    xs = lambda a: G.mul(X(a), ((0,), 1))                       # x^a sigma
    gens = [X(N // 4), xs(N // 8), xs(N // 2 + N // 8)] + [X(4 * k + 1) for k in range(N // 4)]
    C = set()
    for g in gens:
        C |= {G.mul(G.mul(h, g), G.inv(h)) for h in G.els}
    g0 = xs(N // 16)
    gK = [G.mul(g0, X((N // 8) * i)) for i in range(8)]
    S = list(C) + gK; Ss = set(S)
    oriented = not any(G.inv(s) in C for s in C)
    normal = all(G.mul(G.mul(h, s), G.inv(h)) in Ss for h in G.els for s in S)
    zeta = sum(1 for s in S if G.inv(s) in Ss)
    H = hermitian(G, S)
    U = expm(-1j * np.pi / 4 * H)
    col = np.abs(U[:, 0]); js = [i for i in range(1, len(col)) if col[i] > 1 - 1e-7]
    ok = bool(js)
    if ok:
        z = G.els[js[0]]
        ok = np.allclose(U, U[js[0], 0] * right_translation(G, z)) and order(G, z) == 4
    check(f"S3.2 n={n}: C_n oriented, S_n normal, zeta=8, connected, PST at pi/4 with a target of order 4",
          oriented and normal and zeta == 8 and generates(G, S) and ok)

print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED: {fails}")
sys.exit(1 if fails else 0)
