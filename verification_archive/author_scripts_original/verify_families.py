"""Independent verification of the explicit families in the manuscript.

Exact (integer) checks: K^2 = k^2 I, connectivity, abelian-Cayley obstruction.
Floating-point checks (numpy/scipy): spectrum and U(pi/2) = lambda * R_c.
Automorphism orders by backtracking (no nauty needed); small n only.
Exit status is non-zero if any check fails.
"""
import sys
import numpy as np
from scipy.linalg import expm
from common import *

fails = []


def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name, flush=True)
    if not cond:
        fails.append(name)


def pi2_transfer(G, S, c, lam):
    H = hermitian(G, S + [c])
    U = expm(-1j * np.pi / 2 * H)
    return np.allclose(U, lam * right_translation(G, c))


# ---- Proposition (D_n family), S' = {x, x a^2, a}, c = a^{n/2}, k = 2 -----------
print("== Proposition Dn: Gamma_n on D_n, S'={x, x a^2, a}, c=a^{n/2}")
for n in range(4, 17, 2):
    G = cyclic_dihedral(n)
    c = ((n // 2,), 0)
    S = [((0,), 1), ((2 % n,), 1), ((1,), 0)]
    check(f"n={n}: K^2=4I (exact)", square_is_scalar(G, S, 2))
    check(f"n={n}: connected", generates(G, S + [c]))
    H = hermitian(G, S + [c])
    ev = np.round(np.linalg.eigvalsh(H), 6)
    vals, cnts = np.unique(ev, return_counts=True)
    check(f"n={n}: spectrum {{-3,-1,1,3}} each with multiplicity n/2",
          list(vals) == [-3, -1, 1, 3] and all(m == n // 2 for m in cnts))
    check(f"n={n}: U(pi/2) = i R_c", pi2_transfer(G, S, c, 1j))
    ab = abelian_test_passes(G, S)
    check(f"n={n}: abelian-Cayley obstruction {'absent (n=4)' if n == 4 else 'holds'}",
          ab if n == 4 else not ab)
    Hm = hermitian(G, S + [c])
    a_mixed = aut_order(colour_matrix(Hm))
    check(f"n={n}: |Aut(mixed graph)| = {16 if n == 4 else 2 * n} (got {a_mixed})",
          a_mixed == (16 if n == 4 else 2 * n))
    if n in (6, 8, 10, 12):
        a_und = aut_order(colour_matrix(Hm, undirected=True))
        want = {6: 1440, 8: 256, 10: 640, 12: 1536}[n]
        check(f"n={n}: |Aut(underlying undirected graph)| = {want} (got {a_und})",
              a_und == want)

# ---- Proposition (weight 2 on Dic_n and SD_{2^k}) -------------------------------
print("== Weight 2: S'={x, x c, g} with g^2=c")
for n in range(3, 13):
    G = dicyclic(n)
    c = ((n,), 0)
    S = [((0,), 1), ((n,), 1), ((1,), 1)]
    check(f"Dic_{n}: g^2=c", G.mul(((1,), 1), ((1,), 1)) == c)
    check(f"Dic_{n}: K^2=4I (exact)", square_is_scalar(G, S, 2))
    check(f"Dic_{n}: connected", generates(G, S + [c]))
    check(f"Dic_{n}: abelian-Cayley obstruction holds", not abelian_test_passes(G, S))
    if n <= 6:
        a = aut_order(colour_matrix(hermitian(G, S + [c])))
        check(f"Dic_{n}: |Aut| = n 2^(n+1) = {n * 2 ** (n + 1)} (got {a})",
              a == n * 2 ** (n + 1))
for k in range(4, 9):
    G = semidihedral(k)
    c = ((2 ** (k - 2),), 0)
    S = [((0,), 1), ((2 ** (k - 2),), 1), ((1,), 1)]
    check(f"SD_{2 ** k}: g^2=c", G.mul(((1,), 1), ((1,), 1)) == c)
    check(f"SD_{2 ** k}: K^2=4I (exact)", square_is_scalar(G, S, 2))
    check(f"SD_{2 ** k}: connected", generates(G, S + [c]))

# ---- Corollary (weight 4 on Dic_n, n >= 5) --------------------------------------
print("== Weight 4 on Dic_n: R_u={0,n,2,n+2}, T={1,n+1,2,n-2}, R_a={a,a+4}")
for n in range(5, 31):
    G = dicyclic(n)
    m = 2 * n
    a = 4 if n == 5 else 1
    Ru = [0, n, 2, (n + 2) % m]
    T = [1, (1 + n) % m, 2, (n - 2) % m]
    Ra = [a, (a + 4) % m]
    S = [((r,), 1) for r in Ru] + [((t,), 0) for t in T] + [((r,), 1) for r in Ra]
    c = ((n,), 0)
    ok = len(set(S)) == 10 and square_is_scalar(G, S, 4)
    check(f"Dic_{n}: |S'|=10 and K^2=16I (exact)", ok)
    check(f"Dic_{n}: connected", generates(G, S + [c]))
    check(f"Dic_{n}: abelian-Cayley obstruction holds", not abelian_test_passes(G, S))
    if n <= 12:
        H = hermitian(G, S + [c])
        ev = sorted(set(np.round(np.linalg.eigvalsh(H), 6)))
        check(f"Dic_{n}: spectrum {{-5,-3,3,5}}", ev == [-5, -3, 3, 5])
        check(f"Dic_{n}: U(pi/2) = -i R_c", pi2_transfer(G, S, c, -1j))

# ---- Proposition (Z_4 x Z_4m) ----------------------------------------------------
print("== Weight 4 on Dih(Z_4 x Z_4m)")
for m in range(2, 7):
    G = generalized_dihedral([4, 4 * m])
    T = [(0, 1), (0, m), (1, 0), (1, 2 * m), (2, 1), (2, 3 * m)]
    R = [(0, 0), (2, 0), (0, 2), (2, 2)]
    S = [(t, 0) for t in T] + [(r, 1) for r in R]
    invols = [a for a in G.A if a != (0, 0) and all((2 * x) % mm == 0 for x, mm in zip(a, G.mods))]
    for cc in invols:
        c = (cc, 0)
        check(f"m={m}, c={cc}: K^2=16I, connected, abelian obstruction holds",
              square_is_scalar(G, S, 4) and generates(G, S + [c]) and not abelian_test_passes(G, S))

print()
print("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED: {fails}")
sys.exit(1 if fails else 0)
