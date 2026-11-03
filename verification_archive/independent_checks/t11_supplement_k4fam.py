"""Independent check of Proposition prop:k4fam of the supplementary note, with the sets EXACTLY as stated there:
G = Dih(Z4 x Z_q), q = 4m, f = -id, y = 0, c = (0,2m),
R = {(0,0),(0,2),(0,2m),(0,2m+2)} (undirected elements (r,1)),
T = {(0,1),(0,m),(0,2m+1),(1,0),(3,2m),(2,m)} (arcs (t,0)),  S' = {(r,1)} u {(t,0)},  S = S' u {c}.
Claims tested for each m: c not in S'; K^2 = 16 I (exact integer arithmetic over Z[i]); zeta = 5; spectrum of H inside {+-3,+-5};
U(pi/2) = -i R_c; connected; commutation test of Lemma abtest non-zero (claimed for m >= 3, and claimed to VANISH for m = 2); |Aut| by nauty.
The remark of the note says |G| = 64 and |Aut| = 6144 for m = 2."""
import sys, numpy as np
from collections import Counter
from scipy.linalg import expm
from lib import *

def run(m):
    q = 4 * m
    G = Grp((4, q), lambda a: ((-a[0]) % 4, (-a[1]) % q), (0, 0))
    e = lambda a, b, r: G.idx[((a % 4, b % q), r)]
    c = e(0, 2 * m, 0)
    R = [(0, 0), (0, 2), (0, 2 * m), (0, 2 * m + 2)]
    T = [(0, 1), (0, m), (0, 2 * m + 1), (1, 0), (3, 2 * m), (2, m)]
    Sp = [e(a, b, 1) for a, b in R] + [e(a, b, 0) for a, b in T]
    assert len(set(Sp)) == 10 and c not in Sp
    S = Sp + [c]
    n = G.n
    # exact K^2 = 16 I in Z[i][G]
    w = {}
    Sset = set(S) - {c}
    for s in Sp:
        si = int(G.inv[s])
        if si in Sset:
            w[s] = (1, 0)
        else:
            w[s] = (0, 1); w[si] = (0, -1)
    acc = {}
    for s, (a, b) in w.items():
        for t, (c2, d) in w.items():
            g = int(G.mt[s, t]); x, y = acc.get(g, (0, 0)); acc[g] = (x + a * c2 - b * d, y + a * d + b * c2)
    exact = acc.get(G.one) == (16, 0) and all(v == (0, 0) for g, v in acc.items() if g != G.one)
    H = G.H(S)
    zeta = sum(1 for s in S if int(G.inv[s]) in set(S))
    ev = sorted(set(int(x) for x in np.round(np.linalg.eigvalsh(H))))
    inside = set(ev) <= {-5, -3, 3, 5}
    U = expm(-1j * np.pi / 2 * H)
    pst = np.allclose(U, -1j * G.Rmat(c), atol=1e-7)
    conn = len(gen_subgroup(G, S)) == n
    Su = [s for s in S if int(G.inv[s]) in set(S)]; Ta = [s for s in S if int(G.inv[s]) not in set(S)]
    A = Counter(); B = Counter()
    for s in Su:
        for t in Ta:
            A[G.mt[t, s]] += 1; B[G.mt[s, t]] += 1
    nonabelian_test = (A != B)
    aut = digraph_aut_order(H)
    return dict(m=m, order=n, K2_exact=exact, zeta=zeta, spectrum=ev, spectrum_ok=inside, U_is_minus_i_Rc=pst, connected=conn,
                commutation_test_nonzero=nonabelian_test, Aut=aut)

if __name__ == "__main__":
    for m in range(2, 11):
        print(run(m), flush=True)
