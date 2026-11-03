"""Weight-4 weighing sets on Dic(A,y) for non-cyclic A of order <= 16 (Remark comp of the manuscript).

Group: G = A x| Z2 with f = -id and x^2 = y in A[2]; c ranges over the central involutions A[2]\\{0}.
A set S' (S' subset of G\\{1}) is a weighing set of weight 4 if K = sum_{s in S'} w_s rho(s) satisfies K^2 = 16 I
(w_s = 1 for s in S' cap S'^-1, w_s = i for the other elements of S', and the inverse of such an s carries -i in K),
and it is *connected for c* if c is not in S' and <S' u {c}> = G.

Method (all arithmetic exact: Gaussian integers as two integer-valued float64 arrays, compared with ==, values < 2^53):
  * K^2 = 16 I is decided on the row of the identity (vertex-transitivity): (K^2)_{1,z} = sum_{g h = z} w_g w_h.
  * The only pruning uses two consequences of K^2 = 16 I alone:  |U| = |S' cap S'^-1| = 4  (because K 1 = (|U|) 1 and
    K^2 1 = 16 * 1)  and  |U| + 2|T| = 16, i.e. |T| = 6  (the diagonal of K^2 counts the non-zero entries of a row).
    Hence U is a union of inverse classes of total size 4 (singletons = involutions, pairs) and T takes one element
    from each of 6 further inverse pairs.  Every such S' is tested; nothing from the Fourier criterion is used.
  * S' is enumerated independently of c; afterwards every c in A[2]\\{0} with c not in S' is tested for connectivity.
  * Type I: U meets xA;  Type II: U is contained in A;  "mixed" (neither) would contradict Theorem gd:fourier(b) and is counted.
Usage:  python3 noncyclic_weight4_third_check.py            full run for the groups of the manuscript (about 1 hour)
        python3 noncyclic_weight4_third_check.py --validate    only the validation cases (about 2 minutes)
        python3 noncyclic_weight4_third_check.py 6,2 8,2        only the listed groups A = Z_6 x Z_2, Z_8 x Z_2
Exit status 0 iff every claim check of the full run prints MATCH (validation: iff all validations pass)."""
import sys, time, itertools
import numpy as np
from lib import Grp


def build(mods, y):
    return Grp(mods, lambda a: tuple((-x) % m for x, m in zip(a, mods)), y)


def A2(mods):
    return [a for a in itertools.product(*[range(m) for m in mods]) if all((2 * x) % m == 0 for x, m in zip(a, mods))]


def solve(mods, y, batch=40000):
    """Return (list of S' as tuples of 10 element indices with K^2=16I, group object)."""
    G = build(mods, y)
    n, mt, inv = G.n, G.mt, G.inv
    sing, pairs, seen = [], [], set()
    for e in range(n):
        if e == G.one or e in seen:
            continue
        h = int(inv[e]); seen |= {e, h}
        (sing.append(e) if h == e else pairs.append((e, h)))
    P = np.array(pairs).reshape(-1, 2)
    m = len(pairs)
    if m < 6:
        return [], G                                    # |T| = 6 needs six inverse pairs
    orient = np.array(list(itertools.product((0, 1), repeat=6)))          # (64,6)
    Tc = np.array(list(itertools.combinations(range(m), 6)))
    sols = []
    for ns in (0, 2, 4):
        npair = (4 - ns) // 2
        for Ss in itertools.combinations(sing, ns):
            for Up in itertools.combinations(range(m), npair):
                Ue = np.array(list(Ss) + [x for p in Up for x in pairs[p]], dtype=int)
                mask = np.ones(len(Tc), dtype=bool)
                for p in Up:
                    mask &= ~(Tc == p).any(axis=1)
                T = Tc[mask]
                if len(T) == 0:
                    continue
                ch = np.where(orient[None, :, :] == 0, P[T][:, None, :, 0], P[T][:, None, :, 1]).reshape(-1, 6)
                iv = np.where(orient[None, :, :] == 0, P[T][:, None, :, 1], P[T][:, None, :, 0]).reshape(-1, 6)
                B = len(ch)
                for s in range(0, B, batch):
                    c_, i_ = ch[s:s + batch], iv[s:s + batch]
                    b = len(c_)
                    el = np.concatenate([np.broadcast_to(Ue, (b, 4)), c_, i_], axis=1)            # (b,16)
                    w = np.concatenate([np.ones((b, 4)), np.full((b, 6), 1j), np.full((b, 6), -1j)], axis=1)
                    z = mt[el[:, :, None], el[:, None, :]]
                    pr = w[:, :, None] * w[:, None, :]
                    idx = (np.arange(b)[:, None, None] * n + z).ravel()
                    re = np.bincount(idx, weights=pr.real.ravel(), minlength=b * n).reshape(b, n)
                    im = np.bincount(idx, weights=pr.imag.ravel(), minlength=b * n).reshape(b, n)
                    want = np.zeros(n); want[G.one] = 16
                    ok = (re == want).all(axis=1) & (im == 0).all(axis=1)
                    for j in np.where(ok)[0]:
                        sols.append(tuple(int(x) for x in el[j][:4]) + tuple(int(x) for x in c_[j]))
    return sols, G


def closure_size(G, gens):
    seen = {G.one}; fr = [G.one]
    while fr:
        nf = []
        for h in fr:
            for t in gens:
                x = int(G.mt[h, t])
                if x not in seen:
                    seen.add(x); nf.append(x)
        fr = nf
    return len(seen)


def analyse(mods, y):
    t0 = time.time()
    sols, G = solve(mods, y)
    cs = [G.idx[(a, 0)] for a in A2(mods) if any(a)]
    pairs_I = pairs_II = mixed = 0
    per_c = {c: 0 for c in cs}; setsI = set(); setsII = set()
    for S in sols:
        U = [e for e in S if int(G.inv[e]) in S]
        in_x = [G.els[e][1] == 1 for e in U]
        typ = "I" if all(in_x) else ("II" if not any(in_x) else "mixed")
        if typ == "mixed":
            mixed += 1
        for c in cs:
            if c in S:
                continue
            if closure_size(G, list(S) + [c]) == G.n:
                per_c[c] += 1
                if typ == "I": pairs_I += 1; setsI.add(S)
                elif typ == "II": pairs_II += 1; setsII.add(S)
    return dict(mods=mods, y=y, n_K2=len(sols), mixed=mixed, I=pairs_I, II=pairs_II,
                connected=pairs_I + pairs_II, per_c={G.els[c][0]: v for c, v in per_c.items()}, secs=time.time() - t0)


def show(r):
    print(f"A={list(r['mods'])} y={r['y']}: sets with K^2=16I: {r['n_K2']}; connected (S',c) pairs: {r['connected']} "
          f"(Type I {r['I']}, Type II {r['II']}, neither {r['mixed']}); per c: {r['per_c']}   [{r['secs']:.0f} s]", flush=True)


def validate():
    ok = True
    print("VALIDATION 1: cyclic A = Z_2n, y = n  (one central involution c = n): connected sets must be 80, 288, 1176, 2688")
    for n, want in ((5, 80), (6, 288), (7, 1176), (8, 2688)):
        r = analyse([2 * n], (n,)); got = r['connected']
        print(f"  Dic_{n}: {got} (expected {want}) -> {'OK' if got == want else 'FAIL'}   [{r['secs']:.0f} s]", flush=True); ok &= got == want
    print("VALIDATION 2: cyclic A = Z_2n with y = 0 (dihedral): none")
    for n in (5, 6, 7):
        r = analyse([2 * n], (0,)); print(f"  A=Z_{2*n}, y=0: {r['connected']} -> {'OK' if r['connected'] == 0 else 'FAIL'}", flush=True); ok &= r['connected'] == 0
    print("VALIDATION 3: A = Z_6 x Z_2 against the archive's own enumerator (wenum_exact.py, summed over c): y=(0,0): 0; y=(0,1),(3,0),(3,1): 864 each")
    for y, want in (((0, 0), 0), ((0, 1), 864), ((3, 0), 864), ((3, 1), 864)):
        r = analyse([6, 2], y); show(r); ok &= r['connected'] == want and r['mixed'] == 0
    print("VALIDATION 4: A = Z_8 x Z_2, y = (0,0): none (archive enumerator: 0)")
    r = analyse([8, 2], (0, 0)); show(r); ok &= r['connected'] == 0
    print("VALIDATIONS PASSED" if ok else "VALIDATION FAILED")
    return ok


GROUPS = [[8], [4, 2], [6, 2], [8, 2], [4, 4], [4, 2, 2]]


def full(groups):
    res = {}
    for mods in groups:
        for y in A2(mods):
            r = analyse(mods, y); show(r); res[(tuple(mods), y)] = r
    return res


def claims(res):
    """Compare with the sentences of Remark comp (v11): see the comments for the exact claim."""
    ok = True
    def has(mods, y): return res[(tuple(mods), tuple(y))]['connected'] > 0
    def line(txt, cond):
        nonlocal ok
        ok &= bool(cond); print(("MATCH     " if cond else "MISMATCH  ") + txt)
    print("\nCLAIM CHECKS (manuscript v11, Remark comp)")
    for mods in ([6, 2], [8, 2], [4, 4]):
        ys = [y for y in A2(mods) if any(y)]
        line(f"connected sets exist for A={mods} and every y != 0", all(has(mods, y) for y in ys))
    ys = [y for y in A2([4, 2, 2]) if any(y)]
    got = sorted(y for y in ys if not has([4, 2, 2], y))
    line(f"A=[4,2,2]: connected sets exist for every y != 0 except y=(2,0,0)  (y != 0 without sets: {got})", got == [(2, 0, 0)])
    line("no connected sets for A=Z_8 (any y) and A=Z_4 x Z_2 (any y)",
         not any(has(m, y) for m in ([8], [4, 2]) for y in A2(m)))
    y0 = [m for m in ([6, 2], [8, 2], [4, 4], [4, 2, 2], [8], [4, 2]) if has(m, tuple(0 for _ in m))]
    line(f"for y=0 only Z_4^2 has connected sets (groups with sets for y=0: {y0})", y0 == [[4, 4]])
    t2 = {tuple(m): any(res[(tuple(m), y)]['II'] > 0 for y in A2(m) if any(y)) for m in ([4, 4], [8, 2])}
    line(f"Type II sets occur for Z_4^2 and Z_8 x Z_2 with some y != 0 ({t2})", all(t2.values()))
    mixed = sum(r['mixed'] for r in res.values())
    line(f"no solution is neither Type I nor Type II (Theorem gd:fourier(b)) (count {mixed})", mixed == 0)
    return ok


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--validate"]:
        sys.exit(0 if validate() else 1)
    groups = [[int(x) for x in a.split(",")] for a in args] if args else GROUPS
    res = full(groups)
    if not args:
        sys.exit(0 if claims(res) else 1)
