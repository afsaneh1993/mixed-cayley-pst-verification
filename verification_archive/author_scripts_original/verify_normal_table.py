"""Reproduces Table S1 (normal connection sets with S cap S^-1 nonempty): numbers of such sets,
with integral spectrum, with PST, and with a PST target of order 4; also checks that every target
is central.  Sets are unions of conjugacy classes of G\\{1}.  U_{6n} = <a,b | a^{2n}=b^3=1, a^-1 b a = b^-1>
is built as A x| Z2 with A=Z_n x Z_3, f(i,j)=(i,-j), y=(1,0).  About 1 minute."""
import sys, math, random
import numpy as np
from common import *
from verify_nn_tables import GROUPS


def U6n(n):
    return Group([n, 3], lambda a: (a[0], (-a[1]) % 3), (1, 0))


TABLE = {"S3": (3, 3, 0, 0), "D4": (15, 15, 8, 0), "Q8": (15, 15, 8, 0), "D5": (7, 3, 0, 0), "D7": (15, 3, 0, 0),
         "D6": (31, 31, 2, 0), "Dic3": (29, 29, 2, 0), "D8": (63, 31, 16, 0), "Q16": (63, 31, 16, 0),
         "SD16": (61, 31, 16, 0), "M16": (485, 485, 254, 24), "Z4sdZ4": (503, 503, 400, 0),
         "Z2xD4": (511, 511, 384, 0), "Z2xQ8": (511, 511, 384, 0), "Z4oD4": (509, 509, 254, 16),
         "Z4Z2sdZ2": (503, 503, 400, 0), "U18": (229, 31, 0, 0), "U24": (1967, 485, 26, 2),
         "U30": (15655, 31, 0, 0)}
GR = dict(GROUPS)
GR.update({"U18": lambda: U6n(3), "U24": lambda: U6n(4), "U30": lambda: U6n(5)})


def order(G, g):
    k, h = 1, g
    while h != G.one:
        h = G.mul(h, g)
        k += 1
    return k


def run(G):
    els = G.els
    seen, classes = set(), []
    for g in els:
        if g == G.one or g in seen:
            continue
        cl = {G.mul(G.mul(h, g), G.inv(h)) for h in els}
        seen |= cl
        classes.append(sorted(cl))
    Z = {z for z in els if all(G.mul(z, g) == G.mul(g, z) for g in els)}
    nS = nI = nP = n4 = 0
    central = True
    for mask in range(1 << len(classes)):
        S = [g for i, cl in enumerate(classes) if mask >> i & 1 for g in cl]
        Ss = set(S)
        if not any(G.inv(s) in Ss for s in S):
            continue
        nS += 1
        w, V = np.linalg.eigh(hermitian(G, S))
        if np.max(np.abs(w - np.round(w))) > 1e-7:
            continue
        nI += 1
        zeta = sum(1 for s in S if G.inv(s) in Ss)
        d = 0
        for t in np.round(w).astype(int):
            d = math.gcd(d, abs(int(t) - zeta))
        if d == 0:
            continue
        targets = set()
        for l in (1, 2, 3):
            U = (V * np.exp(-1j * math.pi * l / (2 * d) * w)) @ V.conj().T
            targets |= {j for j in range(1, len(els)) if abs(U[j, 0]) > 1 - 1e-7}
        if targets:
            nP += 1
            n4 += any(order(G, els[j]) == 4 for j in targets)
            central &= all(els[j] in Z for j in targets)
    return (nS, nI, nP, n4), central


if __name__ == "__main__":
    bad = 0
    for name in TABLE:
        got, cen = run(GR[name]())
        ok = got == TABLE[name]
        bad += not ok
        print(f"{name}: {got} table {TABLE[name]} {'OK' if ok else 'MISMATCH'}; all targets central: {cen}", flush=True)
    sys.exit(1 if bad else 0)
