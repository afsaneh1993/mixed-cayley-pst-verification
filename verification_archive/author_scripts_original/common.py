r"""Shared tools: groups A x_f Z2, mixed Cayley graphs, exact K^2 test, automorphism order.

Conventions follow the manuscript: elements (a, r) stand for x^r a, x^2 = y,
x^{-1} a x = f(a); Cay(G,S) has H_{g,h} = 1, i, -i, 0 according as g^{-1}h lies in
S cap S^{-1}, S \ S^{-1}, (S \ S^{-1})^{-1}, or elsewhere.
"""
import sys
from itertools import product
from collections import Counter
import numpy as np

sys.setrecursionlimit(100000)


class Group:
    def __init__(self, mods, f, y):
        self.mods = tuple(mods)
        self.f = f
        self.y = tuple(y)
        self.A = list(product(*[range(m) for m in self.mods]))
        self.els = [(a, r) for r in (0, 1) for a in self.A]
        self.idx = {e: i for i, e in enumerate(self.els)}
        self.one = (tuple(0 for _ in self.mods), 0)

    def add(self, a, b):
        return tuple((x + z) % m for x, z, m in zip(a, b, self.mods))

    def neg(self, a):
        return tuple((-x) % m for x, m in zip(a, self.mods))

    def mul(self, g, h):
        (a, r), (b, s) = g, h
        if r == 0 and s == 0:
            return (self.add(a, b), 0)
        if r == 0 and s == 1:
            return (self.add(self.f(a), b), 1)
        if r == 1 and s == 0:
            return (self.add(a, b), 1)
        return (self.add(self.add(self.y, self.f(a)), b), 0)

    def inv(self, g):
        a, r = g
        if r == 0:
            return (self.neg(a), 0)
        return (self.neg(self.add(self.f(a), self.y)), 1)


def cyclic_dihedral(n):  # D_n of order 2n: A=Z_n, f=-id, y=0
    return Group([n], lambda a: ((-a[0]) % n,), (0,))


def dicyclic(n):  # Dic_n of order 4n: A=Z_2n, f=-id, y=n
    return Group([2 * n], lambda a: ((-a[0]) % (2 * n),), (n,))


def semidihedral(k):  # SD_{2^k}: A=Z_{2^(k-1)}, f=(2^(k-2)-1)*, y=0
    m = 2 ** (k - 1)
    t = 2 ** (k - 2) - 1
    return Group([m], lambda a: ((t * a[0]) % m,), (0,))


def generalized_dihedral(mods):  # Dih(A): f=-id, y=0
    return Group(mods, lambda a: tuple((-x) % m for x, m in zip(a, mods)),
                 tuple(0 for _ in mods))


def weights(G, S):
    """Exponent of i for every nonzero entry of the row of H at the identity:
    0 -> 1 (undirected), 1 -> i (arc), 3 -> -i (reverse arc)."""
    Ss = set(S)
    w = {}
    for s in S:
        w[s] = 0 if G.inv(s) in Ss else 1
    for s in S:
        if G.inv(s) not in Ss:
            w[G.inv(s)] = 3
    return w


_I = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # i^0..i^3 as Gaussian integers


def square_is_scalar(G, S, k):
    """Exact test: kappa^2 = k^2 * identity in Z[i][G]  <=>  K^2 = k^2 I."""
    w = weights(G, S)
    acc = {}
    for g, p in w.items():
        for h, q in w.items():
            z = G.mul(g, h)
            re, im = _I[(p + q) % 4]
            a = acc.get(z, (0, 0))
            acc[z] = (a[0] + re, a[1] + im)
    for z, (re, im) in acc.items():
        want = (k * k, 0) if z == G.one else (0, 0)
        if (re, im) != want:
            return False
    return G.one in acc or k == 0


def generates(G, S):
    seen = {G.one}
    frontier = [G.one]
    while frontier:
        nxt = []
        for g in frontier:
            for s in S:
                h = G.mul(g, s)
                if h not in seen:
                    seen.add(h)
                    nxt.append(h)
        frontier = nxt
    return len(seen) == len(G.els)


def hermitian(G, S):
    Ss = set(S)
    N = len(G.els)
    H = np.zeros((N, N), complex)
    for g in G.els:
        gi = G.inv(g)
        for h in G.els:
            d = G.mul(gi, h)
            if d in Ss:
                H[G.idx[g], G.idx[h]] = 1 if G.inv(d) in Ss else 1j
            elif G.inv(d) in Ss:
                H[G.idx[g], G.idx[h]] = -1j
    return H


def right_translation(G, c):
    N = len(G.els)
    R = np.zeros((N, N), complex)
    for g in G.els:
        R[G.idx[G.mul(g, c)], G.idx[g]] = 1
    return R


def abelian_test_passes(G, S):
    """Lemma (abtest): returns True if the necessary condition for being a Cayley
    graph of an abelian group holds, i.e. multisets {ts} and {st} coincide."""
    Ss = set(S)
    Su = [s for s in S if G.inv(s) in Ss]
    T = [s for s in S if G.inv(s) not in Ss]
    a = Counter(G.mul(t, s) for s in Su for t in T)
    b = Counter(G.mul(s, t) for s in Su for t in T)
    return a == b


def aut_order(M):
    """|Aut| of a vertex-transitive colored digraph given by an integer matrix M
    (equal entries = same colour): (number of automorphisms fixing vertex 0) * N.
    Backtracking search; suitable for graphs on up to ~40 vertices."""
    N = len(M)
    order = [0]
    seen = {0}
    for u in order:
        for v in range(N):
            if v not in seen and (M[u][v] != 0 or M[v][u] != 0):
                seen.add(v)
                order.append(v)
    order += [v for v in range(N) if v not in seen]
    img = [-1] * N
    used = [False] * N
    count = 0

    def rec(i):
        nonlocal count
        if i == N:
            count += 1
            return
        u = order[i]
        for w in range(N):
            if used[w] or (i == 0 and w != 0):
                continue
            if M[u][u] != M[w][w]:
                continue
            ok = True
            for j in range(i):
                v = order[j]
                if M[u][v] != M[w][img[v]] or M[v][u] != M[img[v]][w]:
                    ok = False
                    break
            if ok:
                img[u] = w
                used[w] = True
                rec(i + 1)
                used[w] = False
                img[u] = -1

    rec(0)
    return count * N


def colour_matrix(H, undirected=False):
    N = len(H)
    code = {0: 0, 1: 1, 1j: 2, -1j: 3}
    out = []
    for i in range(N):
        row = []
        for j in range(N):
            z = complex(round(H[i, j].real), round(H[i, j].imag))
            row.append((1 if z != 0 else 0) if undirected else code[z])
        out.append(row)
    return out
