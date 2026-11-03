"""Weight-4 solutions of  R R^(-1) + D D^(-1) = 16  in Z[Z_n]  (Lemma S1.3 with k=4, A = Z_n cyclic):
R = any 4-subset of Z_n (up to translation 0 in R), T = 6 elements from 6 distinct pairs {t,-t}, t != -t.
Equivalent Fourier form: |sum_R psi_j|^2 + 4 (sum_T Im psi_j)^2 = 16 for every character j.
Meet in the middle on rounded vectors.  A solution needs n >= 13 (six pairs).
Usage: python search_cyclic_k4.py LAST [FIRST=13].  The manuscript's bound n <= 64 is NOT reproduced here;
the runtime roughly doubles every 4 steps of n (n=43 takes about one minute)."""
import sys, time
import numpy as np
from itertools import combinations, product


def solutions(n):
    J = np.arange(n)
    ch = np.exp(2j * np.pi * np.outer(J, np.arange(n)) / n)
    pairs = [(a, n - a) for a in range(1, (n + 1) // 2) if 2 * a != n]
    if len(pairs) < 6:
        return 0
    table = {}
    for comb in combinations(range(1, n), 3):
        R = (0,) + comb
        v = np.abs(ch[:, list(R)].sum(axis=1)) ** 2
        table.setdefault(tuple(np.round(v, 5)), []).append(R)
    cnt = 0
    for comb in combinations(range(len(pairs)), 6):
        for sg in product((0, 1), repeat=6):
            T = [pairs[i][s] for i, s in zip(comb, sg)]
            s = ch[:, T].imag.sum(axis=1)
            cnt += len(table.get(tuple(np.round(16 - 4 * s * s, 5)), []))
    return cnt


if __name__ == "__main__":
    last = int(sys.argv[1]) if len(sys.argv) > 1 else 34
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 13
    for n in range(first, last + 1):
        t = time.time()
        print(f"n={n}: number of solutions = {solutions(n)}  ({time.time() - t:.1f} s)", flush=True)
