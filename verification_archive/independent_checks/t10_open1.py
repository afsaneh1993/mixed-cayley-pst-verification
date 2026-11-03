"""Check of the numerical statement in Open problem 1: connected sets S (zeta>0) containing a central involution c with
c in <S'> (S'=S\\{c}) and spec(K), K=H-L_c, inside one residue class mod 4 (=> PST at pi/2 by Theorem pi2) and >=3 distinct eigenvalues of K."""
import numpy as np, sys
from lib import *
from t5_tabnn import groups
def run(name):
    mods,f,y=groups()[name]; G=Grp(mods,f,y); n=G.n
    cen=[z for z in G.center() if z!=G.one and G.mt[z,z]==G.one]
    nonid=[s for s in range(n) if s!=G.one]; cnt=0; seen=set()
    for mask in range(1<<(n-1)):
        S=[nonid[i] for i in range(n-1) if mask>>i&1]; Sset=set(S)
        for c in cen:
            if c not in Sset: continue
            Sp=[s for s in S if s!=c]
            H=G.H(S); K=H-G.Lmat(c); w=np.linalg.eigvalsh(K)
            if np.max(np.abs(w-np.round(w)))>1e-6: continue
            wi=np.round(w).astype(int)
            if len(set(wi%4))!=1: continue
            if len(set(wi))<3: continue
            if len(gen_subgroup(G,S))!=n: continue
            if c not in gen_subgroup(G,Sp if Sp else [G.one]): continue
            seen.add(mask)
    return len(seen)
for nm in sys.argv[1:]: print(nm,"sets:",run(nm),flush=True)
