from lib import *
import numpy as np, random
from collections import Counter

def abelian_test(G,S):
    Sset=set(S)
    Su=[s for s in S if G.inv[s] in Sset]; T=[s for s in S if G.inv[s] not in Sset]
    A=Counter();B=Counter()
    for s in Su:
        for t in T:
            A[G.mt[t,s]]+=1; B[G.mt[s,t]]+=1
    return A!=B   # True => not abelian Cayley

def K_from(G,Sp):
    return G.H(Sp)

# Corollary: Dic_n, n>=5
for n in range(5,17):
    G=Grp([2*n], lambda a,n=n:((-a[0])%(2*n),), (n,))
    el=lambda a,r: G.idx[((a%(2*n),),r)]
    c=el(n,0)
    rho,u=1,2
    a= 1 if n>=6 else 4
    Ru=[0,n,2*rho,2*rho+n]; T=[rho,rho+n,u,n-u]; Ra=[a,a+2*u]
    Sp=[el(r,1) for r in Ru]+[el(t,0) for t in T]+[el(r,1) for r in Ra]
    # admissibility: check data claims
    H=G.H(Sp); K=H  # Sp has no c
    k2=np.allclose(K@K,16*np.eye(G.n))
    S=Sp+[c]
    Hs=G.H(S)
    from scipy.linalg import expm
    U=expm(-1j*np.pi/2*Hs)
    ok=np.allclose(U,-1j*G.Rmat(c))
    ev=sorted(set(np.round(np.linalg.eigvalsh(Hs),5)))
    gen=len(gen_subgroup(G,S))==G.n
    zeta=sum(1 for s in S if G.inv[s] in set(S))
    na=abelian_test(G,S)
    print("Dic_%d"%n,"K2=16I",k2,"U=-iR_c",ok,"spec",ev,"connected",gen,"zeta",zeta,"notabelian(test)",na, "|Sp|",len(set(Sp)))
