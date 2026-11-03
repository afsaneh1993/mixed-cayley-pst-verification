"""Extra checks missing from the author's verify_families_v24.py: for every explicit family, test
connectivity, the commutation obstruction of Lemma abtest (=> not a Cayley graph of any abelian group, when nonzero),
K^2=k^2 I, PST at pi/2 (U = lambda L_c), and |Aut| (nauty). Independent implementation (lib.py)."""
from lib import *
from scipy.linalg import expm
import numpy as np
from collections import Counter
def not_abelian_test(G,S):
    Sset=set(S)
    Su=[s for s in S if G.inv[s] in Sset]; T=[s for s in S if G.inv[s] not in Sset]
    A=Counter();B=Counter()
    for s in Su:
        for t in T: A[G.mt[t,s]]+=1; B[G.mt[s,t]]+=1
    return A!=B
def check(name,G,Sp,c,k,expect_aut=None):
    S=list(Sp)+[c]
    Hs=G.H(S); K=G.H(Sp)
    k2=np.allclose(K@K,k*k*np.eye(G.n))
    U=expm(-1j*np.pi/2*Hs); lam=U[c,G.one]
    pst=abs(abs(lam)-1)<1e-8 and np.allclose(U,lam*G.Lmat(c),atol=1e-7)
    conn=len(gen_subgroup(G,S))==G.n
    na=not_abelian_test(G,S)
    aut=digraph_aut_order(Hs) if G.n<=128 else None
    ok=k2 and pst and conn
    print(("OK  " if ok else "FAIL"),name,"K2=k2I",k2,"PST(pi/2)",pst,"connected",conn,"commutation-test-nonzero",na,"|Aut|",aut,
          ("(expected %s: %s)"%(expect_aut,aut==expect_aut) if expect_aut else ""),flush=True)
    return ok
allok=True
for n in range(6,17,2):
    G=Grp([n],lambda a,n=n:((-a[0])%n,),(0,)); e=lambda a,r:G.idx[((a%n,),r)]
    x=e(0,1); c=e(n//2,0)
    allok&=check("Gamma_%d on D_%d"%(n,n),G,[x,int(G.mt[x,e(2,0)]),e(1,0)],c,2,2*n)
for n in range(3,8):
    G=Grp([2*n],lambda a,n=n:((-a[0])%(2*n),),(n,)); e=lambda a,r:G.idx[((a%(2*n),),r)]
    x=e(0,1); c=e(n,0)
    allok&=check("Dic_%d (g=xa)"%n,G,[x,int(G.mt[x,c]),e(1,1)],c,2,n*2**(n+1))
for kk in (4,5,6):
    m=2**(kk-1); mult=2**(kk-2)-1
    G=Grp([m],lambda a,m=m,mult=mult:((mult*a[0])%m,),(0,)); e=lambda a,r:G.idx[((a%m,),r)]
    x=e(0,1); c=e(2**(kk-2),0)
    allok&=check("SD_%d"%2**kk,G,[x,int(G.mt[x,c]),e(1,1)],c,2)
for m in range(2,11):
    mods=(4,4*m)
    G=Grp(mods,lambda a,m=m:((-a[0])%4,(-a[1])%(4*m)),(0,0)); e=lambda a,r:G.idx[(tuple(a),r)]
    T=[(0,1),(0,m),(1,0),(1,2*m),(2,1),(2,3*m)]; R=[(0,0),(2,0),(0,2),(2,2)]
    for c in [(2,0),(0,2*m),(2,2*m)]:
        allok&=check("Dih(Z4xZ%d) c=%s"%(4*m,c),G,[e(r,1) for r in R]+[e(t,0) for t in T],e(c,0),4)
print("ALL FAMILY CHECKS PASSED" if allok else "SOME CHECK FAILED")
