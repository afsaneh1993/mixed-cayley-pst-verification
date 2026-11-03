from lib import *
import numpy as np
from scipy.linalg import expm
res=[]
for n in range(4,21,2):
    G=Grp([n], lambda a:((-a[0])%n,), (0,))
    el=lambda a,r: G.idx[((a%n,),r)]
    ah=el(n//2,0); x=el(0,1); xa2=el(2,1); a1=el(1,0)
    S=[ah,x,xa2,a1]
    H=G.H(S)
    c=ah
    K=H-G.Lmat(c)
    k2 = np.allclose(K@K, 4*np.eye(G.n))
    ev=np.round(np.linalg.eigvalsh(H),6)
    from collections import Counter
    cnt=Counter(ev)
    U=expm(-1j*np.pi/2*H)
    ok_U=np.allclose(U, 1j*G.Rmat(c))
    # least PST time: scan t in (0,pi/2) for |U_{vu}|=1 any v!=u
    best=None
    for t in np.linspace(0.001,np.pi/2-0.001,3000):
        pass
    # exact: times pi*l/4
    pst=[]
    for l in range(1,9):
        Ut=expm(-1j*np.pi*l/4*H)
        col=np.abs(Ut[:,G.one])
        v=np.argmax(col); 
        if col[v]>1-1e-8 and v!=G.one: pst.append(l)
    # edge/arc counts
    nedges=int(np.sum(np.abs(H-1)<1e-9))//2; narcs=int(np.sum(np.abs(H-1j)<1e-9))
    # abelian test lemma
    Su=[s for s in S if G.inv[s] in S]; T=[s for s in S if G.inv[s] not in S]
    from collections import Counter as C
    A=C(); B=C()
    for s in Su:
        for t in T:
            A[G.mt[t,s]]+=1; B[G.mt[s,t]]+=1
    comm_diff = (A!=B)
    aut = digraph_aut_order(H) if n<=16 else None
    # undirected underlying aut
    Hu=(np.abs(H)>0).astype(complex)
    autu=digraph_aut_order(Hu) if n<=12 else None
    print(n,"K2=4I",k2,"spec",dict(cnt),"U=iR_c",ok_U,"PSTl",pst,"edges",nedges,"arcs",narcs,"nonabelian-test",comm_diff,"Aut",aut,"AutUnd",autu)
