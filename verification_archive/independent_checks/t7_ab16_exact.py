"""Table tab:ab for all nine groups of order 16 (and, by the same code, order <=12):
connected non-normal connection sets with PST (decided EXACTLY as in t5_exact.py), isomorphism classes of the resulting mixed
graphs (nauty), comparison with ALL mixed Cayley graphs of all abelian groups of the same order (nauty certificates),
and |Aut| of the classes that are not Cayley graphs of an abelian group (per group)."""
import numpy as np, sys, time
from math import gcd
from functools import reduce
from collections import Counter
from lib import *
from t5_tabnn import groups
IPOW=[(1,0),(0,-1),(-1,0),(0,1)]
def lcm(a,b): return a*b//gcd(a,b)
def pst_targets_exact(G,S,H,e1):
    n=G.n; w=np.linalg.eigvalsh(H)
    if np.max(np.abs(w-np.round(w)))>1e-6: return None
    ths=sorted(set(int(x) for x in np.round(w))); Sset=set(S); zeta=sum(1 for s in S if G.inv[s] in Sset)
    def apply(v,exclude=None):
        for t in ths:
            if t==exclude: continue
            v=H@v-t*v
        return v
    if np.any(apply(e1)!=0): raise RuntimeError("uncertified")
    a={t:apply(e1,exclude=t) for t in ths}; supp=[t for t in ths if np.any(a[t]!=0)]
    others=[t for t in supp if t!=zeta]
    if not others: return set()
    gam=reduce(gcd,[abs(t-zeta) for t in others]); d={t:int(np.prod([t-u for u in ths if u!=t])) for t in supp}
    D=reduce(lcm,[abs(d[t]) for t in supp])
    ar={t:[int(x) for x in np.rint(a[t].real)] for t in supp}; ai={t:[int(x) for x in np.rint(a[t].imag)] for t in supp}
    T=set()
    for l in range(1,5):
        Vr=[0]*n; Vi=[0]*n
        for t in supp:
            m=(t-zeta)//gam; pr,pi_=IPOW[(l*m)%4]; c=D//d[t]
            for j in range(n):
                xr,xi=ar[t][j]*c,ai[t][j]*c; Vr[j]+=pr*xr-pi_*xi; Vi[j]+=pr*xi+pi_*xr
        nz=[j for j in range(n) if Vr[j] or Vi[j]]
        if len(nz)==1 and nz[0]!=G.one: T.add(nz[0])
    return T
def abelian_groups(order):
    L={}
    if order==12:
        L['Z12']=((6,),lambda a:a,(1,)); L['Z6xZ2']=((6,),lambda a:a,(0,))
    if order==16:
        L['Z16']=((8,),lambda a:a,(1,)); L['Z8xZ2']=((8,),lambda a:a,(0,)); L['Z4xZ4']=((4,2),lambda a:a,(0,1))
        L['Z4xZ2xZ2']=((4,2),lambda a:a,(0,0)); L['Z2^4']=((2,2,2),lambda a:a,(0,0,0))
    if order==8:
        L['Z8']=((4,),lambda a:a,(1,)); L['Z4xZ2']=((4,),lambda a:a,(0,)); L['Z2^3']=((2,2),lambda a:a,(0,0))
    return L
def abelian_certs(order):
    certs=set()
    for nm,(mods,f,y) in abelian_groups(order).items():
        G=Grp(mods,f,y); nonid=[s for s in range(G.n) if s!=G.one]
        for mask in range(1,1<<(G.n-1)):
            S=[nonid[i] for i in range(G.n-1) if mask>>i&1]
            certs.add(digraph_cert(G.H(S)))
    return certs
def run(names,order):
    t0=time.time(); certs=abelian_certs(order); print("order",order,": isomorphism classes of mixed Cayley graphs of all abelian groups:",len(certs),"(%.0fs)"%(time.time()-t0),flush=True)
    for name in names:
        t0=time.time(); mods,f,y=groups()[name]; G=Grp(mods,f,y); n=G.n
        nonid=[s for s in range(n) if s!=G.one]; e1=np.zeros(n,dtype=complex); e1[G.one]=1
        conj=np.array([[G.mt[G.mt[G.inv[g],s],g] for s in range(n)] for g in range(n)])
        classes={}; nsets=0
        for mask in range(1<<(n-1)):
            S=[nonid[i] for i in range(n-1) if mask>>i&1]; Sset=set(S)
            if not any(G.inv[s] in Sset for s in S): continue
            H=G.H(S); T=pst_targets_exact(G,S,H,e1)
            if not T: continue
            if all(set(conj[g,s] for s in S)==Sset for g in range(n)): continue    # normal
            if len(gen_subgroup(G,S))!=n: continue
            nsets+=1; c=digraph_cert(H)
            if c not in classes: classes[c]=digraph_aut_order(H)
        ab=sum(1 for c in classes if c in certs); non=[v for c,v in classes.items() if c not in certs]
        dist=Counter(non)
        print(f"{name}: conn. non-normal PST sets {nsets}; classes {len(classes)}; with abelian Cayley group {ab}; without {len(non)}; |Aut| of those without: "+", ".join(f"{k}^({v})" for k,v in sorted(dist.items())) + "  (%.0fs)"%(time.time()-t0),flush=True)
if __name__=="__main__":
    order=int(sys.argv[1]); run(sys.argv[2:],order)
