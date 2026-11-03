"""EXACT (integer-arithmetic) enumeration of all S' subset G\\{1,c} with K^2 = k^2 I, K = sum_{s in S'} w_s rho(s),
w_s in {1, i, -i}. Gaussian integers are carried as two int64 arrays, so every comparison is exact.
Only pruning: the number of non-zero entries of a row of K equals k^2 (diagonal of K^2 = k^2 I)."""
from lib import *
import numpy as np, itertools, sys, time
def classes(G, exclude):
    seen=set(); pairs=[]; invols=[]
    for s in range(G.n):
        if s in exclude or s in seen: continue
        t=int(G.inv[s]); seen.add(s); seen.add(t)
        (invols if t==s else pairs).append(s if t==s else (s,t))
    return pairs, invols
def enum_weighing_exact(G,c,k,batch=20000,prune_u=None):
    """prune_u: optional (u, a): additionally require exactly u undirected elements and a arcs (follows from K^2=k^2 I, see README)."""
    pairs,invols=classes(G,{G.one,c}); n=G.n
    Q=np.array([[G.mt[G.inv[s],g] for g in range(n)] for s in range(n)])
    target=k*k; sols=[]; tot=0
    ni=len(invols)
    def gen():
        for j in range(0,ni+1):
            if (target-j)%2: continue
            m=(target-j)//2
            if m<0 or m>len(pairs): continue
            for I in itertools.combinations(invols,j):
                for P in itertools.combinations(range(len(pairs)),m):
                    for opts in itertools.product((0,1,2),repeat=m):
                        if prune_u is not None:
                            u,a=prune_u
                            if j+2*sum(1 for o in opts if o==2)!=u: continue
                        yield I,P,opts
    buf=[]
    def flush():
        if not buf: return
        B=len(buf); Wr=np.zeros((B,n),dtype=np.int64); Wi=np.zeros((B,n),dtype=np.int64)
        for b,(I,P,opts) in enumerate(buf):
            for s in I: Wr[b,s]=1
            for p,o in zip(P,opts):
                s,t=pairs[p]
                if o==0: Wi[b,s]=1; Wi[b,t]=-1
                elif o==1: Wi[b,t]=1; Wi[b,s]=-1
                else: Wr[b,s]=1; Wr[b,t]=1
        Xr=Wr[:,Q]; Xi=Wi[:,Q]                      # W[s^-1 g]
        cr=(Wr[:,:,None]*Xr-Wi[:,:,None]*Xi).sum(axis=1)
        ci=(Wr[:,:,None]*Xi+Wi[:,:,None]*Xr).sum(axis=1)
        want=np.zeros(n,dtype=np.int64); want[G.one]=target
        good=np.all(cr==want[None,:],axis=1)&np.all(ci==0,axis=1)
        for b in np.where(good)[0]: sols.append(buf[b])
        buf.clear()
    for item in gen():
        buf.append(item); tot+=1
        if len(buf)>=batch: flush()
    flush()
    return sols,pairs,tot
def to_set(sol,pairs):
    I,P,opts=sol; Sp=set(I)
    for p,o in zip(P,opts):
        s,t=pairs[p]
        if o==0: Sp.add(s)
        elif o==1: Sp.add(t)
        else: Sp|={s,t}
    return Sp
if __name__=="__main__":
    mode=sys.argv[1]; ns=list(map(int,sys.argv[2:]))
    for n in ns:
        t0=time.time()
        G=Grp([2*n],lambda a,n=n:((-a[0])%(2*n),),(n,)); c=G.idx[((n,),0)]
        pr=(4,6) if mode=="pruned" else None
        sols,pairs,tot=enum_weighing_exact(G,c,4,prune_u=pr)
        conn=sum(1 for s in sols if len(gen_subgroup(G,list(to_set(s,pairs))+[c]))==G.n)
        print("Dic_%d [%s, exact integer arithmetic]: candidates tested %d, weight-4 weighing sets %d, connected %d (%.1fs)"%(n,mode,tot,len(sols),conn,time.time()-t0),flush=True)
