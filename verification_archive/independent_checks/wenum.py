from lib import *
import numpy as np, itertools, sys, time

def classes(G, exclude):
    seen=set(); pairs=[]; invols=[]
    for s in range(G.n):
        if s in exclude or s in seen: continue
        t=int(G.inv[s]); seen.add(s); seen.add(t)
        if t==s: invols.append(s)
        else: pairs.append((s,t))
    return pairs, invols

def enum_weighing(G,c,k, batch=20000):
    """all S' subset G\\{1,c} with K^2=k^2 I. Row nonzero count must be k^2."""
    pairs,invols=classes(G,{G.one,c})
    n=G.n
    Q=np.zeros((n,n),dtype=int)
    for s in range(n):
        for g in range(n): Q[s,g]=G.mt[G.inv[s],g]
    target=k*k
    sols=[]
    # number of nonzero entries: invol in S' -> 1 ; pair option nonempty -> 2
    ni=len(invols)
    # choose subset of involutions I (size j) and m=(target-j)/2 nonempty pairs
    def gen():
        for j in range(0,ni+1):
            if (target-j)%2: continue
            m=(target-j)//2
            if m>len(pairs) or m<0: continue
            for I in itertools.combinations(invols,j):
                for P in itertools.combinations(range(len(pairs)),m):
                    for opts in itertools.product((0,1,2),repeat=m):
                        yield I,P,opts
    buf=[];meta=[]
    def flush():
        if not buf: return
        W=np.zeros((len(buf),n),dtype=complex)
        for b,(I,P,opts) in enumerate(buf):
            for s in I: W[b,s]=1
            for p,o in zip(P,opts):
                s,t=pairs[p]
                if o==0: W[b,s]=1j; W[b,t]=-1j      # arc s: H entry i at g->gs ; weight i on s, -i on s^-1
                elif o==1: W[b,t]=1j; W[b,s]=-1j
                else: W[b,s]=1; W[b,t]=1
        conv=(W[:,:,None]*W[:,Q]).sum(axis=1)      # coefficient of g in kappa^2
        good=np.all(np.abs(conv-np.where(np.arange(n)==G.one,target,0)[None,:])<1e-9,axis=1)
        for b in np.where(good)[0]:
            sols.append(buf[b])
        buf.clear()
    tot=0
    for item in gen():
        buf.append(item); tot+=1
        if len(buf)>=batch: flush()
    flush()
    return sols,pairs,tot

def dic(n):
    return Grp([2*n], lambda a:((-a[0])%(2*n),), (n,))
if __name__=="__main__":
    for n in map(int,sys.argv[1:]):
        t0=time.time()
        G=dic(n); c=G.idx[((n,),0)]
        assert c in G.center() 
        sols,pairs,tot=enum_weighing(G,c,4)
        conn=0; typeI=0
        for (I,P,opts) in sols:
            Sp=set(I)
            for p,o in zip(P,opts):
                s,t=pairs[p]
                if o==0: Sp.add(s)
                elif o==1: Sp.add(t)
                else: Sp|={s,t}
            gen=gen_subgroup(G,list(Sp)+[c])
            if len(gen)==G.n:
                conn+=1
        print("Dic_%d: candidates %d, weight-4 weighing sets %d, connected %d  (%.1fs)"%(n,tot,len(sols),conn,time.time()-t0),flush=True)
