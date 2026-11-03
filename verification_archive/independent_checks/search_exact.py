"""Exact (integer) search for weight-4 solutions with y=0:  R R^(-1) + 2 D D^(-1) - D^2 - (D^(-1))^2 = 16 in Z[A],
R = 4-subset of A containing 0 (translation normalisation), D = sum of 6 elements of A taken from 6 distinct pairs {t,-t}, t != -t.
(This is the group-ring form of  |sum_R psi|^2 + 4 (sum_T Im psi)^2 = 16  for all characters psi of A.)
Meet in the middle on exact integer coefficient vectors; no rounding anywhere.
Usage: python3 search_exact.py cyc FIRST LAST   |   python3 search_exact.py ab m1 m2 ... (A = Z_m1 x Z_m2 x ...)"""
import sys, time, itertools, numpy as np
def setup(mods):
    els=list(itertools.product(*[range(m) for m in mods])); idx={e:i for i,e in enumerate(els)}; N=len(els)
    add=np.array([[idx[tuple((x+y)%m for x,y,m in zip(a,b,mods))] for b in els] for a in els])
    neg=np.array([idx[tuple((-x)%m for x,m in zip(a,mods))] for a in els])
    return els,idx,N,add,neg
def bincount_rows(idx2d,N):
    B=idx2d.shape[0]; off=(np.arange(B)*N)[:,None]
    return np.bincount((idx2d+off).ravel(),minlength=B*N).reshape(B,N)
def solutions(mods):
    els,idx,N,add,neg=setup(mods); zero=0
    pairs=[(a,int(neg[a])) for a in range(N) if a<neg[a]]
    if len(pairs)<6: return 0,[]
    # table of R R^(-1)
    table={}
    for comb in itertools.combinations(range(1,N),3):
        R=np.array((zero,)+comb); diff=add[R[:,None],neg[R][None,:]].ravel()
        v=np.bincount(diff,minlength=N).astype(np.int16)
        table.setdefault(v.tobytes(),[]).append(tuple(R))
    cnt=0; found=[]
    combs=np.array(list(itertools.combinations(range(len(pairs)),6)))
    signs=np.array(list(itertools.product((0,1),repeat=6)))
    P=np.array(pairs)
    for comb in combs:
        Tb=P[comb[None,:],signs]                       # 64 x 6 elements
        s=add[Tb[:,:,None],Tb[:,None,:]].reshape(64,-1)                    # t_i+t_j
        d=add[Tb[:,:,None],neg[Tb][:,None,:]].reshape(64,-1)               # t_i-t_j
        D2=bincount_rows(s,N); DD=bincount_rows(d,N)
        Dm2=bincount_rows(neg[s],N)
        V=-(2*DD-D2-Dm2); V[:,zero]+=16                # must equal R R^(-1)
        for b in range(64):
            key=V[b].astype(np.int16).tobytes()
            if key in table:
                cnt+=len(table[key]); found.append((tuple(Tb[b]),table[key]))
    return cnt,found
if __name__=="__main__":
    if sys.argv[1]=="cyc":
        for n in range(int(sys.argv[2]),int(sys.argv[3])+1):
            t=time.time(); c,_=solutions((n,)); print(f"A=Z_{n}: number of solutions = {c}  ({time.time()-t:.1f}s)",flush=True)
    else:
        mods=tuple(map(int,sys.argv[2:])); t=time.time(); c,f=solutions(mods)
        print(f"A=Z_{'xZ_'.join(map(str,mods))}: number of solutions = {c}  ({time.time()-t:.1f}s)")
        if f:
            els=setup(mods)[0]; T,Rs=f[0]
            print("  example: T =",[els[i] for i in T]," R =",[els[i] for i in Rs[0]])
