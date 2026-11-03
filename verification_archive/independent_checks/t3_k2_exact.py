from lib import *
from wenum_exact import enum_weighing_exact as enum_weighing
import numpy as np, itertools, sys

def dicAy(mods,y):
    f=lambda a: tuple((-p)%m for p,m in zip(a,mods))
    return Grp(mods,f,y)

def run(mods):
    cnt_tot=0; cnt_bad=0
    A=list(itertools.product(*[range(m) for m in mods]))
    neg=lambda a: tuple((-p)%m for p,m in zip(a,mods))
    add=lambda a,b: tuple((p+q)%m for p,q,m in zip(a,b,mods))
    zero=tuple(0 for _ in mods)
    A2=[a for a in A if add(a,a)==zero]
    if len(A2)==len(A): return None
    for y in A2:
        G=dicAy(mods,y)
        for c in A2:
            if c==zero: continue
            ci=G.idx[(c,0)]
            sols,pairs,_=enum_weighing(G,ci,2)
            found=set()
            for (I,P,opts) in sols:
                Sp=set(I)
                for p,o in zip(P,opts):
                    s,t=pairs[p]
                    if o==0: Sp.add(s)
                    elif o==1: Sp.add(t)
                    else: Sp|={s,t}
                if len(gen_subgroup(G,list(Sp)+[ci]))==G.n: found.add(frozenset(Sp))
            # theorem list
            pred=set()
            el=lambda a,r: G.idx[(a,r)]
            def gens_ok(Sp): return len(gen_subgroup(G,list(Sp)+[ci]))==G.n
            for r in A:
                for t in A:
                    if y==zero:
                        if add(t,t)!=zero:  # B1
                            Sp={el(r,1),el(add(r,add(t,t)),1),el(t,0)}
                            if ci not in Sp and gens_ok(Sp): pred.add(frozenset(Sp))
                    else:
                        if add(t,t)==y: #B2a
                            Sp={el(r,1),el(add(r,y),1),el(t,0)}
                            if gens_ok(Sp): pred.add(frozenset(Sp))
                        for u in A:
                            if u not in (r,add(r,y)):
                                Sp={el(r,1),el(add(r,y),1),el(u,1)}
                                if gens_ok(Sp): pred.add(frozenset(Sp))
                        if add(r,r)==y:  #B3 with d=r: S'={(d,0),(-d,0),(u,1)}
                            for u in A:
                                Sp={el(r,0),el(neg(r),0),el(u,1)}
                                if ci not in Sp and gens_ok(Sp): pred.add(frozenset(Sp))
            cnt_tot+=1
            if found!=pred:
                cnt_bad+=1
                print("MISMATCH",mods,"y",y,"c",c,len(found),len(pred), "only found:",len(found-pred),"only pred:",len(pred-found))
    return cnt_tot,cnt_bad

total=0;bad=0
for mods in [(m,) for m in range(3,25)]+[(4,2),(6,2),(8,2),(4,4),(10,2),(12,2),(4,2,2),(6,4),(3,3)]:
    r=run(mods); print(mods,r,flush=True)
    if r: total+=r[0]; bad+=r[1]
print("TOTAL (A,y,c) triples checked:",total,"mismatches:",bad)
