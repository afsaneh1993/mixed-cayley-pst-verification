"""Exact check of Corollary gd:dicn (independent implementation): on Dic_n (A=Z_{2n}, f=-id, y=n, c=(n,0)) the set S' given by
R_u={0,n,2,n+2}, T={1,2,n-2,n+1}, R_a={a,a+4} (a=1 for n>=6, a=4 for n=5) satisfies the hypotheses of Proposition gd:k4 and K^2=16 I,
tested in exact arithmetic over Z[i] (Gaussian integers as pairs of Python ints), for 5<=n<=60."""
from lib import *
import sys
def check(n):
    G=Grp([2*n],lambda a:((-a[0])%(2*n),),(n,)); m=2*n
    e=lambda a,r:G.idx[((a%m,),r)]
    a0=4 if n==5 else 1
    Ru=[0,n,2,n+2]; T=[1,2,n-2,n+1]; Ra=[a0,a0+4]
    # hypotheses: 2rho notin {0,y}, T distinct, T cap -T empty, 2u notin {0,y}, a,a+2u notin R_u
    hyp = (2%m not in (0,n)) and len(set(T))==4 and all((-t)%m not in T for t in T) and (4%m not in (0,n)) and all(r%m not in [x%m for x in Ru] for r in Ra)
    Su=[e(r,1) for r in Ru]; Ar=[e(t,0) for t in T]+[e(r,1) for r in Ra]
    c=e(n,0)
    allel=Su+Ar
    distinct=len(set(allel))==len(allel) and c not in allel and G.one not in allel and all(int(G.inv[t]) not in set(Ar) for t in Ar)
    w={}
    for s in Su: w[s]=(1,0)
    for t in Ar: w[t]=(0,1); w[int(G.inv[t])]=(0,-1)
    acc={}
    for s,(a,b) in w.items():
        for t,(c2,d) in w.items():
            g=int(G.mt[s,t]); x,y_=acc.get(g,(0,0)); acc[g]=(x+a*c2-b*d,y_+a*d+b*c2)
    ok=acc.get(G.one)==(16,0) and all(v==(0,0) for g,v in acc.items() if g!=G.one)
    return hyp and distinct and ok
res=[(n,check(n)) for n in range(5,61)]
print("Corollary gd:dicn, exact Z[i] arithmetic: hypotheses + K^2=16I hold for all n in 5..60:",all(r for _,r in res),"; failures:",[n for n,r in res if not r])
