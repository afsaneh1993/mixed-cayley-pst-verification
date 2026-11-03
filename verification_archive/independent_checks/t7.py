from lib import *
from t5_tabnn import groups
from t6 import pst_targets
import numpy as np
def allcerts_abelian(mods):
    G=Grp(mods,lambda a:a,tuple(0 for _ in mods))
    # abelian group of order n: use A=mods with x trivial -> order 2|A|; want order 12: mods=(6,) gives 12 = Z6xZ2 ; Z12 needs special
    return G
# build abelian groups of order 12 directly as A x_id Z2 with y chosen: Z12 = Z6 with y=1? y must be fixed elt; G=Z6 x_y Z2 with y=(1,) gives Z12
def abel(mods,y): return Grp(mods,lambda a:a,y)
Z12=abel((6,),(1,)); Z6Z2=abel((6,),(0,))
certs=set()
for G in (Z12,Z6Z2):
    nonid=[s for s in range(G.n) if s!=G.one]
    for mask in range(1<<(G.n-1)):
        S=[nonid[i] for i in range(G.n-1) if mask>>i&1]
        if not S: continue
        H=G.H(S); certs.add(digraph_cert(H))
print("abelian-Cayley mixed graph classes (iso, both groups):",len(certs))
for name in ("D6","Dic3"):
    mods,f,y=groups()[name]; G=Grp(mods,f,y); n=G.n
    nonid=[s for s in range(n) if s!=G.one]
    conj=np.array([[G.mt[G.mt[G.inv[g],s],g] for s in range(n)] for g in range(n)])
    classes={}
    for mask in range(1<<(n-1)):
        S=[nonid[i] for i in range(n-1) if mask>>i&1]; Sset=set(S)
        if not any(G.inv[s] in Sset for s in S): continue
        if not pst_targets(G,S): continue
        if any(set(conj[g,s] for s in S)!=Sset for g in range(n))==False: continue  # normal
        if len(gen_subgroup(G,S))!=n: continue
        H=G.H(S); c=digraph_cert(H)
        classes.setdefault(c,[0,digraph_aut_order(H)]); classes[c][0]+=1
    ab=sum(1 for c in classes if c in certs)
    print(name,"conn nonnormal PST sets",sum(v[0] for v in classes.values()),"classes",len(classes),"abelian",ab,"nonabelian",len(classes)-ab,"Aut of nonabelian:",sorted(v[1] for c,v in classes.items() if c not in certs))
