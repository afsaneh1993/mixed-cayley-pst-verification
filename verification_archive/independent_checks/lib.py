import itertools, numpy as np
from scipy.linalg import expm
import pynauty

class Grp:
    """G = A x_f Z2, A = prod Z_{m_i}, elements (a,r) meaning x^r a, x^2=y, x^-1 a x = f(a)."""
    def __init__(self, mods, f, y):
        self.mods = tuple(mods)
        self.Aels = list(itertools.product(*[range(m) for m in mods]))
        self.f = f; self.y = tuple(y)
        self.els = [(a,r) for r in (0,1) for a in self.Aels]
        self.idx = {e:i for i,e in enumerate(self.els)}
        self.n = len(self.els)
        add = lambda a,b: tuple((p+q)%m for p,q,m in zip(a,b,self.mods))
        self.add = add
        neg = lambda a: tuple((-p)%m for p,m in zip(a,self.mods))
        self.neg = neg
        def mul(g,h):
            (a,r),(b,s) = g,h
            if r==0: return (add(f(a),b) if s==1 else add(a,b), s)   # (a,0)(b,1)=(f(a)+b,1)
            else:
                if s==0: return (add(a,b),1)
                return (add(add(self.y,f(a)),b),0)
        # careful: (a,0)(b,0)=(a+b,0); (a,0)(b,1)=(f(a)+b,1)
        def mul2(g,h):
            (a,r),(b,s)=g,h
            if r==0 and s==0: return (add(a,b),0)
            if r==0 and s==1: return (add(f(a),b),1)
            if r==1 and s==0: return (add(a,b),1)
            return (add(add(self.y,f(a)),b),0)
        self.mul = mul2
        self.mt = np.array([[self.idx[mul2(g,h)] for h in self.els] for g in self.els])
        self.one = self.idx[(tuple(0 for _ in mods),0)]
        self.inv = np.zeros(self.n,dtype=int)
        for i in range(self.n):
            j = int(np.where(self.mt[i]==self.one)[0][0]); self.inv[i]=j
        self.check_assoc()
    def check_assoc(self):
        n=self.n; mt=self.mt
        for _ in range(3000):
            a,b,c = np.random.randint(0,n,3)
            assert mt[mt[a,b],c]==mt[a,mt[b,c]]
    def center(self):
        return [i for i in range(self.n) if all(self.mt[i,j]==self.mt[j,i] for j in range(self.n))]
    def H(self, S):
        """S: iterable of element indices (arbitrary subset of G\\{1}); H[g,h]= 1,i,-i,0 per g^-1 h in S_u,T,T^-1"""
        S=set(S); n=self.n
        H=np.zeros((n,n),dtype=complex)
        for g in range(n):
            ginv=self.inv[g]
            for s in S:
                h=self.mt[g,s]
                if self.inv[s] in S: H[g,h]=1
                else: H[g,h]=1j
        # fix Hermitian: arcs g->h get i at (g,h); then (h,g) must be -i
        for g in range(n):
            for h in range(n):
                if H[g,h]==1j and H[h,g]==0: H[h,g]=-1j
        assert np.allclose(H,H.conj().T)
        return H
    def Lmat(self,c):
        n=self.n; L=np.zeros((n,n))
        for v in range(n): L[self.mt[c,v],v]=1
        return L
    def Rmat(self,c):
        n=self.n; R=np.zeros((n,n))
        for v in range(n): R[self.mt[v,c],v]=1
        return R

def digraph_aut_order(H):
    n=H.shape[0]
    adj={i:[] for i in range(n)}
    for g in range(n):
        for h in range(n):
            if abs(H[g,h])>1e-9: adj[g].append(h)   # H_gh=1 gives both directions; i gives one direction g->h (and -i entry at (h,g) skip)
    # arcs: H[g,h]=i  => directed g->h only. H[h,g]=-i => skip
    adj={i:[] for i in range(n)}
    for g in range(n):
        for h in range(n):
            if abs(H[g,h]-1)<1e-9 or abs(H[g,h]-1j)<1e-9: adj[g].append(h)
    gr=pynauty.Graph(n,directed=True,adjacency_dict=adj)
    gens,grpsize1,grpsize2,orbits,norb=pynauty.autgrp(gr)
    return int(round(grpsize1*10**grpsize2))

def digraph_cert(H):
    n=H.shape[0]
    adj={i:[] for i in range(n)}
    for g in range(n):
        for h in range(n):
            if abs(H[g,h]-1)<1e-9 or abs(H[g,h]-1j)<1e-9: adj[g].append(h)
    gr=pynauty.Graph(n,directed=True,adjacency_dict=adj)
    return pynauty.certificate(gr)

def gen_subgroup(G, S):
    seen={G.one}; frontier=[G.one]
    S=list(S)
    while frontier:
        nf=[]
        for g in frontier:
            for s in S:
                h=G.mt[g,s]
                if h not in seen: seen.add(h); nf.append(h)
        frontier=nf
    return seen
