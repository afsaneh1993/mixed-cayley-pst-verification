"""EXACT recomputation of the columns of Table tab:nn (all connection sets S with zeta>0).
For every set whose float spectrum is within 1e-6 of integers:
  (1) the integer spectrum is certified exactly: prod_theta (H - theta) e_1 = 0 (integer-valued arithmetic, == comparison);
  (2) E_theta e_1 = N_theta e_1 / d_theta with N_theta = prod_{theta'!=theta}(H-theta'), d_theta = prod (theta-theta') (exact integers);
  (3) U(t) e_1 at t = pi*l/(2*gamma), l=1..4 equals e^{-i t zeta} * V_l with V_l = sum_theta (-i)^{l m_theta} E_theta e_1,
      m_theta=(theta-zeta)/gamma; PST from 1 to z at such a time  <=>  V_l has exactly one non-zero entry, at z != 1 (U unitary);
      exact Gaussian-integer arithmetic (Python ints) for V_l.
By Theorem G(e) (proved in the paper) every PST time has the form pi*l/(2 gamma) with 4 not dividing l; l=4 is the full period 2 pi/gamma. Integrality of the spectrum under PST is Theorem G(b) (proved in the paper),
so sets with non-integral float spectrum are taken to have no PST.
All numpy values are integer-valued and asserted < 2^50, hence exact in double precision."""
import numpy as np, math, sys, time
from math import gcd
from functools import reduce
from lib import *
from t5_tabnn import groups
IPOW=[(1,0),(0,-1),(-1,0),(0,1)]            # (-i)^k for k mod 4
def lcm(a,b): return a*b//gcd(a,b)
def analyse_exact(name):
    mods,f,y=groups()[name]; G=Grp(mods,f,y); n=G.n
    Aidx=set(G.idx[(a,0)] for a in G.Aels); center=set(G.center())
    def order(g):
        k=1;h=g
        while h!=G.one: h=G.mt[h,g];k+=1
        return k
    ords=[order(g) for g in range(n)]
    nonid=[s for s in range(n) if s!=G.one]
    conj=np.array([[G.mt[G.mt[G.inv[g],s],g] for s in range(n)] for g in range(n)])
    st=dict(sets=0,pst=0,nonnormal=0,ord2=0,ord4=0,noncentral=0,othercoset=0,conn=0,conn_nn=0,pst_normal=0)
    uncertified=0; integral_sets=0; viol={'not_in_normaliser':0,'order':0,'S1st_not_cyclic':0}
    e1=np.zeros(n,dtype=complex); e1[G.one]=1
    for mask in range(1<<(n-1)):
        S=[nonid[i] for i in range(n-1) if mask>>i&1]; Sset=set(S)
        if not any(G.inv[s] in Sset for s in S): continue
        st['sets']+=1
        H=G.H(S)
        w=np.linalg.eigvalsh(H)
        if np.max(np.abs(w-np.round(w)))>1e-6: continue
        integral_sets+=1
        ths=sorted(set(int(x) for x in np.round(w)))
        zeta=sum(1 for s in S if G.inv[s] in Sset)
        def apply(v,exclude=None):
            for t in ths:
                if t==exclude: continue
                v=H@v-t*v
                assert np.max(np.abs(v))<2**50
            return v
        if np.any(apply(e1)!=0): uncertified+=1; continue          # spectrum not exactly certified
        a={t:apply(e1,exclude=t) for t in ths}
        supp=[t for t in ths if np.any(a[t]!=0)]
        if zeta not in supp: uncertified+=1; continue
        others=[t for t in supp if t!=zeta]
        if not others: continue
        gam=reduce(gcd,[abs(t-zeta) for t in others])
        d={t:int(np.prod([t-u for u in ths if u!=t])) for t in supp}
        D=reduce(lcm,[abs(d[t]) for t in supp])
        ar={t:[int(x) for x in np.rint(a[t].real)] for t in supp}; ai={t:[int(x) for x in np.rint(a[t].imag)] for t in supp}
        assert all(np.all(a[t].real==np.rint(a[t].real)) and np.all(a[t].imag==np.rint(a[t].imag)) for t in supp)
        targets=set(); first=None
        for l in range(1,5):
            Vr=[0]*n; Vi=[0]*n
            for t in supp:
                m=(t-zeta)//gam; pr,pi_=IPOW[(l*m)%4]; c=D//d[t]
                for j in range(n):
                    xr,xi=ar[t][j]*c,ai[t][j]*c
                    Vr[j]+=pr*xr-pi_*xi; Vi[j]+=pr*xi+pi_*xr
            nz=[j for j in range(n) if Vr[j]!=0 or Vi[j]!=0]
            if len(nz)==1 and nz[0]!=G.one:
                targets.add(nz[0])
                if first is None: first=nz[0]
        if not targets: continue
        # exact structural checks of the paper: z in N_G(S), ord(z) in {2,4}, S_1^st = <z0> cyclic of order 2 or 4 (z0 = target at least PST time)
        for z in targets:
            if set(conj[z,s] for s in S)!=Sset: viol['not_in_normaliser']+=1
            if ords[z] not in (2,4): viol['order']+=1
        cyc=gen_subgroup(G,[first])
        if cyc!=(targets|{G.one}) or len(cyc) not in (2,4): viol['S1st_not_cyclic']+=1
        st['pst']+=1
        normal=all(set(conj[g,s] for s in S)==Sset for g in range(n))
        isconn=len(gen_subgroup(G,S))==n
        if isconn: st['conn']+=1
        if normal:
            st['pst_normal']+=1
            assert all(z in center for z in targets)
        else:
            st['nonnormal']+=1
            if isconn: st['conn_nn']+=1
            for z in targets:
                if ords[z]==2: st['ord2']+=1
                elif ords[z]==4: st['ord4']+=1
                else: print("UNEXPECTED order",ords[z])
                if z not in center: st['noncentral']+=1
                if z not in Aidx: st['othercoset']+=1
    st['_integral_sets']=integral_sets; st['_uncertified']=uncertified; st['_violations']=viol
    return st
if __name__=="__main__":
    for name in sys.argv[1:]:
        t0=time.time(); r=analyse_exact(name)
        print(name,{k:v for k,v in r.items() if not k.startswith('_')},"| integral-spectrum sets:",r['_integral_sets'],"uncertified:",r['_uncertified'],"structural-check violations:",r['_violations'],"| %.0fs"%(time.time()-t0),flush=True)
