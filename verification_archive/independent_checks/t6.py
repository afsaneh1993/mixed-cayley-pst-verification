from lib import *
import numpy as np, itertools
from t5_tabnn import groups
# (1) Z_n mixed: which zeta>0 sets have PST (independent test via same method)
def pst_targets(G,S):
    n=G.n; H=G.H(S); w,V=np.linalg.eigh(H)
    if np.max(np.abs(w-np.round(w)))>1e-6: return []
    wi=np.round(w).astype(int); ths=sorted(set(wi)); Sset=set(S)
    zeta=sum(1 for s in S if G.inv[s] in Sset)
    E={th:V[:,wi==th]@V[:,wi==th].conj().T for th in ths}
    v1={th:E[th][:,G.one] for th in ths}
    supp=[th for th in ths if np.linalg.norm(v1[th])>1e-8]
    out=[]
    for z in range(n):
        if z==G.one: continue
        ok=True; mu={}
        for th in ths:
            vz=E[th][:,z]; a=v1[th]; na=np.linalg.norm(a)
            if na<1e-8:
                if np.linalg.norm(vz)>1e-8: ok=False;break
                continue
            m=np.vdot(a,vz)/np.vdot(a,a)
            if abs(abs(m)-1)>1e-7 or np.linalg.norm(vz-m*a)>1e-7: ok=False;break
            mu[th]=m
        if not ok: continue
        others=[th for th in supp if th!=zeta]
        if not others: continue
        th1=others[0]; d=th1-zeta
        for m_ in range(abs(d)):
            t=(-np.angle(mu[th1])+2*np.pi*m_)/d
            if all(abs(np.exp(-1j*t*(th-zeta))-mu[th])<1e-6 for th in supp): out.append(z);break
    return out
if __name__=="__main__":
    for n in (6,10,14):
        G=Grp([n],lambda a:a,(0,))   # f=id: cyclic group Z_n x Z2? need plain Z_n
        # build plain cyclic group via f=id and A=Z_{n/2}? simpler: Grp with A=Z_n, treat only (a,0) elements
        els=[G.idx[((a,),0)] for a in range(1,n)]
        # subgroup A x{0} is closed; restrict mt
        cnt=0;pst=[]
        for mask in range(1<<(n-1)):
            S=[els[i] for i in range(n-1) if mask>>i&1]
            Sset=set(S)
            if not any(G.inv[s] in Sset for s in S): continue
            cnt+=1
            # H on full G has two components; PST in component A: use targets within A
            t=pst_targets(G,S)
            if t: pst.append((sorted(G.els[s][0][0] for s in S),[G.els[z][0][0] for z in t]))
        print("Z_%d"%n,"zeta>0 sets",cnt,"with PST:",pst)
