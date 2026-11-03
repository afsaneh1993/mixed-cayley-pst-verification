from lib import *
import numpy as np, itertools, sys, time
from collections import Counter

def groups():
    d={}
    d['S3']=((3,),lambda a:((-a[0])%3,),(0,))
    d['D4']=((4,),lambda a:((-a[0])%4,),(0,))
    d['Q8']=((4,),lambda a:((-a[0])%4,),(2,))
    d['D5']=((5,),lambda a:((-a[0])%5,),(0,))
    d['D6']=((6,),lambda a:((-a[0])%6,),(0,))
    d['Dic3']=((6,),lambda a:((-a[0])%6,),(3,))
    d['D7']=((7,),lambda a:((-a[0])%7,),(0,))
    d['D8']=((8,),lambda a:((-a[0])%8,),(0,))
    d['Q16']=((8,),lambda a:((-a[0])%8,),(4,))
    d['SD16']=((8,),lambda a:((3*a[0])%8,),(0,))
    d['M16']=((8,),lambda a:((5*a[0])%8,),(0,))
    d['Z4sdZ4']=((4,2),lambda a:((-a[0])%4,a[1]),(0,1))
    d['Z2xD4']=((4,2),lambda a:((-a[0])%4,a[1]),(0,0))
    d['Z2xQ8']=((4,2),lambda a:((-a[0])%4,a[1]),(2,0))
    d['Z4oD4']=((4,2),lambda a:((a[0]+2*a[1])%4,a[1]),(0,0))
    d['Z4Z2sdZ2']=((4,2),lambda a:(a[0],(a[1]+a[0])%2),(0,0))
    return d

def analyse(name):
    mods,f,y=groups()[name]
    G=Grp(mods,f,y); n=G.n
    Aidx=set(G.idx[(a,0)] for a in G.Aels)
    center=set(G.center())
    # order of elements
    def order(g):
        k=1;h=g
        while h!=G.one: h=G.mt[h,g];k+=1
        return k
    ords=[order(g) for g in range(n)]
    nonid=[s for s in range(n) if s!=G.one]
    # conjugation table: g^-1 s g
    conj=np.array([[G.mt[G.mt[G.inv[g],s],g] for s in range(n)] for g in range(n)])
    tot=0;pst=0;nonnorm=0;ord2=ord4=noncentral=othercoset=0;conn=0;connnn=0
    pst_normal=0
    pstsets=[]
    for mask in range(1<<(n-1)):
        S=[nonid[i] for i in range(n-1) if mask>>i&1]
        Sset=set(S)
        if not any(G.inv[s] in Sset for s in S): continue
        tot+=1
        H=G.H(S)
        w,V=np.linalg.eigh(H)
        if np.max(np.abs(w-np.round(w)))>1e-6: continue
        wi=np.round(w).astype(int)
        zeta=int(round(w.max())) if False else sum(1 for s in S if G.inv[s] in Sset)
        # spectral projections
        ths=sorted(set(wi))
        E={th:V[:,wi==th]@V[:,wi==th].conj().T for th in ths}
        v1={th:E[th][:,G.one] for th in ths}
        supp=[th for th in ths if np.linalg.norm(v1[th])>1e-8]
        targets=[]
        for z in range(n):
            if z==G.one: continue
            ok=True; mu={}
            for th in ths:
                vz=E[th][:,z]; a=v1[th]
                na=np.linalg.norm(a)
                if na<1e-8:
                    if np.linalg.norm(vz)>1e-8: ok=False;break
                    continue
                m=np.vdot(a,vz)/np.vdot(a,a)
                if abs(abs(m)-1)>1e-7 or np.linalg.norm(vz-m*a)>1e-7: ok=False;break
                mu[th]=m
            if not ok: continue
            # need t with exp(-i t (th-zeta)) = mu[th]/mu[zeta]  (mu[zeta]=1)
            others=[th for th in supp if th!=zeta]
            if not others: continue
            th1=others[0]; d=th1-zeta
            found=False
            for m_ in range(abs(d)):
                t=(-np.angle(mu[th1]) + 2*np.pi*m_)/d
                if all(abs(np.exp(-1j*t*(th-zeta))-mu[th])<1e-6 for th in supp):
                    found=True;break
            if found: targets.append(z)
        if not targets: continue
        pst+=1
        normal=all(set(conj[g,s] for s in S)==Sset for g in range(n))
        isconn=len(gen_subgroup(G,S))==n
        if isconn: conn+=1
        if normal: pst_normal+=1
        else:
            nonnorm+=1
            if isconn: connnn+=1
            for z in targets:
                if ords[z]==2: ord2+=1
                elif ords[z]==4: ord4+=1
                else: print("UNEXPECTED order",ords[z])
                if z not in center: noncentral+=1
                if z not in Aidx: othercoset+=1
        pstsets.append((S,targets,normal,isconn))
        # normal sets: targets must be central
        if normal:
            for z in targets:
                assert z in center
        # order check for all
        for z in targets: assert ords[z] in (2,4)
    return dict(sets=tot,pst=pst,nonnormal=nonnorm,ord2=ord2,ord4=ord4,noncentral=noncentral,othercoset=othercoset,conn=conn,conn_nn=connnn,pst_normal=pst_normal)

if __name__=="__main__":
    for name in sys.argv[1:]:
        t0=time.time()
        print(name,analyse(name),"%.0fs"%(time.time()-t0),flush=True)
