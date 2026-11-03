from lib import *
import numpy as np, itertools, random
from scipy.linalg import expm
random.seed(1); np.random.seed(1)

def chars(mods):
    A=list(itertools.product(*[range(m) for m in mods]))
    # character psi_j(d)=exp(2 pi i sum j_i d_i/m_i)
    return A,[ (lambda d,j=j: np.exp(2j*np.pi*sum(ji*di/m for ji,di,m in zip(j,d,mods)))) for j in A], A

def test(mods,f,y,trials=20):
    G=Grp(mods,f,y)
    A,psis,js=chars(mods)
    nA=len(A)
    worst=0
    for _ in range(trials):
        S=[s for s in range(G.n) if s!=G.one and random.random()<0.35]
        if not S: continue
        H=G.H(S)
        # h_r(d)=H_{1,(d,r)}
        h=lambda d,r: H[G.one,G.idx[(d,r)]]
        # block matrices
        ev_blocks=[]
        Mblocks=[]
        for j,psi in zip(js,psis):
            alpha=sum(h(d,0)*np.conj(psi(d)) for d in A)
            beta=sum(h(d,1)*np.conj(psi(d)) for d in A)
            psif=lambda d: psi(f(d))
            alphaf=sum(h(d,0)*np.conj(psif(d)) for d in A)
            M=np.array([[alpha, psi(y)*np.conj(beta)],[np.conj(psi(y))*beta, alphaf]])
            Mblocks.append((M,psi,alpha,beta))
            ev_blocks+=list(np.linalg.eigvalsh((M+M.conj().T)/2))
        ev=np.sort(np.linalg.eigvalsh(H)); evb=np.sort(ev_blocks)
        worst=max(worst,np.max(np.abs(ev-evb)))
        # U entries
        t=random.random()*3
        U=expm(-1j*t*H)
        for a in A[:6]:
            ent0=sum(np.conj(psi(a))*expm(-1j*t*M)[0,0] for (M,psi,_,_) in Mblocks)/nA
            ent1=sum(np.conj(psi(f(a)))*expm(-1j*t*M)[1,0] for (M,psi,_,_) in Mblocks)/nA
            worst=max(worst,abs(ent0-U[G.idx[(a,0)],G.one]),abs(ent1-U[G.idx[(a,1)],G.one]))
    return worst

tests={
 "D6": ((6,),lambda a:((-a[0])%6,),(0,)),
 "Dic3": ((6,),lambda a:((-a[0])%6,),(3,)),
 "SD16": ((8,),lambda a:((3*a[0])%8,),(0,)),
 "M16": ((8,),lambda a:((5*a[0])%8,),(0,)),
 "Z4^2:Z2": ((4,4),lambda a:(a[1],a[0]),(0,0)),
 "U12": ((2,3),lambda a:(a[0],(-a[1])%3),(1,0)),
 "U18": ((3,3),lambda a:(a[0],(-a[1])%3),(1,0)),
 "Dic(Z4xZ2,y)": ((4,2),lambda a:((-a[0])%4,(-a[1])%2),(2,0)),
}
for k,(mods,f,y) in tests.items():
    print(k, "max deviation:", test(mods,f,y))
