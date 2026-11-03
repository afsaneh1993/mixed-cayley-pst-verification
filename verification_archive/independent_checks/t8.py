from lib import *
from scipy.linalg import expm
from collections import Counter
G=Grp((2,8),lambda a:(a[0],(3*a[1])%8),(0,0))
e=lambda z,r,s:G.idx[((z%2,r%8),s)]
# element z^a r^b s^c = (a,b),c  (x^c * a-part): s r = ? our (a,r) means x^r a; s*r^k = (f(k),1)?? use (k,1) = s r^k
S=[e(0,1,0),e(0,7,0),e(1,0,0),e(1,2,0),e(1,6,0),e(0,0,1),e(0,2,1),e(1,0,1),e(1,4,1)]
# note (k,1)= s r^k ; paper writes r^2 s = s r^{?}: r^2 s = s f(r^2)... compute
def mul(a,b): return G.mt[a,b]
r2=e(0,2,0); s=e(0,0,1); zr4s=mul(e(1,4,0),s); r2s=mul(r2,s); zs=mul(e(1,0,0),s)
S=[e(0,1,0),e(0,7,0),e(1,0,0),e(1,2,0),e(1,6,0),s,r2s,zs,zr4s]
Sset=set(S); print("size",len(Sset),"inverse-closed",all(G.inv[x] in Sset for x in S))
c=e(1,4,0); print("c central",c in G.center())
H=G.H(S); print(sorted(Counter(np.round(np.linalg.eigvalsh(H),5)).items()))
Ho=G.H(S+[c]); print("H0",sorted(Counter(np.round(np.linalg.eigvalsh(Ho),5)).items()))
print("U(pi/2)=i R_c?",np.allclose(expm(-1j*np.pi/2*H),1j*G.Rmat(c)) or np.allclose(expm(-1j*np.pi/2*H),-1j*G.Rmat(c)) or np.allclose(expm(-1j*np.pi/2*H),G.Rmat(c)) or np.allclose(expm(-1j*np.pi/2*H),-G.Rmat(c)))
print("Aut",digraph_aut_order(H))
