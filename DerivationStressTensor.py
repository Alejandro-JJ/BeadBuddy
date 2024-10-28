"""
All the necessary functions to generate analytical solutions to the elastic problem.
The final product are expressions of the radial stress tensor, only dependent on 
(theta, phi, nu, G) and the Spherical Harmonics coefficients c_{l,m} of a body
This derivation is directly based on Lea Krueger et al. - Biophysical Journal (2024)

The analytical solutions are stored in dill format to be loaded afterwards
Using the function "GenerateSolution(l_max)", a user can generate its own solutions 
up to an arbitrary order.
For simplicity, the default version of BeadBuddy includes solutions up to l_max=14
"""

import sympy as sp
from sympy.vector import CoordSys3D
from sympy.physics.quantum.cg import CG as ClebschGordan
from sympy import symbols, IndexedBase, Idx, Ynm
from time import time
import dill
from tqdm import tqdm

# Necessary symbols for sympy 
r, r0, theta, phi, l, m, pi, x, nu, G = symbols('r, r0, θ, φ, l, m, π, x, ν, G')
# Coordinate system
C = CoordSys3D('C') #unitary vectors are (C.i, C.j, C.k)
# Spherical unitary vectors
er = sp.sin(theta)*sp.cos(phi)*C.i + sp.sin(theta)*sp.sin(phi)*C.j + sp.cos(theta)*C.k 

D = IndexedBase('D')
nn,mm = symbols('nn mm', cls=Idx)

def N(n,m):
    '''
    Normalization constant for SHs
    '''
    if sp.sqrt(m**2)<=n:
        return sp.sqrt((2*n + 1)*sp.factorial(n - m)/(4*pi*sp.factorial(n + m))).subs(pi, sp.pi).doit()
    else:
        return 1

def Kx(n,m): 
    n, m = int(n), int(m)
    if sp.sqrt(m**2)<=n: 
        return sp.pi.evalf()*sp.sqrt(24/3)*( 
        (D[n,m+1]*ClebschGordan(n,m+1,1,-1,n,m)-D[n,m-1]*ClebschGordan(n,m-1,1,1,n,m))*ClebschGordan(n,0,1,0,n,0)+ 
        (D[n-1,m+1]*ClebschGordan(n-1,m+1,1,-1,n,m)-D[n-1,m-1]*ClebschGordan(n-1,m-1,1,1,n,m))*ClebschGordan(n-1,0,1,0,n,0)*sp.sqrt((2*n-1)/(2*n+1)) +
        (D[n+1,m+1]*ClebschGordan(n+1,m+1,1,-1,n,m)-D[n+1,m-1]*ClebschGordan(n+1,m-1,1,1,n,m))*ClebschGordan(n+1,0,1,0,n,0)*sp.sqrt((2*n+3)/(2*n+1))).doit()
    else:
        return 0
   
def Ky(n,m): 
    n, m = int(n), int(m)
    if sp.sqrt(m**2)<=n: 
        return 1j*sp.pi.evalf()*sp.sqrt(24/3)*( 
        (D[n,m+1]*ClebschGordan(n,m+1,1,-1,n,m)+D[n,m-1]*ClebschGordan(n,m-1,1,1,n,m))*ClebschGordan(n,0,1,0,n,0)+ 
        (D[n-1,m+1]*ClebschGordan(n-1,m+1,1,-1,n,m)+D[n-1,m-1]*ClebschGordan(n-1,m-1,1,1,n,m))*ClebschGordan(n-1,0,1,0,n, 0)*sp.sqrt((2*n-1)/(2*n+1)) +
        (D[n+1,m+1]*ClebschGordan(n+1,m+1,1,-1,n,m)+D[n+1,m-1]*ClebschGordan(n+1,m-1,1,1,n,m))*ClebschGordan(n+1,0,1,0,n,0)*sp.sqrt((2*n+3)/(2*n+1))).doit()
    else:
        return 0

def Kz(n,m): 
    n, m = int(n), int(m)
    if sp.sqrt(m**2)<=n: 
        return 4*sp.pi.evalf()*( 
        (D[n,m]*ClebschGordan(n,m,1,0,n,m)*ClebschGordan(n,0,1,0,n,0))+ 
        (D[n-1,m]*ClebschGordan(n-1,m,1,0,n,m)*ClebschGordan(n-1,0,1,0,n,0)*sp.sqrt((2*n-1)/(2*n+1)))+
        (D[n+1,m]*ClebschGordan(n+1,m,1,0,n,m)*ClebschGordan(n+1,0,1,0,n,0)*sp.sqrt((2*n+3)/(2*n+1)))).doit()
    else:
        return 0


def K(n,m):
    return Kx(n,m)*C.i + Ky(n,m)*C.j + Kz(n,m)*C.k
    # Only for testing. Correction 4pi to all K terms
#    return (Kx(n,m)*C.i + Ky(n,m)*C.j + Kz(n,m)*C.k)/(4*sp.pi)


def c(n,m):
    if sp.sqrt(m**2)<=(n-1) and n>=0:
        return 1/(2.*r0)*(
                2*N(n,m)/N(n-1,m) * Kz(n,m)*(n+m) + 
                N(n,m-1)/N(n-1,m) * (Kx(n,m-1) - 1j*Ky(n,m-1)) - 
                N(n,m+1)/N(n-1,m) * (Kx(n,m+1) + 1j*Ky(n,m+1))*(m+n)*(m+n+1)).evalf()
    else:
        return 0
        
def a(n,m): 
    return c(n+1,m)/(2*(-1+n*(4*nu-3)+2*nu))

def ComplexSH(n,m):
    if abs(m)>n:
        return 0
    else:
        return sp.simplify(Ynm(n,m,theta, phi).expand(func=True))

def Tinrr(n,m):
    '''
    Final expression for each component of the radial stress tensor for (n,m)
    '''
    Tinrr = 2*G*(r/r0)**n*((nu/(1-2*nu)*(-2-4*n+4*nu*(1+2*n))+2*n+n*(n-1)*(1-(r0/r)**2)) * a(n,m)*ComplexSH(n,m) + (K(n+1,m)&er)*(n+1)*ComplexSH(n+1,m)/r0)
    return Tinrr.subs(pi, sp.pi.evalf()).doit().simplify()


def GenerateSolution(lmax, savepath=None):
    '''
    Generate a master equation for an arbitrary order and save it in dill
    format for later use in BeadBuddy
    '''
    dill.settings['recurse']=True
    Trr = 0
    start = time()
    for n in tqdm(range(0,lmax+1)):
        for m in range(-n-1, n+2):
            Trr = Trr + Tinrr(n,m)
    Trr = Trr.evalf() # final simplification for sqrts, pi and factor
    print(f'Generation of solution for lmax={lmax} took: {int(time()-start)} seconds')
    if savepath==None:
        savepath = f'./GeneralSolution_lmax={str(lmax).zfill(2)}.txt'        
    with open(savepath, 'wb') as f:
        dill.dump(Trr,f)
    print('Solution saved \n')

#%% Single use before publication of BeadBuddy
'''
A better usage would be to generate each order individually and sum the necessary ones afterwards,
when solving. To avoid loading several dill files later, however, we opt for this solution,  
at least for the initialization of BeadBuddy.
Afterwards a user could add Tinrr for additional higher orders
'''
for lmax in range(1,11):
    savepath = f'./GeneralSolutions_4piK/GeneralSolution_lmax={str(lmax).zfill(2)}.txt'
    GenerateSolution(lmax, savepath=savepath)