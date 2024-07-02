"""
All the necessary functions to make use of the pre-comp[uted solutions for the stress tensor
generated in DerivationStressTensor.py

"""
import sympy as sp
import pyshtools as sh
from pyshtools.shio import SHrtoc
import numpy as np
from sympy import IndexedBase, lambdify
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import dill 
from matplotlib.ticker import MaxNLocator

dill.settings['recursive']=True
r, r0, theta, phi, l, m, pi, x, nu, G = sp.symbols('r, r0, θ, φ, l, m, π, x, ν, G')

# Two simple function to separate real and img part of arrays
take_real = np.vectorize(np.real)
take_imag = np.vectorize(np.imag)


def ComplexCoeffs(coeff_table):
    '''
    Converts a table of real Spherical Harmonics coefficients to 
    complex form. 
    The normalization of the spherical harmonics is maintained,
    as well as the Condon-Shortley convention 
    '''
    table_split_mpos = SHrtoc(coeff_table)
    table_mpos = table_split_mpos[0]+1j*table_split_mpos[1]
    rs, cs = np.shape(table_mpos)
    m_buffer = np.repeat(np.array(range(0,cs)), rs, axis=0).reshape(cs,rs).T
    table_mneg =  np.conj(table_mpos) *(-1)**(m_buffer)
    table = np.stack((table_mpos, table_mneg), axis=0)
    
    return table


def create_table(filepath, units='um'):
    '''
    Creates a table of Spherical Harmonics coefficients, both in real and complex
    form (for calculations and plots purposes respectively) from a saved file.
    It can read both .npy python structures and .txt files. If any other filetype is inputted 
    it will raise an error.
    Default units are 'um', but 'm' can be chosen. For the purpose of the stress tensor 
    calculation with Lea Krueger's theory, the normalization has been hardcoded to 'orthonormal'.
    Returns:
        lmax: highest order of expansion available
        coeff_table_real: real coeffs
        coeff_table_complex: complex coeffs
        initial_radius: supposing conservation of volume
    '''
    extension = filepath.split('.')[-1]
    if extension=='npy':
        coeffs = sh.SHCoeffs.from_file(filepath, format='npy', normalization='ortho', csphase=-1)
        coeff_table_real = coeffs.coeffs
    
    elif extension=='txt':
        with open (filepath, 'r') as f:
            data = [list(map(float,line.split(' '))) for line in f if ('Positive' not in line and 'Negative' not in line)]
            m_pos = np.array(data[0:int(len(data)/2)])
            m_neg = np.array(data[int(len(data)/2):])
            coeff_table_real = np.array([m_pos, m_neg])
            coeffs = sh.SHCoeffs.from_array(coeff_table_real, normalization='ortho', csphase=-1)
    else:
        raise Exception('Filetype must be ".npy" or ".txt"')
    
    lmax = np.shape(coeff_table_real)[1]-1
    # Original volume and d00
    d00_prime = float(coeff_table_real[0,0,0])
    volume = coeffs.convert(normalization='4pi').volume()
    initial_radius = np.cbrt(3*volume/(4*np.pi))
    # We need to truncate to avoid small contributions from delta-volume
    d00 = np.round(d00_prime-initial_radius*np.sqrt(4*np.pi), 12)
    # Substitute d00 for the [0,0,0] value in the complex table 
    coeff_table_real[0,0,0] = d00
    coeff_table_complex = ComplexCoeffs(coeff_table_real)

    # Convert to SI units
    if units =='m':
        coeff_table_real = coeff_table_real*1e-6
        coeff_table_complex = coeff_table_complex*1e-6
        initial_radius = initial_radius*1e-6
        
    return lmax, coeff_table_real, coeff_table_complex, initial_radius


def ParameterSubstitution(expr, coeff_table, initial_radius, G_exp, nu_exp, Order, first_n=0): 
    '''
    Substitute all necesary parameters in the MasterEquation to leave it only dependent
    on the spatial coordinates (r, theta, phi).
    The order ius dictated by the solution we loaded
    This creates a dictionary containing:
        * All SH coefficients from the loaded table
        * Initial radius, G and nu
    Returns the expression ready to evaluate in a custom coordinate grid
    '''
    D = IndexedBase('D')
#    start = time()
    maxn = np.shape(coeff_table)[1]
    substs = {}
    for n in tqdm(range(first_n, Order+3), colour='cyan'): #Order+2 will be necessary
        for m in range(-n, n+1):
            if n>=maxn: # if the coefficient is not in our table
                substs[D[n,m]] = 0
            elif m >=0:
                substs[D[n,m]] = coeff_table[0,n,m]
            else:
                substs[D[n,m]] = coeff_table[1,n,-m]
    # Other parameters
    substs[r0] = initial_radius
    substs[G] = G_exp
    substs[nu] = nu_exp
    expr = expr.subs(substs) # Simultaneous substitution of the whole dictionary
#    print(f'\nSubstitution of parameters took {int(time()-start)} seconds')
    return expr.evalf()

def Equation2Maps(sympy_expression, coeff_table_complex, initial_radius, N_lats=100, N_lons=200):
    '''
    This functions takes a sympy expression of the Stress Tensor (with the Spherical
    Harmonics coefficients and the physical parameters already substituted) and converts
    it to a set of 2D maps (radius, stress tensor) in latitude-longitude spherical projections.
    The input expression must only be dependent on {r, θ, φ}, i.e. the spatial coordinates
    
    
    When using the current convention, 
    * longitutes (phi) span [0...2pi]
    * latitudes (theta) span [0...pi]
    '''
#    start = time()
    # Define a custom size for the evaluation array
    lats_eval, lons_eval = np.meshgrid(np.linspace(0,np.pi,N_lats), np.linspace(0,2*np.pi,N_lons))
    coeffs = sh.SHCoeffs.from_array(coeff_table_complex, normalization='ortho',csphase=-1)
    map_deform = coeffs.expand(colat=lats_eval, lon=lons_eval, degrees=False)
    map_r = (map_deform + initial_radius)
    
    # There is always a small imaginary contribution from rounding errors
    map_r_R = take_real(map_r) 
    map_deform = take_real(map_deform)
    map_deform_norm = map_deform/initial_radius
    
    # Create a function from our expression 
    T_function = lambdify([r, theta, phi], sympy_expression)
    
    # Evaluate it on the grid
    T_complex = T_function(map_r_R, lats_eval, lons_eval) # this is the bottleneck
    map_T_real = take_real(T_complex)
#    map_T_imag = take_imag(T_complex)
    # All arrays must be transposed
#    print(f'Evaluation onto a spherical grid tool {int(time()-start)} seconds')
    return map_deform_norm.T, map_r_R.T, map_T_real.T#, map_T_imag


def Plotter_Maps2D(maps,titles=[], units=[], colorlist=None):
    """
    This function accepts a list with an arbitraty number of maps to be plotted
    and plots them in latitude-longitude fashion
    """
    #plt.style.use('dark_background')
    plt.style.use('default')
    fuente = {'fontsize':12, 'fontname':'Arial'}
    N_maps = len(maps)
    len_tt, len_pp = np.shape(maps[0])[0]-1, np.shape(maps[0])[1]-1
    fig, axs = plt.subplots(N_maps, 1, sharex=True, figsize=(6,6))
    
    if N_maps==1:
        axs=[axs]
    
    for i, amap in enumerate(maps):
        if colorlist==None:
            m = axs[i].imshow(amap, cmap='RdBu')#, vmin=-0.5, vmax=0.5)  # custom range 
            # The custom range can be used in conjunction with the PlotterMapOnMap
            # to get the same color limits
        else:
            m = axs[i].imshow(amap, cmap=colorlist[i])
        axs[i].set_xticks([0,int(len_pp/2),len_pp])
        axs[i].set_yticks([0,int(len_tt/2),len_tt])
        axs[i].set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
        axs[i].set_ylabel(r'$\theta$', **fuente)        
        cbar = plt.colorbar(m, ax=axs[i], fraction=0.024, pad=0.04)
        cbar.formatter.set_powerlimits((0, 0)) # scientific notation
        
        if len(units)==len(maps):
            cbar.ax.set_ylabel(units[i])
            
    if len(titles)==N_maps:
        for i,title in enumerate(titles):
            axs[i].set_title(titles[i],**fuente)
        
    axs[N_maps-1].set_xlabel(r'$\phi$')
    axs[N_maps-1].set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    plt.show()

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.
    
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    
    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])
    
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    

def Plotter_MapOnMap(map_r, map_toplot, title='', color=None, ax=None):
    """
    This function plots the projection of a color-coded map onto
    another 3D map. This is particularly useful to see a color-coded
    magnitude (stress, radius) projected onto the surface of a 3D bead.
    They can take an external axis as an argument, to use with BeadBuddy
    If there is no axis available, we create a new one
    """
    # Optional: forcing normalization of facecolor to match the 2D plots    
    #from matplotlib import colors
    #norm = colors.Normalize(-0.5, 0.5)
    
    
    fuente = {'fontsize':12, 'fontname':'Arial'}
    map_shape = np.shape(map_r) 
    # Coordinates
    pp, tt = np.meshgrid( 
    np.linspace(0, 2*np.pi, map_shape[1]),
    np.linspace(0, np.pi, map_shape[0]))   
    x = map_r * np.sin(tt) * np.cos(pp)
    y = map_r * np.sin(tt) * np.sin(pp)
    z = map_r * np.cos(tt)
    #Plot with custom map as facecolor
    plt.style.use('default')
    #plt.style.use('dark_background')
    if ax==None:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')
        plt.title(title,**fuente)
    ax.set_xlabel(r'$\mu m$', **fuente),ax.set_ylabel(r'$\mu m$',**fuente),ax.set_zlabel(r'$\mu m$',**fuente)
    if color==None:
        ax.plot_surface(x,y,z,facecolors=cm.RdBu((map_toplot-np.amin(map_toplot))/(np.amax(map_toplot)-np.amin(map_toplot))))
    elif color=='BrBG':
        ax.plot_surface(x,y,z,facecolors=cm.BrBG((map_toplot-np.amin(map_toplot))/(np.amax(map_toplot)-np.amin(map_toplot))))

    #ax.plot_surface(x,y,z,facecolors=cm.RdBu_r((map_toplot-np.amin(map_toplot))/(np.amax(map_toplot)-np.amin(map_toplot))))
    #ax.plot_surface(x,y,z,facecolors=cm.plasma(norm(map_toplot)))
    
    # Transparent panes:
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # Force integers in all axes
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.zaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_box_aspect((np.amax(x)-np.amin(x), np.amax(y)-np.amin(y), np.amax(z)-np.amin(z)))
    # Eliminate grid
    ax.grid(False)
    # set axes equal
    set_axes_equal(ax)
    plt.show()
 
    
def IntegrateTension(map_r, map_T_real):
    '''
    This function numerically integrates the radial tension over the surface
    of the bead, as a sum of the product pairs Tension*SurfaceDifferential
    '''
    N_theta, N_phi = np.shape(map_r)
    dtheta, dphi = np.pi/N_theta, 2*np.pi/N_phi
    pp, tt = np.meshgrid( 
            np.linspace(0, 2*np.pi, N_phi),
            np.linspace(0, np.pi, N_theta)) 
    dF = map_T_real * map_r**2 * np.sin(tt) * dphi * dtheta
    F = sum(map(sum,dF))
    return F

def GenerateCustomTable(customcoeffs, lmax=10, radius=10):
    '''
    Generates a table with some custom sample modes to study and debug
    test cases of simple beads. It accepts a list of tuplets (n,m,value)
    Each tuplet indicates the value of the coefficient (n,m)
    The coefficient (0,0) is hardcoded with the radius
    Example use:
        
        GenerateCustomTable([(2,0,0.5),(4,-3,-0.7)], './MyCustomTable.txt')
    '''
    Table = np.zeros((2, lmax+1, lmax+1))
    Table[0,0,0] = radius*np.sqrt(4*np.pi) # 10um bead
    for coeff in customcoeffs:
        n, m, value = coeff
        if m>=0:    
            Table[0,n,m] = value
        else:
            Table[-1,n,abs(m)] = value
    return Table

def BeadSolver(tablepath, order=5, G_exp=1, nu_exp=0.45, N_lats=50, N_lons=100):
    '''
    The main function to use to solve the stress tensor associated of a bead which
    has been expanded into a table os Spherical Harmonics.
    '''
    start = time()
    print(f'Solving stress with analytical solution of order {order}')
    
    # Extract the table of SH coefficients and some initialization parameters
    lmax, coeff_table_real, coeff_table_complex, initial_radius = create_table(tablepath, units='m')
    
    # Load the necessary Master Equation
    EquationPath = f'./GeneralSolutions/GeneralSolution_lmax={str(order).zfill(2)}.txt'
    MasterEquation = dill.load(open(EquationPath, 'rb'))
    
    # Substitute all the Sh coeffs and the physical parameters into the Master Equation
    sympy_expression = ParameterSubstitution(MasterEquation, coeff_table_complex, initial_radius, G_exp, nu_exp, order)
    
    # Evaluate the tension function onto a custom grid
    map_deform_norm, map_r_R, map_T_R = Equation2Maps(sympy_expression, coeff_table_complex, initial_radius, N_lats=N_lats, N_lons=N_lons)
    print(f'Solution took {int(time()-start)} seconds')
    print('='*50+'\n')
    return map_r_R, map_T_R
#    plt.figure(), plt.imshow(map_r_R)
    
    
    
#%% 
'''
Example use
'''
if __name__ == "__main__": # dont execute on import
    mytable = GenerateCustomTable([(2,0,-0.7), (4,-3,0.9)])
    np.save('tablita.npy', mytable)
    plt.close('all')
    tablepath = '/media/alejandro/Coding/MyGits/BEADBUDDY/tablita.npy'
    map_r_R, map_T_R = BeadSolver(tablepath, order=5, N_lats=50, N_lons=100, G_exp=2100)
    Plotter_Maps2D([map_r_R, map_T_R], titles=['Radius', 'Tension'], units=['um', 'Pa'], colorlist=['RdBu', 'BrBG'])
    Plotter_MapOnMap(map_r_R, map_T_R, title='Radial tension', color='BrBG')
    Force = IntegrateTension(map_r_R,map_T_R)
