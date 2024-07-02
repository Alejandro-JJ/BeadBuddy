"""
Analysis Examples
using the function from StressTensor_tools
"""
from StressTensor_tools import BeadSolver, Plotter_MapOnMap, Plotter_Maps2D, IntegrateTension
import os
import matplotlib.pyplot as plt
import numpy as np

#%% Comparison with COMSOL
i = 4
order = 5

plt.close('all')
parentfolder = '/media/alejandro/Coding/MyGits/Derivation_StressTensor/ExampleBeads_comsoled/'
folders = os.listdir(parentfolder)


folder=folders[i]
#for folder in folders:

SHpath = os.path.join(parentfolder, folder)+'/SH_Coefficients.txt'
COMpath = os.path.join(parentfolder, folder)+'/traction_phitheta.txt'

map_r_R, map_T_real = BeadSolver(SHpath, order=order, G_exp=1000, nu_exp=0.48, N_lats=50, N_lons=100)
Plotter_Maps2D([map_T_real], ['Radial tension - Analytical'], units=['Pa'], colorlist=['BrBG'])
Plotter_MapOnMap(map_r_R, map_T_real, color='BrBG')
mat = np.loadtxt(COMpath, delimiter=',')
mat = np.roll(mat.T, int(np.shape(mat)[0]/2), axis=1)

plt.figure()
plt.imshow(mat, cmap='BrBG')
plt.xlabel('phi')
plt.ylabel('theta')
plt.title('COMSOL solution')
plt.show()
plt.colorbar()


    
    
    
    
    