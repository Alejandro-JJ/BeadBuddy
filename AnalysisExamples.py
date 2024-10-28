"""
Analysis Examples
using the function from StressTensor_tools
"""
from StressTensor_tools import BeadSolver, Plotter_MapOnMap, Plotter_Maps2D, IntegrateTension
import os
import matplotlib.pyplot as plt
import numpy as np
import tqdm

parentfolder = '/media/alejandro/Coding/MyGits/Derivation_StressTensor/ExampleBeads_comsoled_smoothed'

folders = os.listdir(parentfolder)
for folder in folders:
    print(folder)

##%% 
#'''
#Compare all solutions of orders 1-9 for all beads
#including correction of 4*pi for the whole stress tensor
#'''
#import pickle
#i = 1
#ratios= []
#
#for folder in folders: # for each bead
#    ratio_bead = []
#    for order in tqdm(range(1,10)): # for each order
#        SHpath = os.path.join(parentfolder, folder)+'/SH_Coefficients.txt'
#        COMpath = os.path.join(parentfolder, folder)+'/traction_phitheta.txt'
#        
#        # Params
#        E = 1000
#        nu_exp= 0.45
#        G_exp = E/(2*(1+nu_exp))
#        
#        map_r_R, map_T_real = BeadSolver(SHpath, order=order, G_exp=G_exp, nu_exp=nu_exp, N_lats=100, N_lons=200)
#        
#        # 4pi correction
##        map_T_real = map_T_real/(4*np.pi)
#        
#        # COMSOL solution
#        mat = np.loadtxt(COMpath, delimiter=',')
#        mat = np.roll(mat.T, int(np.shape(mat)[0]/2), axis=1)
#        
#        # Integrate force
#        F_p = IntegrateTension(map_r_R, map_T_real)
#        F_c = IntegrateTension(map_r_R, mat)
#        ratio_bead.append(F_p/F_c)
#    ratios.append(ratio_bead)
#print('DONE!')
#savepath = '/media/alejandro/Coding/MyGits/Derivation_StressTensor/ExampleBeads_comsoled_smoothed/ratios_nocorrection.npy'
#with open(savepath, "wb") as f:   #Pickling
#   pickle.dump(ratios, f)

#%%
'''
Compare data from all beads and all orders for the different corrections
Previously, forces were calculated integrating the radial stress tensor over the 2D
spherical projections. Krueger et al. is definitely missing a global 4pi factor
'''
import pickle
path_4piK = '/media/alejandro/Coding/MyGits/Derivation_StressTensor/ExampleBeads_comsoled_smoothed/ratios_K_4pi.npy'
path_4pi = '/media/alejandro/Coding/MyGits/Derivation_StressTensor/ExampleBeads_comsoled_smoothed/ratios_total4pi.npy'
path_nocorr = '/media/alejandro/Coding/MyGits/Derivation_StressTensor/ExampleBeads_comsoled_smoothed/ratios_nocorrection.npy'
orders = np.arange(1,10,1)
plt.close('all')
with open(path_4piK, "rb") as f:   # Unpickling
    ratios_4piK = pickle.load(f)
    
with open(path_4pi, "rb") as f:   # Unpickling
    ratios_4pi = pickle.load(f)

# clean:
del ratios_4pi[2]

avg_ratio = [np.nanmean(x) for x in zip(*ratios_4pi)] 
#with open(path_nocorr, "rb") as f:   # Unpickling
#    ratios_nocorr = pickle.load(f)
#    
fig = plt.figure(figsize=(8,6))
ax = fig.subplots(1,1)

ignore = [2] # Gabriela's bead has extreme values
for i in range(0,len(ratios_4pi)):
    if i==0:
#        ax.plot(orders, ratios_4piK[i], 'r.--', label='4pi K correction')
        ax.plot(orders, ratios_4pi[i], 'b.--', label='4pi global correction')
#        ax.plot(ratios_nocorr[i], 'g.--', label='no correction')
    elif i in ignore:
        pass
    else:
#        ax.plot(orders, ratios_4piK[i], 'r.--')
        ax.plot(orders, ratios_4pi[i], 'b.--')
#        ax.plot(ratios_nocorr[i], 'g.--')
ax.plot(orders, avg_ratio, 'r.--', label='Average')
plt.legend()
ax.set_title(r'$\frac{F_{analytical}}{F_{COMSOL}}$', fontsize=20)
#ax.hlines(1,0,9, color='k', linewidth=3)
ax.set_xlim([1,9])
ax.set_xlabel('Solution SH order')


#%% Transform beads from 20201116_tp_59 to .txt files in the old .txt format for Jonnas
import os
import numpy as np
path = '/media/alejandro/PAPER_2024/Figure_5_Examples/PAA_analysis_20201116_tp59/Segmentations20_100_2_1/SH_Analysis_Beads_tp59_SH_11/'
savepath = '/media/alejandro/Coding/MyGits/Derivation_StressTensor/ExampleBeads_comsoled_smoothed/'
tables = sorted([f for f in os.listdir(path) if f.endswith('.npy')])
t= tables[0]
for i,t in enumerate(tables):
    table = np.load(path+t)
    savename = savepath+f'20201116_tp59_Bead_{i}.txt'
    TableTxt = open(savename, 'w')
    TableTxt.write('# Positive m\n')
    np.savetxt(TableTxt, table[0], fmt='%.4f')
    TableTxt.write('# Negative m\n')
    np.savetxt(TableTxt, table[1], fmt='%.4f')        
    TableTxt.close()
    
#%% Make a 2D plot comparing both maps for order 5?
parentfolder = '/media/alejandro/Coding/MyGits/Derivation_StressTensor/ExampleBeads_comsoled_smoothed/'
folders = list(sorted([f for f in os.listdir(parentfolder) if os.path.isdir(parentfolder+f)]))[12:]
# avoided the Beads which are still not calulated

# i=3, i=9, order=6 pretty good
i = 3


folder = folders[i]
SHpath = os.path.join(parentfolder, folder)+'/SH_Coefficients.txt'
COMpath = os.path.join(parentfolder, folder)+'/traction_phitheta.txt'

# Params
E = 1000
nu_exp= 0.45
G_exp = E/(2*(1+nu_exp))
order = 5

map_r_R, map_T_real = BeadSolver(SHpath, order=order, G_exp=G_exp, nu_exp=nu_exp, N_lats=100, N_lons=200)

# 4pi global correction
map_T_real = map_T_real/(4*np.pi)

# COMSOL solution
mat = np.loadtxt(COMpath, delimiter=',')
mat = np.roll(mat.T, int(np.shape(mat)[0]/2), axis=1)

#%% Plot for paper
# I checked that the best roll=0
plt.close('all')
fig = plt.figure(figsize=(6,8))
ax = fig.subplots(3,1,sharex=True)
fs = 15

m1 = ax[0].imshow(map_T_real, cmap = 'BrBG')
cbar = plt.colorbar(m1, ax=ax[0], fraction=0.024, pad=0.04)
ax[0].set_yticks([0,50,100])
ax[0].set_yticklabels(['0',r'$\pi/2$', r'$\pi$' ], fontsize=fs)
ax[0].set_ylabel('Analytical', fontsize=fs)

m2 = ax[1].imshow(mat, cmap = 'BrBG')
cbar = plt.colorbar(m2, ax=ax[1], fraction=0.024, pad=0.04)
ax[1].set_yticks([0,50,100])
ax[1].set_yticklabels(['0',r'$\pi/2$', r'$\pi$' ], fontsize=fs)
ax[1].set_ylabel('COMSOL', fontsize=fs)

m3 = ax[2].imshow(abs(map_T_real-mat), cmap = 'gray')
cbar = plt.colorbar(m3, ax=ax[2], fraction=0.024, pad=0.04)
ax[2].set_ylabel('Abs. difference', fontsize=fs)
ax[2].set_yticks([0,50,100])
ax[2].set_yticklabels(['0',r'$\pi/2$', r'$\pi$' ], fontsize=fs)
ax[2].set_xticks([0,100,200])
ax[2].set_xticklabels(['0', r'$\pi$', r'$2\pi$'], fontsize=fs)
ax[2].set_xlabel(r'$\Phi$', fontsize = fs)


