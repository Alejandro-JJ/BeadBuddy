"""
Function pipeline written initially for L.M.Scharfenstein
Loads:
- labelled image from BeadBuddy
- SH_analysis folder from BeadBuddy
Returns:
- 2D plot of labelled image in max.projection
- 2D heatmap of main deformation (c20/c00)

"""
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorcet as cc
from tqdm import tqdm
import os
from time import perf_counter

# Glasbey colormap with black background
glasbey = cc.glasbey
colors = [(0, 0, 0, 1)] + [mcolors.to_rgba(c) for c in glasbey]
glasbey_dark = mcolors.ListedColormap(colors, name="glasbey_dark")


def crono(func):
    # performance time decorator
    def wrapper(*args, **kwargs):
        t_b = perf_counter()
        func_result = func(*args, **kwargs)
        t_e = perf_counter()
        t_ex = t_e-t_b
        t, units = (np.round(t_ex,1),'seconds') if t_ex<60 else (np.round(t_ex/60,1), 'minutes')      
        print(f'Function {func.__name__} took {t} {units}')
        return func_result
    return wrapper


def plotMaxProj(im):
    max_proj = np.max(im, axis=0)
    fig,ax = plt.subplots(1,1,figsize=(4,4))
    ax.imshow(max_proj, cmap=glasbey_dark, interpolation='nearest')
    plt.tight_layout()
    
def getPixelCoods(im, px=1):
    '''
    Returns (x,y) and (x_com, y_com) of a label
    Ignores depth (axis=0)
    '''
    x = np.where(im==px)[2]
    y = np.where(im==px)[1]
    x_com, y_com = np.mean(x), np.mean(y)
    return (x,y), (x_com, y_com)

def getComDict(im, pxs):
    com = dict()
    print('Calculating centers of mass\n')
    for px in tqdm(pxs):
        beadCoords, beadCom = getPixelCoods(im, px=px)
        com[px] = beadCom
    return com

def getDeformDict(SH_path, max_deform=10):
    pxs = list(sorted([int(f.replace('.', '_').split('_')[3]) for f in os.listdir(SH_path) if f.startswith('SH')]))
    deform = dict()
    print('Calculating deformations\n')
    for px in tqdm(pxs):
        path = SH_path + f'SH_Array_Bead_{str(int(px)).zfill(4)}.npy'
        table = np.load(path)
        deformation = abs(table[0,2,0])/table[0,0,0]
        if 0<deformation<max_deform:
            deform[px] = deformation
        pxs_valid = deform.keys()
    return deform, pxs_valid
        
def plotImAndHeatmap(im, com_dict, deform_dict, bins=200, cmap='cividis', title='', max_deform=10):
    xy = np.array([com_dict[k] for k in com_dict])
    d  = np.array([deform_dict[k] for k in com_dict])
    heat, xedges, yedges = np.histogram2d(
        xy[:,0], xy[:,1], bins=bins, weights=d)
    
    fig,axs = plt.subplots(1,3,figsize=(9,6), gridspec_kw={'width_ratios':[1,1,0.1]}) # for cbar

    # Max Proj
    max_proj = np.max(im, axis=0)
    axs[0].imshow(max_proj, cmap=glasbey_dark, interpolation='nearest')
    axs[0].set_aspect("equal", adjustable="box")  # preserve square coords
    # Heatmap
    m = axs[1].imshow(heat.T, interpolation="nearest", 
           aspect="auto", cmap=cmap, vmin=0, vmax=max_deform)
    axs[1].set_aspect("equal", adjustable="box")  # preserve square coords
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    
    plt.colorbar(m, label="deform", cax=axs[2])
    
    plt.tight_layout() # before formatting cbar
    pos = axs[2].get_position()
    axs[2].set_position([
        pos.x0,
        pos.y0 + 0.25 * pos.height,   # move up
        pos.width*0.5,
        0.5 * pos.height])
    plt.suptitle(title)
    return fig

@time_min
def CompleteAnalysis(bins=20, cmap='cividis', save=True):
    """
    Takes user input on necesary paths
    Import image and SH_tables
    Plots segmentation + deformation heatmap
    Returns com->(x,y) and deform->c20/c00 dictionaries
    """
    IM_PATH = input('Enter path of segmented image:\n')
    SH_PATH = input('Enter path of analysis folder:\n')
    if SH_PATH[-1]!='/': # correct missing slash
        SH_PATH+='/'
    im = io.imread(IM_PATH)
    deform, pxs_valid = getDeformDict(SH_PATH) # first we check for available data
    com = getComDict(im, pxs_valid) 
    dataset_name = SH_PATH.split('/')[-2].strip('SH_Analysis_')
    fig = plotImAndHeatmap(im, com, deform, bins=20, cmap='cividis', title=dataset_name)
    
    if save==True:
        savefolder = os.path.dirname(os.path.dirname(SH_PATH))
        np.save(savefolder+f'/{dataset_name}_CoM.npy', com)
        np.save(savefolder+f'/{dataset_name}_Deform.npy', deform)
        # save pic automatically!!!
        fig.savefig(savefolder+f'/{dataset_name}.png')
    return com, deform


def PostAnalysis(cmap='viridis',bins=50, title='', max_deform=10):
    """
    Once deform.npy and com.npy have been created, 
    this functions allows for fast plotting and 
    customization.
    It needs: image + deformation.npy + com.npy
    """
    IM_PATH = input('Enter path of segmented image:\n')
    DEF_PATH = input('Enter path of deformation file:\n')
    COM_PATH = input('Enter path of CoM file:\n')
    print('Loading and plotting...\n')
    im = io.imread(IM_PATH)
    com = np.load(COM_PATH, allow_pickle=True).item()
    deform = np.load(DEF_PATH, allow_pickle=True).item()
    _ = plotImAndHeatmap(im, com, deform, bins=bins, cmap=cmap, title=title,max_deform=max_deform)
    

    

    
# Single call for Lisa
com, deform = CompleteAnalysis()

#%%
'''
Once analysis is done and files are saved, 
you can plot faster with PostAnalysis()
'''
# Example:
# PostAnalysis(cmap='summer', bins=7, title='New color and deform. limit!', max_deform=5)

