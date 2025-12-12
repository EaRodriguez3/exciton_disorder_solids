#%%

# This script is used to generate plots for the MCMC simulations of disordered solids.
# There is no MCMC simulation code here, we will use the results from the simulations and just plot them.

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib import rcParams, font_manager
# Load the MCMC simulation results from .dat files inside the data\simulations folder
font = font_manager.FontProperties(family='Arial',
                                   style='normal', size=40)

rcParams['font.family'] = font.get_name()
rcParams['font.size'] = 20
rcParams['font.style'] = 'normal'

folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data\\simulations/')

def plot_crystal_structures():
    # load the data

    datas = glob.glob(folder + 'crystal_structure_*.dat')
    # it is large, medium, small
    datas.sort()
    sizes = ['C8', 'C12', 'C16'] # sorted order
    colors = ['r', 'g', 'b']
    for k, data in enumerate(datas):
        data0 = np.loadtxt(data)
        # there are three axis:
        # 0 is x position of cyrcles
        # 1 is y position of circles
        # 2 is radius of circles
        fig, ax = plt.subplots(figsize=(6,6))
        for i in range(data0.shape[0]):
            # almost transparent circles, but edge is visible
            circle = plt.Circle((data0[i,0], data0[i,1]), data0[i,2], facecolor='white', 
                                alpha=1, edgecolor=colors[k], linewidth=2,
                                )
            ax.add_artist(circle)
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)


        ax.set_aspect('equal', adjustable='box')
        plt.title(f'{sizes[k]}')
        plt.xlabel('X (nm)')
        plt.ylabel('Y (nm)')
        plt.xticks([-75, 0, 75])
        plt.yticks([-75, 0, 75])
        #plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'crystal_structure_{sizes[k]}.svg', dpi=300)

def plot_variance_sizes_results():

    # load the data, there are six files, three sizes, for energetic and structural disorder

    datas_strcut_disorder = glob.glob(folder + 'variance_vs_time_st_*.dat')
    datas_energetic_disorder = glob.glob(folder + 'variance_vs_time_en_*.dat')
    datas_strcut_disorder.sort() # sort them to be in order of sizes
    datas_energetic_disorder.sort() # sort them to be in order of sizes
    sizes = ['C8', 'C12', 'C16'] # sorted order
    colors = ['r', 'g', 'b']

    fig1, ax1 = plt.subplots(figsize=(7,6)) # variance vs time
    fig2, ax2 = plt.subplots(figsize=(7,6)) # size vs time
    for i, data_st, data_en in zip(range(3), datas_strcut_disorder, datas_energetic_disorder):
        data_st = np.loadtxt(data_st)
        data_en = np.loadtxt(data_en)


        if i == 0:
            # plot with labels
            # var
            ax1.plot(data_st[:,0]*0.002, (data_st[:,1] - data_st[:,1][0])/1e6, color=colors[i], linestyle='--', label=f'Structural Disorder')
            ax1.plot(data_en[:,0]*0.002, (data_en[:,1] - data_en[:,1][0])/1e6, color=colors[i], linestyle='-', label=f'Energetic Disorder')
            # size
            ax2.plot(data_st[:,0]*0.002, data_st[:,2] , color=colors[i], linestyle='--', label=f'Structural Disorder')
            ax2.plot(data_en[:,0]*0.002, data_en[:,2] , color=colors[i], linestyle='-', label=f'Energetic Disorder')
        else:
            # plot without labels
            # var
            ax1.plot(data_st[:,0]*0.002, (data_st[:,1] - data_st[:,1][0])/1e6, color=colors[i], linestyle='--', )
            ax1.plot(data_en[:,0]*0.002, (data_en[:,1] - data_en[:,1][0])/1e6, color=colors[i], linestyle='-', )
            # size
            ax2.plot(data_st[:,0]*0.002, data_st[:,2] , color=colors[i], linestyle='--', )
            ax2.plot(data_en[:,0]*0.002, data_en[:,2] , color=colors[i], linestyle='-', )
        
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('MSD ($\mu$m$^2$)')
    # scientific notation for y axis
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0),)
    ax1.legend( fontsize = 14, frameon=False)
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 0.0032)
    ax1.set_xticks([0,2,4])
    ax1.set_yticks([0,0.001,0.002,0.003])

    fig1.savefig('variance_vs_time_disorder_types.svg', dpi=300, bbox_inches='tight')

    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Average crystal Size (nm)')
    ax2.legend( fontsize =14, frameon=False)
    ax2.set_xlim(0, 4)
    ax2.set_ylim(2, 14)
    ax2.set_xticks([0,2,4])
    ax2.set_yticks([4,8,12])

    fig2.savefig('size_vs_time_disorder_types.svg', dpi=300, bbox_inches='tight')

def main():
    plot_crystal_structures()
    plot_variance_sizes_results()   

#%%
if __name__ == '__main__':
    main()
# %%
