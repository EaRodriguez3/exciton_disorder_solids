#%%

# This script is used to generate plots for the MCMC simulations of disordered solids.
# There is no MCMC simulation code here, we will use the results from the simulations and just plot them.

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib import rcParams, font_manager
# Load the MCMC simulation results from .dat files inside the data\simulations folder

rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 30
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
            circle = plt.Circle((data0[i,0], data0[i,1]), 
                                data0[i,2], facecolor='white', 
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


# this one does the same as before but adds color based on the energy of the crystals
def crystal_plots_sims():
    # Load the data from the dat files
    fol = 'data\simulations/'
    data_large = np.loadtxt(fol + 'crystal_structure_large.dat')
    data_medium = np.loadtxt(fol + 'crystal_structure_medium.dat')
    data_small = np.loadtxt(fol + 'crystal_structure_small.dat')

    fig, axs = plt.subplots(3, 1, figsize=(2.3, 6))
    # lest try plotting using colormap gist_stern (but cut a bit).
    # the same values should have the same color across all three plots, so we need to find the global min and max of r across all three datasets.
    r_global_min = np.min([np.min(data[:, 2]) for data in [data_large, data_medium, data_small]])
    r_global_max = np.max([np.max(data[:, 2]) for data in [data_large, data_medium, data_small]])

    e_global_min = 0.024*(29.2725*np.exp(-0.3446*2*r_global_max) + 91.5)
    e_global_max = 0.024*(29.2725*np.exp(-0.3446*2*r_global_min) + 91.5)

    # normalize the radius values to be between 0 and 1 for the colormap
    # all plots should same the same color scheme, so same color in all three plots
    # should correspond to the same radius value. we can achieve this by normalizing the radius values to be between 0 and 1 based on the global min and max radius across all three datasets.

    it = 0
    for ax, data, label in zip(axs, [data_small, data_medium, data_large], ['C16', 'C12', 'C8']):
        x = data[:, 0]
        y = data[:, 1]
        r = data[:, 2]
        
        cmap = plt.get_cmap('RdYlBu')  # reversed so that larger radius is redder, smaller is bluer
        # Create a custom colormap that maps 2.1-2.4 to the full colormap, and clamps above 2.4 to blue
        energy = 0.024*(29.2725*np.exp(-0.3446*2*r) + 91.5) # goes from 2.1 to 2.8
        energy_min = 2.2
        energy_max = 2.35
        # I want to make all the values below 2.1 to be the same color as 2.1,
        #  and all the values above 2.4 to be the same color as 2.4,
        #  and the values in between to be mapped to the colormap.
        color_values = np.clip(energy, energy_min, energy_max)
        #color_values = (color_values - energy_min) / (energy_max - energy_min)

        for i in range(len(r)):
            circle = plt.Circle((x[i], y[i]), r[i], 
                        color=plt.cm.ScalarMappable(norm=plt.Normalize(vmin=energy_min, 
                                                    vmax=energy_max),
                                        cmap=cmap).to_rgba(color_values[i]), 
                        alpha=1)
            ax.add_artist(circle)

        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)

        if it == 0:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        # add a colorbar to the right of the last plot. 
        # It maps the energy values associated with the radius

        '''        
        if it == 2:

            norm = plt.Normalize(vmin=energy_min, vmax=energy_max)
            cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax, )
            
            # cabr ticks: 5, 10, 15
            cbar.set_ticks([2.2, 2.25, 2.3, 2.35])
            cbar.set_label('Energy (eV)')


        ax.set_aspect('equal', adjustable='datalim')
        #ax.set_title(label)
        '''

        it += 1

    # add colorbar 

    norm = plt.Normalize(vmin=energy_min, vmax=energy_max)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axs, location='right', pad=0.1, aspect=30)
    cbar.set_ticks([2.2, 2.25, 2.3, 2.35])
    cbar.set_label('Energy (eV)', )

    plt.tight_layout()
    plt.savefig('crystal_structure_simulation_redyelblue.svg', dpi=300)

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


def plot_energetic_spatial_disorder_dynamics():
    fol = 'data\simulations/variances_fine_adjustment/'
    files = glob.glob(fol + '*.dat')
    files = [files[1], files[0], files[2]] # reorder so that C16, C12, C8
    colors = ['b', 'g', 'r']
    labels = ['C16', 'C12', 'C8']

    fig_var, ax_var = plt.subplots(figsize=(5,6))
    fig_deltaE, ax_deltaE = plt.subplots(figsize=(5,6))

    for file, color, label in zip(files, colors, labels):
        data = np.loadtxt(file)
        time_arr = data[:,0]
        var_arr = data[:,1] # nm^2
        spectral_shift_arr = data[:,2] # eV


        var_arr = (var_arr - var_arr[0])
        spectral_shift_arr = (spectral_shift_arr - spectral_shift_arr[0])

        ax_var.plot(time_arr, var_arr, color=color, label=label)
        ax_var.set_xlabel('Time (ns)')
        ax_var.set_ylabel(r'$\sigma^2$ (n$m^2$)')

        ax_var.legend(frameon=False)
        ax_var.set_yticks([0, 1000, 2000])
        
        ax_var.set_xlim(-0.05, 5.05)
        ax_var.set_ylim(0, 2500)


        ax_deltaE.plot(time_arr, spectral_shift_arr, color=color, label=label)
        ax_deltaE.set_xlabel('Time (ns)')
        ax_deltaE.set_ylabel(r'$\Delta E$ (meV)')
        #ax_deltaE.legend(frameon=False)
        ax_deltaE.set_yticks([-20, -10, 0])
        ax_deltaE.set_xlim(-0.05, 5.05)
        ax_deltaE.set_ylim(-25,1)

    # save as pdf inside the folder where the data is
    fig_var.savefig(fol + 'variance_over_time.pdf', dpi=300, bbox_inches='tight')
    fig_deltaE.savefig(fol + 'delta_e_over_time.pdf', dpi=300, bbox_inches='tight')


def plot_bandgap_vs_crystal_size():

    names = ['C16', 'C12', 'C8']
    colors = ['b', 'g', 'r']
    #emission = [5213, 521, 524] # in nm, from the PL spectra of the samples
    # in kT
    bandgaps = [93.65, 92.21, 91.69] # kT for each crystal size
    bandgaps_ev = [bandgap*0.024 for bandgap in bandgaps] # convert to eV
    def energy_from_size(r):
        return 0.024*(29.2725*np.exp(-0.3446*2*r) + 91.5)
    r_linspace = np.linspace(7, 20, 100)
    energy_empirical = energy_from_size(r_linspace/2)

    # distances between the centers of the crystals, measured by SEM, in nm
    #Cn_distance = [9.61, 12.51, 16.23] # measured by Simon with SEM. In nm
    #Cn_distance_var = [1.22, 2.59, 4.57]
    diameters = [7.5, 10.8, 15.0]
    diameters_vars = [1.5, 2.9, 4.8] # just an estimate of the variance in the radius measurements, in nm

    # plot with errorbar associated with the distance measurements
    fig, ax = plt.subplots(figsize=(7,6))
    for i in range(len(names)):
        radius = diameters[i]
        radius_var = diameters_vars[i]
        bandgap_ev = bandgaps_ev[i]
        color = colors[i]
        name = names[i]

        ax.errorbar(radius, bandgap_ev, xerr=radius_var, fmt='o', color=color, label=name,
                    markersize = 14, markerfacecolor = 'white', markeredgewidth = 3,
                    elinewidth=3, capsize=0, zorder = 0
                    )
        
    ax.plot(r_linspace, energy_empirical, color='k', linestyle='-', 
            linewidth = 3, zorder = 1 )

    ax.set_xlabel('Nanoparticle size (nm)')
    ax.set_ylabel('Bandgap Energy (eV)')

    ax.set_xlim(5, 20)
    ax.set_ylim(2.19, 2.255)

    plt.savefig('data\simulations/variances_fine_adjustment/bandgap_vs_crystal_size.pdf', dpi=300, bbox_inches='tight')

# new iteration of the publlication has fig 5 in the main text with some changes to some figures 

def main():
    #plot_crystal_structures()
    crystal_plots_sims()
    #plot_variance_sizes_results() # deprecated 
    #plot_energetic_spatial_disorder_dynamics()
    #plot_bandgap_vs_crystal_size()


#%%
if __name__ == '__main__':
    main()
# %%

