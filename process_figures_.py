#%%

# Author: Enrique Arévalo Rodríguez

import matplotlib
import numpy as np
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import warnings
warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output

import json
# Set up the environment for plotting

import matplotlib.font_manager as font_manager
from matplotlib import rcParams

font = font_manager.FontProperties(family='Arial',                     
                                   style='normal', size=40)

rcParams['font.family'] = font.get_name()
rcParams['font.size'] = 20
rcParams['font.style'] = 'normal'

# import functions to process the data

names = ['c16', 'c12', 'c8']  # names of the samples

import sys
import os
from diff_eval_functions import * 
import glob

def Diffusion_curve(t, D, alpha):
    return 2 * D * t**alpha

def pulses_from_rep_rate(Rep_rate):

    if Rep_rate == 80:
        NPulses = 7
    elif Rep_rate == 40:
        NPulses = 4
    elif Rep_rate == 20:
        NPulses = 2
    else:
        NPulses = 1

    return NPulses
def good_bins(metadata):
    try:
        TimeRes = metadata['Time resolution (ps)']
    except:
        TimeRes = 64
    RepRate = metadata['Rep_rate (MHz)']

    picoharp_used = metadata['Picoharp used']

    if picoharp_used == 'PH300':
        NPulses = pulses_from_rep_rate(RepRate)

    elif picoharp_used == 'PH330':
        NPulses = 1
    else:
        raise Exception('Picoharp not recognized')
    
    Time_betPulse = 1/(RepRate)*10**12 # MHz to Hz and then to s, to ps
    GoodBins = NPulses * Time_betPulse / TimeRes

    return int(GoodBins)

def double_exp_decay(t, A1, tau1, A2, tau2, c):
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + c

def powerlaw_binning(x, y, a=1.5, b=1, min_bin=1, max_bin=None):
    """
    Non-linear binning of x and y data.
    Bin size increases as: bin_size = min_bin + int(a * (i+1)**b)
    Parameters:
        x: array-like, x values
        y: array-like, y values
        a: float, scaling factor for bin size growth
        b: float, exponent for power law
        min_bin: int, minimum bin size (default 1)
        max_bin: int or None, maximum bin size (default None)
    Returns:
        x_binned, y_binned: arrays of binned x and y
    """
    n = len(x)
    idx = 0
    bins = []
    y_bin_err = []
    while idx < n:
        bin_size = min_bin + int(a * (idx+1)**b)
        if max_bin is not None:
            bin_size = min(bin_size, max_bin)
        next_idx = min(idx + bin_size, n)
        bins.append((idx, next_idx))
        idx = next_idx

    x_binned = [np.mean(x[start:end]) for start, end in bins]
    y_binned = [np.mean(y[start:end]) for start, end in bins]
    y_bin_err = [ np.std(y[start:end]) / np.sqrt(end - start) for start, end in bins]
    return np.array(x_binned), np.array(y_binned), np.array(y_bin_err)


def extract_results():
    """
    Extracts results from the diffusion analysis.
    
    Basically, get the data do the basic analysis and then return:
    1. Time values for each bin
    2. Merged, the array with the properly formatted data
    3. MergedNormalized, the normalized data
    4. Delta Sigma Square, the MSD values
    5. Errorsquare, the errors in the MSD values
    6. TimeValuesMerged, the time values for the merged data

    All of these for each dataset (c16, c12, c8).

    """
    # folder should be a subfrectory called tplm
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data\\tplm')

    # grab all subdirectories
    subdirs = [f.path for f in os.scandir(folder) if f.is_dir()]
    # make sure order is c16, c12, c8
    subdirs.sort()  # sort alphabetically
    subdirs = [ subdirs[1], subdirs[0], subdirs[2]]  # reorder to c16, c12, c8

    # names = ['c16', 'c12', 'c8']


    binning_time_function =  ConstantBins
    binning_time_params = [50] # 25 = 10 bins per ns (4ps * n = x bin/ns), assuming 4 ps resolution 


    timevalues_all  = []
    deltaSigmaSquare_all = []
    errorsquare_all = []
    timeValuesMerged_all = []

    Merged_all = []
    MergedNormalized_all = []
    SpatialValues = []

    # for numerical sims
    sims_fol = 'data\\simulations\\variances_fine_adjustment\\'
    files_sims = glob.glob(sims_fol + '*dat')
    files_sims = [files_sims[1], files_sims[0], files_sims[2]] # reorder to c16, c12, c8

    timeValues_sims_all = []
    var_sims_all = []
    energy_sims_all = []


    # loop through each subdirectory
    for i, subdir in enumerate(subdirs):
        print(f'Processing {subdir}')
        name = names[i]
        files = glob.glob(subdir + '/*.csv')
        metadata_path = glob.glob(folder + '2*meas_params.txt')
        TimeResolution, AcquisitionTime, Binning, laser_pw, laser_div, NPulsesSeen, xmin, xmax, ymin, ymax, SpatialResolution, MaxRepRate = extract_parameters_from_metadata(metadata_path,  subdir+'/')
        rawData = LoadData(files)
        dat_OnePeak_All, dataBinned, dataBinnedNorm = rawdata_to_readydata(rawData, TimeResolution, laser_div, NPulsesSeen, MaxRepRate, binning_time_function, binning_time_params)
        spatialValues, timeValuesUncut, BinningArrayTimeUncut = get_spatial_time_values(dat_OnePeak_All, TimeResolution, SpatialResolution, binning_time_function, binning_time_params)
        Merged, MergedNormalized, spatialValues, timeValuesMerged, BinningArrayTime = merge_selected_arrays(dataBinned, np.arange(len(files)), [0,20], timeValuesUncut, BinningArrayTimeUncut, SpatialResolution)
        # get the intenisty from Merged
        DataSummed, lifetime_fit, lifetime_pcov = plot_merge_lifetimes_fit(Merged, timeValuesMerged, timeSlice = [4,0] ,saveFig = False, img = False)
        # if subdir includes c8 img = True, else False
        popt, success = plot_merge_diffusion_map(Merged, spatialValues*1e3, timeValuesMerged, BinningArrayTime, np.arange(len(files)), SpatialResolution, cut_time= 19,saveFig = False, folder = folder, img = True)
        binningSigmas = [0,1,1] # no extra binning for the sigma. [0,,1,1] or [0,1.25,1] typical values for the binning of the sigma
        binningSigmasFunction = PowerLawForBins

        if name == 'c8': # the first point in c8 is a bit off. This happens due to getting the t=0 point is difficult sometimes due to noise
            cutoff = 1
        else:
            cutoff = 0
        timeValues, tend_index, deltaSigmaSquare, errorsquare, errorsquaresum, sigmas = extract_MSD(MergedNormalized, timeValuesMerged, popt, success, cutoff, 6, binningSigmasFunction, binningSigmas, t1_index = 0)

        # if you want to add simulation results, uncomment the few following lines + the end of the function where we return the results
        '''        
        file_sims = files_sims[i]
        data_sims = np.loadtxt(file_sims)

        timeValues_sims_all.append(data_sims[:, 0])
        var_sims_all.append(data_sims[:, 1])
        energy_sims_all.append(data_sims[:, 2])
        '''

        # put results into a list, we will need them to plot our main figures in a bit
        timevalues_all.append(timeValues)
        deltaSigmaSquare_all.append(deltaSigmaSquare)
        errorsquare_all.append(errorsquare)
        timeValuesMerged_all.append(timeValuesMerged)

        Merged_all.append(Merged)
        MergedNormalized_all.append(MergedNormalized)
        SpatialValues.append(spatialValues)
    # return all the results
    return timevalues_all, deltaSigmaSquare_all, errorsquare_all, Merged_all, MergedNormalized_all, names, timeValuesMerged_all, SpatialValues,# timeValues_sims_all, var_sims_all, energy_sims_all

def calculate_diffusion_coefficients(timevalues_all, deltaSigmaSquare_all, errorsquare_all, names, timeValuesMerged_all,):

    """
    Calculates the diffusion coefficients for each dataset.

    Returns:
    1. Fitted diffusion coefficients (D)
    2. Fitted alpha values
    3. Errors in diffusion coefficients (D_err)
    4. Errors in alpha values (alpha_err)
    5. MSD_analy_all: the calculated MSD values from the fitted diffusion coefficients
    6. time_arr_analy_all: the time array for the MSD analysis
    """

    MSD_analy_all = []
    Ds = []
    alphas = []
    err_Ds = []
    err_alphas = []
    time_arr_analy_all = []

    for i, deltaSigmaSquare in enumerate(deltaSigmaSquare_all):
        name = names[i]

        # if the name is c8, we need to set the cutoff to 1
        if name == 'c8':
            cutoff = 1
        else:
            cutoff = 0

        print(f'Calculating diffusion coefficients for {name}')
        # get the time values for this dataset
        timeValues = timevalues_all[i]
        # get the errorsquare for this dataset
        errorsquare = errorsquare_all[i]
        # get the time values merged for this dataset
        timeValuesMerged = timeValuesMerged_all[i]

        # alrighty, now its time to try to fit using a Diffusion model
        time_fit = 6
        time_fit_index = np.argmin(np.abs(timeValues - time_fit))    
        if cutoff == 0: # only becuase c8 has a cutoff of 1, we can use the full data
            msd_bin = deltaSigmaSquare.reshape(-1, 2).mean(axis=1)  # bin the data to 10 points

            popt, pcov = curve_fit(Diffusion_curve, timeValuesMerged[cutoff:time_fit_index][::2] - timeValuesMerged[cutoff], msd_bin[:time_fit_index//2], 
                    p0=[1e-3, 1], bounds=(0, [1e-2, 1.2]) # bounds for D, alpha
                    )
            # get the diffusion coefficient and alpha
            D = popt[0]
            alpha = popt[1]
            if alpha > 1.1:
                print(f'Alpha is too high: {alpha}, setting to 1.1')
                alpha = 1.1
            # get the error of the diffusion coefficient and alpha
            D_err = np.sqrt(pcov[0][0])
            alpha_err = np.sqrt(pcov[1][1])

            # calculate a curve using these values
            time_arr_analy = np.linspace(0, 10, 1000)
            MSD_analy = Diffusion_curve(time_arr_analy, D, alpha)
        else: # same for c8
            msd_bin = deltaSigmaSquare[:-1].reshape(-1, 2).mean(axis=1)  # bin the data to 10 points, if it fails, take all but last point

            # fit the data to the curve
            popt, pcov = curve_fit(Diffusion_curve,
                        timeValuesMerged[cutoff:time_fit_index][::2] - timeValuesMerged[cutoff], msd_bin[:time_fit_index//2], 
                        p0=[1e-3, 1], bounds=(0, [1e-2, 1.2]) # bounds for D, alpha
                        )
            # get the diffusion coefficient and alpha
            D = popt[0]
            alpha = popt[1]

            if alpha > 1.1:
                print(f'Alpha is too high: {alpha}, setting to 1.1')
                alpha = 1.1
            # get the error of the diffusion coefficient and alpha
            D_err = np.sqrt(pcov[0][0])
            alpha_err = np.sqrt(pcov[1][1])
            # calculate a curve using these values
            time_arr_analy = np.linspace(0, 10, 1000)
            MSD_analy = Diffusion_curve(time_arr_analy, D, alpha)

        # get the values of D, alpha, D_err, alpha_err
        MSD_analy_all.append(MSD_analy)
        alphas.append(alpha)
        err_alphas.append(alpha_err)
        Ds.append(D)
        err_Ds.append(D_err)
        time_arr_analy_all.append(time_arr_analy)

    return MSD_analy_all, Ds, alphas, err_Ds, err_alphas, time_arr_analy_all

def plot_Diff_Map(MergedNormalized, SpatialValues, timeValuesMerged_all, names, SAVE = False):    # first. Plot the diffusion maps

    """
    Plots the diffusion maps for each dataset.
    MergedNormalized: list of 2D arrays with the normalized data
    SpatialValues: list of spatial values for each dataset
    timeValuesMerged: list of time values for each dataset
    names: list of names for each dataset
    """
    for i, MergedNorm in enumerate(MergedNormalized):
        name = names[i]
        spatialValues = SpatialValues[i]
        timeValuesMerged = timeValuesMerged_all[i]

        MergedNormalized = MergedNorm
        
        fig, ax = plt.subplots( figsize=(6, 6))
        CS = ax.contourf(MergedNormalized, levels=20, cmap='plasma', 
                        interpolation='bilinear',
                        extent=[spatialValues[0]*1e3, spatialValues[-1]*1e3, timeValuesMerged[0], timeValuesMerged[-1]],
                        location='lower', origin='lower')

        S2 = ax.contour(CS, levels=CS.levels[::2], colors='white',
                        linewidths=1, linestyles='dashed', 
                        #extent=[spatialValues[0]*1e3, spatialValues[-1]*1e3, timeValuesMerged[0], timeValuesMerged[-1]],
                        location='lower', origin='lower'
                        )

        ax.set_xlabel(r'Spatial position ($\mu$m)', fontsize=14)
        ax.set_ylabel('Time (ns)', fontsize=14)
        #plt.colorbar(label='Normalized Intensity')
        #plt.title('Merged data')

        ax.set_xticks( (-1,0,1))
        ax.set_yticks((5, 10, 15, ))
        # add text with the name of the sample
        ax.text(0.1, 0.95, name, transform=ax.transAxes, fontsize=16,
                verticalalignment='top', horizontalalignment='center',
                bbox=dict(facecolor='white', alpha=1, edgecolor='none'))
    
        if SAVE:
            plt.savefig(f'diff_contourf_{name}.svg', dpi=300)

        # also plot the fits for 

def plot_MSD(timevalues_all, deltaSigmaSquare_all, MSD_analy_all, time_arr_analy_all, names, colors, SAVE = False):

    fg, ax = plt.subplots(figsize=(12, 6))

    for i, msd in enumerate(deltaSigmaSquare_all):
        
        name = names[i]
        ax.plot(timevalues_all[i][:], deltaSigmaSquare_all[i], label=name + ' (binned)',
                    color=colors[i], linestyle='', linewidth=1.5, 
                    marker='o', markerfacecolor='white', markeredgecolor=colors[i],    
                    markeredgewidth=3.5, markersize=10, 
                    )
            
        # plot in ax2 D_analy vs time_arr_analy
        ax.plot(time_arr_analy_all[i], MSD_analy_all[i], label=name + ' (fit)',
                    color=colors[i], linestyle='--', linewidth=1.5, 
                    marker='', markerfacecolor='white', markeredgecolor=colors[i],    
                    markeredgewidth=3.5, markersize=10, 
                    )

    ax.set_xlabel('Time (ns)', )
    ax.set_ylabel(r'MSD ($\mathrm{\mu}\mathrm{m^2}$) = $\sigma^2$(t) -$\sigma^2$(0)', )

    ax.legend( bbox_to_anchor=(.0, 1), loc='upper left', frameon=False,
            )

    ax.set_xlim(-0.1, 6)
    ax.set_ylim(-0.0001, 0.0035)

    ax.minorticks_off()

    ax.tick_params(axis='both', which='major', direction='out', length=6, width=1.5,
                top=False, right=False, )
    ax.tick_params(axis='both', which='minor', direction='out', length=3, width=1.5,
                top=False, right=False, )

    ax.set_xticks([0.0, 2.5, 5, ])
    ax.set_yticks([0.0, 0.001, 0.002, 0.003])


    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # bigger font for the labels, scientific notation
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # save
    plt.savefig('MSD_vs_Time.svg', dpi=300)

def plot_alphas(alphas, err_alphas, ligand_lengths, names, Cn, Cn_err, colors, SAVE = False):
    fig,ax = plt.subplots(figsize=(6, 5))
    # plot alpha vs ligand length

    # red color
    color = "#000000ff"

    for i in range(len(alphas)):
        name = names[i]
        #length = ligand_lengths[i]
        Cn_value = Cn[i]
        ax.errorbar(Cn_value, alphas[i], xerr=Cn_err[i], yerr=err_alphas[i],
                    color=color, marker='o', markersize=20, linestyle='None',
                    label=name, linewidth=1.5, markerfacecolor='white',
                    markeredgecolor=colors[i], markeredgewidth=4.0,
                    elinewidth = 3, ecolor= colors[i],
                    )

    ax.set_xlabel(r'$d_{c-c}$ (nm)', )
    ax.set_ylabel(r'$\mathrm{\alpha}$',  )

    ax.set_xlim(7, 17)
    ax.set_ylim(0, 1)

    #ax.legend(bbox_to_anchor=(1.0, 1), loc='upper right', fontsize=12, frameon=False)
    ax.minorticks_off()
    # larger ticklabels
    ax.tick_params(axis='both', which='major', direction='out', length=6, width=1.5, 
                    top=False, right=False, labelsize = 24)
    ax.tick_params(axis='x', which='minor', direction='out', length=0, width=1.5, 
                    top=False, right=False, )
    ax.tick_params(axis='y', which='minor', direction='out', length=3, width=1.5, 
                    top=False, right=False, )

    ax.set_xticks([8, 12, 16])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    # grid only for y axis
    ax.axhline(y=0.75, color='black', linestyle=':', linewidth=2.5, alpha=0.6)
    ax.axhline(y=0.5, color='black', linestyle=':', linewidth=2.5, alpha=0.6)
    ax.axhline(y=0.25, color='black', linestyle=':', linewidth=2.5, alpha=0.6)


    # add tick on y = 0 and 1 but no grid line
    ax.set_yticks([0.0, 0.5, 1.0], minor=True)
    ax.tick_params(axis='y', which='minor', direction='out', length=0, width=1.5, 
                    top=False, right=False, )

    # save
    plt.savefig('Alpha_vs_Ligand_Length.svg', dpi=300)

def plot_D_analy( MSD_analy_all, time_arr_analy_all, names, colors):

    fig, ax2 = plt.subplots(figsize=(5.5, 5))
    for i, msd in enumerate(MSD_analy_all):
        name = names[i]
        time_arr_analy = time_arr_analy_all[i]
        
        D_analy = np.gradient(msd, time_arr_analy[1] - time_arr_analy[0])   # numerical derivative
        # plot in ax2 D_analy vs time_arr_analy
        ax2.plot(time_arr_analy, D_analy*1e6, color=colors[i], label=name, linestyle='-', linewidth=1.5)


    # plot the D_analy vs time_arr_analy
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel(r'D ($\mathrm{nm^2/ns}$)')

    #ax2.legend()

    ax2.set_xlim(-0.1, 10)
    ax2.set_ylim(-0.000, 0.0022*1e6)

    ax2.minorticks_off()
    ax2.tick_params(axis='both', which='major', direction='out', length=6, width=1.5, 
                    top=False, right=False, )
    ax2.tick_params(axis='both', which='minor', direction='out', length=3, width=1.5, 
                    top=False, right=False, )
    ax2.set_xticks([0.0, 5, 10])

    ax2.set_yticks([0.0, 0.001*1e6, 0.002*1e6, ])

    # try saving the figure in the folder

    # scientific notation for the y axis
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # savw
    plt.savefig('Diffusion_Coefficient_vs_Time.svg', dpi=300)

def plot_lifetimes_(Merged_all, timeValuesMerged_all, names, colors, Cn, Cn_err , SAVE = False):

    # get the intensiy for each time point as the sum of the Merged for that index
    
    fig, ax = plt.subplots(figsize=(6, 5))
    fig, ax2 = plt.subplots(figsize=(6, 5))

    for i, Merged in enumerate(Merged_all):
        name = names[i]
        timeValuesMerged = timeValuesMerged_all[i]

        # get the intensity for each time point
        intensity = np.sum(Merged, axis=1)
        # normalize the intensity
        intensity /= np.max(intensity)

        ax.plot(timeValuesMerged, intensity, label=name, color=colors[i], linestyle='-', linewidth=1.5)

        # get tau by using the 1/e decay of the profile_tot

        tau_e = timeValuesMerged[np.where(intensity < intensity[0] / np.e)[0][0]]  # find the first point where the profile is less than 1/e of the initial value

        # fit the profile_tot to a double exponential decay
        p0 = [intensity[0], tau_e, intensity[0] / 10, tau_e * 10, 0.01]  # initial guess
        bounds = ([0, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])  # bounds for the parameters
        try:
            time_arr_analy = np.linspace(0, 10, 1000)  # time array for the analysis
            popt, pcov = curve_fit(double_exp_decay, timeValuesMerged, intensity, p0=p0, bounds=bounds)
            A1, tau1, A2, tau2, c = popt
            # calculate the decay curve
            decay_curve = double_exp_decay(time_arr_analy, *popt)
            ax.plot(time_arr_analy, decay_curve, color=colors[i], linestyle='--', linewidth=1.5)
            ax.text(0.1, 0.1 + 0.2*i, f'{name}:\nTau1: {tau1:.2f} ns\nTau2: {tau2:.2f} ns',
                color=colors[i], transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor=colors[i], boxstyle='round,pad=0.3'))
            ax.axhline(y=intensity[0] / np.e, color=colors[i], linestyle=':', linewidth=1.5, alpha=0.6)


        except:
            print(f'Could not fit {name} to a double exponential decay, using 1/e decay instead')
            A1 = intensity[0]
            tau1 = tau_e
            A2 = intensity[0] / 10
            tau2 = tau_e * 2
            c = 0
            decay_curve = A1 * np.exp(-time_arr_analy / tau1) + A2 * np.exp(-time_arr_analy / tau2) + c
            ax.plot(time_arr_analy, decay_curve, color=colors[i], linestyle='--', linewidth=1.5)
        
        # find the time where the total number of excitons is 1/e
        tau_ld = timeValuesMerged[np.where(intensity < intensity[0] / np.e)[0][0]]  # find the first point where the profile is less than 1/e of the initial value
        ax.axvline(x=tau_ld, color=colors[i], linestyle=':', linewidth=1.5, alpha=0.6)

        ax2.errorbar(Cn[i], tau_ld, xerr=Cn_err[i], color=colors[i], marker='o', markersize=20, linestyle='None',
                 label=name, linewidth=1.5, markerfacecolor='white',
                markeredgecolor=colors[i], markeredgewidth=4.0,
                )
            
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Integrated Amplitude')

    ax.set_yscale('log')

    ax2.set_xlabel(r'$d_{c-c}$ (nm)')
    ax2.set_ylabel(r' $\mathrm{\tau_{1/e} (ns)}$')
    ax2.set_xlim(7, 17)

    ax2.set_ylim(5, 19)

    ax2.set_xticks([8, 12, 16])
    ax2.set_yticks([6,10, 14, 18])

    ax2.minorticks_off()
    ax2.tick_params(axis='both', which='major', direction='out', length=6, width=1.5, 
                    top=False, right=False, labelsize = 24)
    ax2.tick_params(axis='x', which='minor', direction='out', length=3, width=1.5,
                    top=False, right=False, )

    # Add only horizontal grid lines (no vertical ones)
    ax2.axhline(y=6, color='black', linestyle=':', linewidth=2.5, alpha=0.6)
    ax2.axhline(y=8, color='black', linestyle=':', linewidth=2.5, alpha=0.6)
    ax2.axhline(y=10, color='black', linestyle=':', linewidth=2.5, alpha=0.6)
    ax2.axhline(y=12, color='black', linestyle=':', linewidth=2.5, alpha=0.6)
    ax2.axhline(y=14, color='black', linestyle=':', linewidth=2.5, alpha=0.6)
    ax2.axhline(y=16, color='black', linestyle=':', linewidth=2.5, alpha=0.6)
    ax2.axhline(y=18, color='black', linestyle=':', linewidth=2.5, alpha=0.6)

    # save
    plt.savefig('Lifetimes_vs_Ligand_Length.svg', dpi=300)



def plot_diff_lengths( Merged_all, timeValuesMerged_all, deltaSigmaSquare_all, errorsquare_all, Cn, Cn_err, names, colors):

    fig, axLd = plt.subplots(figsize=(5.5, 5))
    fig, ax2 = plt.subplots(figsize=(6, 5))

    for i, Merged in enumerate(Merged_all):
        name = names[i]
        #timeValuesMerged = timeValuesMerged_all[i]
        intensity = np.sum(Merged, axis=1)

        fraction_surviving = intensity / intensity[0]  # fraction of surviving excitons

        net_displacement = np.sqrt(deltaSigmaSquare_all[i])* 1e3 # nm  # convert to nm

        binned_surviving, binned_net_displ, binned_net_displ_err = powerlaw_binning(fraction_surviving, net_displacement,
                                                            a=1., b=0.5, min_bin=1, max_bin=None)
        
        axLd.plot( binned_net_displ, binned_surviving, 
                  color=colors[i], marker='o', markersize=10, linestyle='None',
                label=name, markeredgewidth=4, markerfacecolor='white',)

        axLd.axhline( y= 1/np.e, color = 'black', 
                    linestyle=':', linewidth=2.5, alpha=0.6)

        # find the index of binned surviving where it is less than 1/e
        index_1_e = np.where(binned_surviving < 1/np.e)[0]
        # get the value of net_displ at that index, that is the diffusion length, Ld
        if index_1_e.size > 0:
            Ld = binned_net_displ[index_1_e[0]]
        #    Ld_err = binned_net_displ_err[index_1_e[0]]
        else:
            Ld = np.nan

        axLd.set_xlabel('Net Displacement (nm)')
        axLd.set_ylabel('Fraction Surviving')
        axLd.set_title('Diffusion Lengths')
        axLd.legend()
        plt.tight_layout()

        # plot diff length vs ligand length

        Cn_value = Cn[i]
        ax2.errorbar(Cn_value, Ld, xerr=Cn_err[i], color=colors[i], marker='o', markersize=20, linestyle='None',
            label=name, linewidth=3.5, markerfacecolor='White',
            markeredgecolor=colors[i], markeredgewidth=4.,
            )

    ax2.set_xlabel(r'$d_{c-c}$ (nm)')
    ax2.set_ylabel(r'$\mathrm{L_D} (nm)$')

    ax2.set_xlim(7, 17)
    ax2.set_ylim(0, 90)

    ax2.set_xticks([8, 12, 16])
    ax2.set_yticks([0, 20, 40, 60])

    #ax2.legend()
    # try saving the figure in the folder

    ax2.minorticks_off()
    ax2.tick_params(axis='both', which='major', direction='out', length=6, width=1.5, 
                    top=False, right=False, labelsize = 24)
    ax2.tick_params(axis='x', which='minor', direction='out', length=0, width=1.5, 
                    top=False, right=False, )
    ax2.tick_params(axis='y', which='minor', direction='out', length=3, width=1.5, 
                    top=False, right=False, )

    ax2.set_xticks([8, 12, 16])
    ax2.set_yticks([0, 20, 40, 60, 80])

    # grid only for y axis
    ax2.axhline(y=20, color='black', linestyle=':', linewidth=2.5, alpha=0.6)
    ax2.axhline(y=40, color='black', linestyle=':', linewidth=2.5, alpha=0.6)
    ax2.axhline(y=60, color='black', linestyle=':', linewidth=2.5, alpha=0.6)
    ax2.axhline(y=80, color='black', linestyle=':', linewidth=2.5, alpha=0.6)

    # save the figure
    plt.savefig('Diffusion_Lengths_vs_Ligand_Length.svg', dpi=300)


def read_trpl_data(file, metadata_file):

    # read metadata
    with open(metadata_file) as f:
        metadata = json.load(f)

    NPulsesSeen = pulses_from_rep_rate(metadata['Rep_rate (MHz)'])

    # assumes its a .csv with the first column being the time array and the rest the data.

    df = pd.read_csv(file)

    Time_arr = df['Time (ps)'].values

    powers = list(df.columns[2:])
    #powers = [float(i) for i in powers]

    # here it is. For each data row, perform the analysis

    ready_data = []

    for pow in powers:

        data_pw = df[pow].values
        BinsBetweenPulse = good_bins(metadata)
        # Here we need differnet analysis functions for PH300 and PH330

        if metadata['Picoharp used'] == 'PH300':
            
            # analysis for PH300. Merge pulses given the Rep rate
            for pulse_nr in range(NPulsesSeen):
                if pulse_nr == 0:
                    '''
                    FIRST POSITION
                    '''
                    counts = data_pw[:BinsBetweenPulse//NPulsesSeen]
                else:
                    '''
                    SECOND TO LAST POSITIONS
                    '''
                    counts += data_pw[BinsBetweenPulse//NPulsesSeen*pulse_nr:BinsBetweenPulse//NPulsesSeen*(pulse_nr+1)]
            
            # roll to get the max position in the first position
            max_pos = np.argmax(counts)
            counts = np.roll(counts, -max_pos)

            ready_data.append(counts) # perfect, now we have the data ready for analysis


        elif metadata['Picoharp used'] == 'PH330':

            # no merge pulses for PH330, but we need to cut a bit of the data
            counts = data_pw[1:BinsBetweenPulse] # chec up to which bin we need to cut

            # roll
            max_pos = np.argmax(counts)
            counts = np.roll(counts, -max_pos)
        
            ready_data.append(counts)
            
        else:
            raise Exception('Picoharp not recognized')
        
    # plot all normlaized in the same plot
    #plt.figure()
    #for i in len(powers):
    #    plt.plot(Time_arr, ready_data[i]/np.max(ready_data[i]), label = powers[i])
    #plt.xlabel('Time (ps)'), plt.ylabel('Counts (a.u.)')
    #plt.legend()

    Time_arr = Time_arr[:len(ready_data[0])]
        
    return np.array(ready_data), Time_arr, powers, metadata

def get_plot_powerData():
    # folder in data should be in a directory called power
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data\\power/')
    # get subdirectories
    subdirs = [f.path for f in os.scandir(folder) if f.is_dir()]
    # make sure c8 is the last subdir
    subdirs = [subdirs[1], subdirs[0], subdirs[2]] # c16, c12, c8
    names = ['c16', 'c12', 'c8']
    # get the csv and json files in the folder
    
    for i, subdir in enumerate(subdirs):
        name = names[i]
        fig, ax = plt.subplots(figsize=(10, 6))

        select = 0 # select all the data in the folder
        file = glob.glob(subdir + '/*.csv')[select] # assuming only one .csv file in the folder
        metadata_file = glob.glob(subdir + '/*.json')[select]
        print(f'Processing {file} and {metadata_file}')
        # analyze data
        ready_data, Time_arr, powers_, _= read_trpl_data(file, metadata_file)

        #turn powers_ to float
        powers_ = [float(i) for i in powers_] # convert to float
        powers_ = np.array(powers_) # convert to numpy array

        # order the data by powers
        order = np.argsort(powers_)
        ready_data = [ready_data[i] for i in order] # sort the data according to the powers
        powers_ = [powers_[i] for i in order] # sort the powers according to the order of the data

        np.array(ready_data).sum(axis=1) # sum the data over all powers
        ready_data = ready_data / np.max(np.array(ready_data).sum(axis=1))

        def power_law_fit(x, a, b, c):
            return a * x ** b + c

        # fit the data to a power law
        popt, pcov = curve_fit(power_law_fit, powers_, np.array(ready_data).sum(axis=1), p0=[3, 1, 1e5], maxfev=100000,
                            bounds = ([0, 0, 0], [1e8, 2, 1e6]))

        ax.plot(powers_, np.array(ready_data).sum(axis=1), 'o', label='Edge')
        pow = np.linspace(min(powers_), max(powers_), 100)
        ax.plot(pow, power_law_fit(np.array(pow), *popt), label='Power law fit', color='red')
        ax.set_xlabel(r'Power ($\mu W$)', )
        ax.set_ylabel(r'Intensity (a.u.)',)

        ax.text(0.35, 0.9, f'Power law fit exponent: {popt[1]:.2f}', 
            fontsize=24, ha='center', va='center', transform=ax.transAxes)

#%%
def main():

    """
    The code is not super optimized so it may take a bit to run, tested on several machines and it works well.

    Took
    """
    
    # Extract results from the diffusion analysis:
    # if no simulation this line
    timevalues_all, deltaSigmaSquare_all, errorsquare_all, Merged_all, MergedNormalized_all, names, timeValuesMerged_all, SpatialValues, = extract_results()
    # if simulation this one
    #timevalues_all, deltaSigmaSquare_all, errorsquare_all, Merged_all, MergedNormalized_all, names, timeValuesMerged_all, SpatialValues, timeValues_sims_all, var_sims_all, energy_sims_all = extract_results()
    print("Results extracted successfully.")
    
    # make the fits
    MSD_analy_all, Ds, alphas, err_Ds, err_alphas, time_arr_analy_all = calculate_diffusion_coefficients(timevalues_all, deltaSigmaSquare_all, errorsquare_all, names, timeValuesMerged_all)
    print("Diffusion coefficients calculated successfully.")

    # Start plotting the results

    colors = ['blue', 'green', 'red',]
    names = ['c16', 'c12', 'c8']
    ligand_lengths = [16, 12, 8]  # in carbons
    Cn_distance = [9.61, 12.51, 16.23] # measured by Simon with SEM. In nm
    Cn_distance_var = [1.22, 2.59, 4.57]

    # Plot the diffusion maps. 4a (+ SI)
    plot_Diff_Map(MergedNormalized_all, SpatialValues, timeValuesMerged_all, names)

    # plot MSD (4b)
    plot_MSD(timevalues_all, deltaSigmaSquare_all, MSD_analy_all, time_arr_analy_all, names, colors)

    # Next is the alpha vs ligand length plot (4c)
    plot_alphas(alphas, err_alphas, ligand_lengths, names, Cn_distance, Cn_distance_var, colors)

    # plot the analytic diffusion coefficient as a function of time (4d)
    plot_D_analy( MSD_analy_all, time_arr_analy_all, names, colors)

    # Plot the lifetime + fits (SI), also plot tau(1/e) vs ligand length (inset)
    plot_lifetimes_(Merged_all, timeValuesMerged_all, names, colors, Cn_distance, Cn_distance_var)

    plot_diff_lengths( Merged_all, timeValuesMerged_all, deltaSigmaSquare_all, errorsquare_all, Cn_distance, Cn_distance_var, names, colors)
    
    # one more figure... Power dependent data (SI)
    get_plot_powerData()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
    plt.show()  # Show the plots interactively



# %%
