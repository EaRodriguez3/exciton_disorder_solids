

#%%

# Author: Antonella Cutrupi

import numpy as np
import os
import json
import re
import glob

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.optimize import curve_fit
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d

# Set up the environment for plotting

import matplotlib.font_manager as font_manager
from matplotlib import rcParams

font = font_manager.FontProperties(family='Arial',
                                   
                                   style='normal', size=40)

rcParams['font.family'] = font.get_name()
rcParams['font.size'] = 20
rcParams['font.style'] = 'normal'

import warnings
warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output

def load_data(filename):
    # load the data from the npz file
    data_dict = np.load(filename, allow_pickle=True)['arr_0'].item()
    
    return data_dict

def load_data_and_metadata(folder_path):
    """
    Load npz data files and their corresponding metadata json files from a folder.
    
    Args:
        folder_path (str): Path to the folder containing npz and json files
        
    Returns:
        list: List of dictionaries containing matched data and metadata files
    """
    
    # Get all npz and json files
    data_files = glob.glob(os.path.join(folder_path, '*.npz'))
    metadata_files = glob.glob(os.path.join(folder_path, '*.json'))
    
    # Create a dictionary to store metadata files by timestamp
    metadata_dict = {}
    for meta_file in metadata_files:
        timestamp = os.path.getmtime(meta_file)
        metadata_dict[timestamp] = meta_file
    
    # Match data files with metadata
    matched_pairs = []
    for data_file in data_files:
        data_timestamp = os.path.getmtime(data_file)
        
        # Find closest metadata file timestamp
        closest_timestamp = min(metadata_dict.keys(), 
                              key=lambda x: abs(x - data_timestamp))
        
        # Match if within 60 seconds
        if abs(closest_timestamp - data_timestamp) <= 60:
            with open(metadata_dict[closest_timestamp], 'r') as f:
                metadata = json.load(f)
            
            # Load the data using your existing load_data function
            data_dict = load_data(data_file)
            
            matched_pairs.append({
                'data_file': data_file,
                'metadata_file': metadata_dict[closest_timestamp],
                'data': data_dict,  # This is now a dictionary with 'wavelength', 'data', 'time' keys
                'metadata': metadata,
                'laser_power': metadata.get('laser_pw', None)
            })
            
    return matched_pairs


def gauss(x, a, x0, sigma, c):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + c

def bin_data(data, bin_size):
    num_bins = len(data) // bin_size 
    return np.array([np.mean(data[i * bin_size:(i + 1) * bin_size]) for i in range(num_bins)])

def weighted_smoothing(data, time, sigma_min=0.5, sigma_max=5):
    time_normalized = (time - np.min(time)) / (np.max(time) - np.min(time))
    sigma = sigma_min + (sigma_max - sigma_min) * time_normalized
    smoothed_data = np.zeros_like(data)
    for i in range(len(data)):
        smoothed_data[i] = gaussian_filter1d(data, sigma[i])[i]
    
    return smoothed_data

def map_median(time_arr_binned, median_arr, median_arr_scaled, data_norm,time_arr, wavelengths, time_lim_index, sigma_min, sigma_max, time_lim, Time):
    median_arr_smoothed_plot = weighted_smoothing(median_arr, time_arr_binned, sigma_min, sigma_max)
    median_arr_smoothed = weighted_smoothing(median_arr_scaled, time_arr_binned, sigma_min, sigma_max)

    return median_arr_smoothed_plot, median_arr_smoothed

def param_from_metadata(metadata_json):

    # given a metadata json file, extract the parameters

    TimeResolution = metadata_json['TimeResolution']
    AcquisitionTime = metadata_json['AcquisitionTime']
    laserPower = metadata_json['laserPower']
    laserDivider = metadata_json['laserDivider']
    spotsize = metadata_json['spot_size']
    steps = metadata_json['steps']
    center_w = metadata_json['center_w']
    wavelength_range_start = metadata_json['wavelenght_range_start']
    wavelength_range_end = metadata_json['wavelenth_range_end']
    objective = metadata_json['objective']
    maxRepRate = metadata_json['MaxRepRate']
    sample = metadata_json['sample']

    return TimeResolution, AcquisitionTime, laserPower, laserDivider, spotsize, steps, center_w,wavelength_range_start,wavelength_range_end, objective, maxRepRate, sample

def extract_parameters_from_metadata(metadata_path, folder=None):
    if isinstance(metadata_path, list):
        metadata_path = metadata_path[0]

    if os.path.isdir(metadata_path):
        metadata_path = os.path.join(metadata_path, "metadata.json")

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return param_from_metadata(metadata)

def param_analysis(laser_div, maxrep, time_res):

    time_windw = int(laser_div*1e06/(maxrep*time_res))#ns, time window for the histogram
    time_arr = np.arange(time_windw) * time_res*1e-03 #ns, time array for the histogram

    return time_windw, time_arr

def single_lifetime_trace(dat, time_windw, NPulsesSeen):

    data_av = [] # array to save the data

    for i in range(int(NPulsesSeen)):
            start_idx = i * int(time_windw/NPulsesSeen)
            end_idx = int(time_windw/NPulsesSeen) * (i+1)
            segment = dat[start_idx:end_idx]
            data_av.append(segment)

    data_av = np.array(data_av)
    data_onePulse = np.mean(data_av, axis=0)
    max_pos = np.unravel_index(np.argmax(data_onePulse, axis=None), data_onePulse.shape)
    data_roll = np.roll(data_onePulse, -max_pos[0], axis=0)[:-100] # remove the last 100 points to avoid the tail of the histogram

    return data_roll, data_onePulse, data_av

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

def energy_decay(t, e0, k):
    return e0*(1-np.exp(-k*t))

def fit_energy_decay_per_code(gaussian_peaks_dict):
    fit_results = {}

    for code, data in gaussian_peaks_dict.items():
        time = data['time']
        energy_meV = 1240000 / data['peaks_smoothed']
        y_data = energy_meV - energy_meV[0]

        mask = (time >= 0) & (time <= 20) & np.isfinite(y_data)
        time_clean = time[mask]
        y_clean = y_data[mask]

        try:
            popt, pcov = curve_fit(
                energy_decay,
                time_clean,
                y_clean,
                p0=[-5, 0.1],
                bounds=([-100, 0], [-1, 0.5])
            )
        
            perr = np.sqrt(np.diag(pcov)) 
        
            residuals = y_clean - energy_decay(time_clean, *popt)
            dof = len(y_clean) - len(popt) 
            chi2_red = np.sum((residuals)**2) / dof
            
            fit_results[code] = {
                'params': popt,
                'param_errors': perr,
                'chi2_red': chi2_red,
                'covariance': pcov
            }
            percent_error = 100 * perr / np.abs(popt)
            print(f"{code}: Fit OK — e0 = {popt[0]:.3f} ± {perr[0]:.3f} ({percent_error[0]:.1f}%), "
                f"k = {popt[1]:.3f} ± {perr[1]:.3f} ({percent_error[1]:.1f}%)")
        except RuntimeError:
            print(f"{code}: Fit fallito")
            fit_results[code] = None

    return fit_results

def lifetimes_data_clean(Time, wavelengths, data, time_arr, time_windw, MaxRepR, laser_Div, power, binning, time_lim_idx, NPulsesSeen):
    data_roll = []
    data_onePulse = []
    data_av = []

    for i in range(len(wavelengths)):
        roll, pulse, av = single_lifetime_trace(
                                            data[i][:time_windw],
                                            time_windw, NPulsesSeen)
        data_roll.append(roll)
        data_onePulse.append(pulse)
        data_av.append(av/np.max(av))

    data_roll = np.array(data_roll)
    data_onePulse = np.array(data_onePulse)
    data_av = np.array(data_av)

    # First bin the rolled data
    data_roll_binned = []
    for i in range(len(data_roll)):
        roll_binned = bin_data(data_roll[i], binning)
        roll_binned = roll_binned[:time_lim_idx]  # Truncate to time limit
        data_roll_binned.append(roll_binned)

    data_roll_binned = np.array(data_roll_binned)
    data_norm = data_roll_binned/np.max(data_roll_binned)

    # Now bin and truncate the time array to match the data
    time_arr_binned = bin_data(time_arr[:len(data_roll[0])], binning)  # Only bin the relevant part
    time_arr_binned = time_arr_binned[:time_lim_idx]  # Truncate to same length as data

    return data_roll, data_onePulse, data_av, data_roll_binned, time_arr_binned, data_norm

def darker_color(rgba, factor=1.2):
            r, g, b, a = rgba
            return (r * factor, g * factor, b * factor, a)

def plot_all_smoothed_peaks_with_fit(gaussian_peaks_dict, fit_results, fit_time_range, plot_time_range=(0, 20), ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    sorted_codes = sorted(gaussian_peaks_dict.keys(), key=lambda x: int(x[1:]))

    # Define custom colors in the desired order
    custom_colors = ['red', 'green', 'blue']
    n_colors = len(custom_colors)  # Number of custom colors
    
    custom_colors_dark = [darker_color(to_rgba(c), factor=0.7) for c in custom_colors]

    # Dictionary to store results for each code
    plot_fit_results = {}

    for i, code in enumerate(sorted_codes):
        
        data = gaussian_peaks_dict[code]
        time_arr_binned = data['time']
        gaussian_peaks_smoothed = data['peaks_smoothed']
        energy_meV = 1240000 / gaussian_peaks_smoothed
        y_data = energy_meV - energy_meV[0]

        # Mask for the fit range
        fit_mask = (time_arr_binned >= fit_time_range[0]) & (time_arr_binned <= fit_time_range[1]) & np.isfinite(y_data)
        time_fit = time_arr_binned[fit_mask]
        y_fit = y_data[fit_mask]

        # Mask for the plot range
        plot_mask = (time_arr_binned >= plot_time_range[0]) & (time_arr_binned <= plot_time_range[1]) & np.isfinite(y_data)
        time_plot = time_arr_binned[plot_mask]
        y_plot = y_data[plot_mask]

        color = custom_colors[i % n_colors]
        lighter_color = to_rgba(color, alpha=0.7)  # Colore più chiaro per i punti
        color_dark = custom_colors_dark[i % len(custom_colors)]

        # Plot all points in the plot range
        ax.plot(time_plot, y_plot, 'o', color=lighter_color, markerfacecolor='none', markersize=6, markeredgewidth=3, label=f'{code}',alpha=0.6)

        # Highlight the points used for the fit
        ax.plot(time_fit, y_fit, 'o', color=color, markersize=6, alpha=0.2)

        # Plot the fitted curve if available
        fitted_curve = None
        if fit_results.get(code) is not None:
            params = fit_results[code]['params']
            fitted_curve = energy_decay(time_plot, *params)
            ax.plot(time_plot, fitted_curve, '--', color=color_dark, linewidth=4, alpha = 1.0)

        # Save results for this code
        plot_fit_results[code] = {
            'time_plot': time_plot,
            'y_plot': y_plot,
            'y_fit': y_fit,
            'fitted_curve': fitted_curve,
            'time_fit': time_fit
        }

    ax.set_xlabel('Time (ns)', fontsize=16)
    ax.set_ylabel('$\Delta$E (meV)', fontsize=16)
    #legend
    ax.legend(fontsize=20, loc='upper right', frameon=False)
    ax.set_autoscale_on(False)
    ax.set_xlim(plot_time_range)
    ax.set_ylim(np.min(fitted_curve) - 5 if fitted_curve is not None else -3, 1)
    #increase the tick size
    ax.tick_params(axis='both', which='major', labelsize=16)

    return plot_fit_results

def calculate_gaussian_peak(data_norm, time_arr_binned, wavelengths, time_lim, Time, time_arr, time_lim_index):
    """
    Calculate the Gaussian peak position for each time slice.

    Args:
        data_norm (ndarray): Normalized data (wavelengths x time).
        time_arr_binned (ndarray): Binned time array.
        wavelengths (ndarray): Array of wavelengths.
        time_lim (float): Maximum time limit.
        Time (ndarray): Original time array.
        time_arr (ndarray): Non-binned time array.
        time_lim_index (int): Index of the maximum time limit.

    Returns:
        ndarray: Gaussian peak positions for each time slice.
    """
    gaussian_peaks = []  # List to store the Gaussian peak positions

    for t_idx in range(data_norm.shape[1]):  # Iterate over time slices
        pl_intensity = data_norm[:, t_idx]  # PL intensity for the current time slice

        # Perform Gaussian fitting
        try:
            p0 = [np.max(pl_intensity), wavelengths[np.argmax(pl_intensity)], 10, np.min(pl_intensity)]
            popt, _ = curve_fit(gauss, wavelengths, pl_intensity, p0=p0)
            gaussian_peaks.append(popt[1])  # Save the peak position (mu)
        except RuntimeError:
            gaussian_peaks.append(np.nan)  # If the fit fails, save NaN

    gaussian_peaks = np.array(gaussian_peaks)

    return gaussian_peaks

def process_all_files_and_plot(folder_path, time_lim=20, binning=1):

    all_measurements = load_data_and_metadata(folder_path)
    gaussian_peaks_per_code = {}

    for measurement in all_measurements:
        file_name = os.path.basename(measurement['data_file'])
        match = re.search(r'_C(\d+)_', file_name)
        
        code = f"c{match.group(1)}"

        results_dict = measurement['data']
        TimeResolution, AcquisitionTime, laser_pw, laser_divider, spotsize, steps, center_w, wavelength_range_start, wavelength_range_end, objective, MaxRepRate, sample = extract_parameters_from_metadata([measurement['metadata_file']], folder_path)

        Time = results_dict['time']
        wavelengths = results_dict['wavelength']
        data = results_dict['data']

        time_lim_index = int(time_lim / TimeResolution * 1e03)
        NPulsesSeen = pulses_from_rep_rate(MaxRepRate / laser_divider)
        time_windw, time_arr = param_analysis(laser_divider, MaxRepRate, TimeResolution)

        _, _, _, _, time_arr_binned, data_norm = lifetimes_data_clean(
            Time, wavelengths, data, time_arr, time_windw,
            MaxRepRate, laser_divider, laser_pw,
            binning, time_lim_index, NPulsesSeen
        )

        gaussian_peaks = calculate_gaussian_peak(data_norm, time_arr_binned, wavelengths, time_lim, Time, time_arr, time_lim_index)

        sigma_min = 0.5
        sigma_max = 1.5
        gaussian_peaks_smoothed = weighted_smoothing(gaussian_peaks, time_arr_binned, sigma_min, sigma_max)

        median_arr = np.median(data_norm, axis=0)

        gaussian_peaks_per_code[code] = {
            'wavelengths': wavelengths,
            'peaks_raw': gaussian_peaks,
            'peaks_smoothed': gaussian_peaks_smoothed,
            'time': time_arr_binned,
            'data_norm': data_norm,
            'median': median_arr
        }

    return gaussian_peaks_per_code


def plot_full_figure(
    gaussian_peaks_per_code,
    gaussian_peaks_dict_no_binned,
    fit_results,
    popt0,
    popt1,
    x,
    color_map,
    labels,
    time_arr,
    time_lim=20
):
    fig = plt.figure(figsize=(12, 14)) 
    gs = GridSpec(3, 2, width_ratios=[1, 1.5], height_ratios=[1, 1, 1], wspace=0.25, hspace=0.25)
    axes_left = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax_fit = fig.add_subplot(gs[0:2, 1])  
    gs_sub = gs[2, 1].subgridspec(1, 2, wspace=0.5)
    ax_e0 = fig.add_subplot(gs_sub[0, 0])
    ax_k0 = fig.add_subplot(gs_sub[0, 1])

    def format_code(code):
        return 'C' + code[1:] if code.lower().startswith('c') else code

    # --- Plot left subplots with formatted key labels ---
    for i, (code, ax_left) in enumerate(zip(gaussian_peaks_per_code.keys(), axes_left)):
        data_dict = gaussian_peaks_per_code[code]
        time_arr_binned = data_dict['time']
        gaussian_peaks_smoothed = data_dict['peaks_smoothed']
        data_norm = data_dict['data_norm']
        wavelengths = data_dict['wavelengths']

        im = ax_left.imshow(
            data_norm, cmap='magma',
            extent=[time_arr[0], time_lim, wavelengths[0], wavelengths[-1]],
            origin='upper', aspect=0.3
        )
        ax_left.set_xlim(0, time_lim)
        ymin, ymax = wavelengths[0], wavelengths[-1]
        median_inverted = ymax + ymin - gaussian_peaks_smoothed
        ax_left.plot(time_arr_binned, median_inverted, 'w-', lw=2)
        ax_left.text(time_lim-4, 490, format_code(code), fontsize=14, color='white', va='top')
        ax_left.tick_params(axis='both', which='major', labelsize=16)
        yticks_nm = ax_left.get_yticks()
        yticks_eV = 1240.0 / yticks_nm
        #ax_left.invert_yaxis()
        ax_left.set_yticklabels([f"{e:.1f}" for e in yticks_eV])
        if i == 0:
            first_im = im  # save handle of the first map

    # --- Colorbar positioning ---
    pos = axes_left[0].get_position()
    cbar_ax = fig.add_axes([pos.x0-0.025, pos.y1 + 0.07, pos.width, 0.02])
    cbar = fig.colorbar(
        first_im,
        cax=cbar_ax,
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.set_label('Intensity (-)', fontsize=16, labelpad=10)
    cbar.ax.tick_params(labelsize=18)

    axes_left[-1].set_xlabel('Time (ns)', fontsize=16)
    fig.text(0.03, 0.5, 'Energy (eV)', va='center', rotation='vertical', fontsize=16)

    # --- Plot fit on ax_fit as usual ---
    plot_fit_dict = plot_all_smoothed_peaks_with_fit(
        gaussian_peaks_dict_no_binned,
        fit_results,
        fit_time_range=(0, 10),
        plot_time_range=(0, 20),
        ax=ax_fit
    )

    ax_fit.set_autoscale_on(False)

    # --- Update legend on ax_fit with formatted labels ---
    handles, labels_ = ax_fit.get_legend_handles_labels()
    labels_formatted = [format_code(lbl) for lbl in labels_]
    ax_fit.legend(handles, labels_formatted, frameon=False, fontsize=16)

    # --- Plot e0 on ax_e0 as usual ---
    for j, label in enumerate(labels):
        e0 = popt0[j]
        e0_error = fit_results[label]['param_errors'][0]
        ax_e0.errorbar(
            x[j],e0,  yerr=e0_error, fmt='o',
            color=color_map[label],
            markerfacecolor=color_map[label],
            markeredgewidth=2,
            markersize=10, label=label,
            alpha=0.6, capsize=2, elinewidth=1
        )

    ax_e0.grid(axis='y', color='gray', linestyle='--', linewidth=0.7)
    ax_e0.set_xticks([8,12,16])
    ax_e0.set_xlim(5, 19)
    ax_e0.set_ylim(-24,-14)
    ax_e0.set_xlabel('d$_{C-C}$ (nm)', fontsize=16)
    ax_e0.set_ylabel('$\Delta$E$_{\infty}$ (meV)', fontsize=16)
    ax_e0.tick_params(axis='both', labelsize=16)

    # --- Plot per k ---
    for j, label in enumerate(labels):
        k0 = popt1[j]
        k0_error = fit_results[label]['param_errors'][1]
        ax_k0.errorbar(
            x[j], k0,yerr=k0_error,
            fmt='o', color=color_map[label],
            markerfacecolor=color_map[label],
            markeredgewidth=2,
            markersize=10, label=label,
            alpha=0.6, capsize=2, elinewidth=1)

    ax_k0.grid(axis='y', color='gray', linestyle='--', linewidth=0.7)
    ax_k0.set_xticks([8,12,16])
    ax_k0.set_xlim(5, 19)
    ax_k0.set_ylim(0.15, 0.35)
    ax_k0.set_xlabel('d$_{C-C}$ (nm)', fontsize=16)
    ax_k0.set_ylabel('k (ns$^{-1}$)', fontsize=16)
    ax_k0.tick_params(axis='both', labelsize=16)

    fig.subplots_adjust(left=0.08, right=0.88, top=0.94, bottom=0.12)
    for ax in fig.axes:
        ax.set_autoscale_on(False)

    return fig


#%%

def main():
        
    #load the data

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data\\spectrally_resolved_LF_solid')
    #folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '/Users/antonellacutrupi/Desktop/Simon_SP_LF/all')

    # Load all matched data and metadata
    measurements = load_data_and_metadata(folder)

    # Sort and display available measurements
    print("\nAvailable measurements:")
    for i, meas in enumerate(measurements):
        print(f"[{i}] File: {os.path.basename(meas['metadata_file'])}")

    gaussian_peaks_per_code = process_all_files_and_plot(folder, time_lim=20, binning=5)

    sorted_codes = sorted(gaussian_peaks_per_code.keys(), key=lambda x: int(x[1:]))

    gaussian_peaks_per_code = {key: gaussian_peaks_per_code[key] for key in sorted_codes}


    gaussian_peaks_dict_no_binned = process_all_files_and_plot(folder, time_lim=20, binning=1)

    TimeResolution, AcquisitionTime, laserPower, laserDivider, spotsize, steps, center_w, wavelength_range_start, wavelength_range_end, objective, maxRepRate, sample = extract_parameters_from_metadata(meas['metadata_file'], folder)

    time_windw, time_arr = param_analysis(laserDivider, maxRepRate, TimeResolution)
    # Fit Gaussian peaks for all codes
    fit_results = fit_energy_decay_per_code(gaussian_peaks_dict_no_binned)

    labels = list(fit_results.keys())
    labels.sort(key=lambda x: int(x[1:])) 
    popt0 = [fit_results[k]['params'][0] for k in labels]
    popt1 = [fit_results[k]['params'][1] for k in labels]

    c_to_c_distance = np.array([16.2,12.5,9.5])
 
    custom_colors = ['red', 'green', 'blue']
    n_colors = len(custom_colors)  # Number of custom colors
    color_map = {label: custom_colors[i % n_colors] for i, label in enumerate(labels)}  # Associa un colore a ogni codice

    fig = plot_full_figure ( gaussian_peaks_per_code,
        gaussian_peaks_dict_no_binned,
        fit_results,
        popt0,
        popt1,
        c_to_c_distance,
        color_map,
        labels,
        time_arr,
        time_lim=20)
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
    
    plt.show()  # Show the plots interactively


# %%
