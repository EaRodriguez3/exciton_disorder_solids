

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

def bin_data(data, bin_size):
    num_bins = len(data) // bin_size 
    return np.array([np.mean(data[i * bin_size:(i + 1) * bin_size]) for i in range(num_bins)])

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

def single_lifetime_trace_full(dat, NPulsesSeen):
    # average over the pulses seen
    data_av = []
    segment_len = int(len(dat) / NPulsesSeen)
    
    for i in range(NPulsesSeen):
        start_idx = i * segment_len
        end_idx = start_idx + segment_len
        segment = dat[start_idx:end_idx]
        data_av.append(segment)

    data_av = np.array(data_av)
    data_onePulse = np.mean(data_av, axis=0)

    return data_onePulse


def plot_lifetime_histograms(measurements, folder_liquid):
    
    peak_wavelengths = {
        'c8': 525,
        'c12': 520,
        'c16': 518
    }

    color_dict = {
        'c8': 'red',
        'c12': 'green',
        'c16': 'blue'
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)

    for measurement in measurements:
        file_name = os.path.basename(measurement['data_file'])
        match = re.search(r'_C(\d+)_', file_name)
        if not match:
            continue

        code = f"c{match.group(1)}"
        if code not in peak_wavelengths:
            continue

        color = color_dict.get(code, 'black')

        if code == 'c8':
            ax = axes[0]
        elif code == 'c12':
            ax = axes[1]
        elif code == 'c16':
            ax = axes[2]

        results_dict = measurement['data']
        Time = results_dict['time']
        wavelengths = results_dict['wavelength']
        data = results_dict['data']

        (TimeResolution, AcquisitionTime, laser_pw, laser_divider,
         spotsize, steps, center_w, wavelength_range_start,
         wavelength_range_end, objective, MaxRepRate, sample) = \
            extract_parameters_from_metadata(
                [measurement['metadata_file']], folder_liquid
            )

        NPulsesSeen = pulses_from_rep_rate(MaxRepRate / laser_divider)

        peak_wl = peak_wavelengths[code]

        ranges = {
            "-20 nm": (peak_wl - 20, peak_wl),
            "+20 nm": (peak_wl, peak_wl + 20)
        }

        alpha_dict = {
            "-20 nm": 1.0,
            "+20 nm": 0.4
        }

        for label_range, (wl_min, wl_max) in ranges.items():

            mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
            indices_wl = np.where(mask)[0]
            if len(indices_wl) == 0:
                continue

            pulse_sum = None

            for idx_wl in indices_wl:
                pulse_avg = single_lifetime_trace_full(
                    data[idx_wl, :],
                    NPulsesSeen
                )
                pulse_sum = pulse_avg if pulse_sum is None else pulse_sum + pulse_avg

            pulse_avg_combined = pulse_sum / len(indices_wl)

            bin_size = 7
            pulse_binned = bin_data(pulse_avg_combined, bin_size)

            time_arr = np.arange(len(Time)) * TimeResolution
            time_arr = bin_data(time_arr, bin_size)

            bg = np.mean(pulse_binned[:200])
            pulse_binned -= bg
            pulse_binned /= np.max(pulse_binned)

            if label_range == "-20 nm":
                label = r"$\lambda_{\mathrm{peak}} - \delta$"
            else:
                label = r"$\lambda_{\mathrm{peak}} + \delta$"

            ax.plot(
                time_arr * 1e-3,
                pulse_binned[:len(time_arr)],
                marker='.',
                lw=2,
                color=color,
                alpha=alpha_dict[label_range],
                label=label
            )

        ax.text(
            0.08, 0.95, code,
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment='top'
        )

        ax.set_xlabel('Time (ns)', fontsize=24)
        ax.set_xlim(0, 100)
        ax.legend()

        if code == 'c16':
            ax.set_ylim(0.5e-1, 1.1)
        else:
            ax.set_ylim(-0.01, 1.1)

    axes[0].set_ylabel('Counts (a.u.)', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()

#%%

def main():
        
    #load the data

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data\\spectrally_resolved_LF_liquid')

    # Load all matched data and metadata
    measurements = load_data_and_metadata(folder)

    # Sort and display available measurements
    print("\nAvailable measurements:")
    for i, meas in enumerate(measurements):
        print(f"[{i}] File: {os.path.basename(meas['metadata_file'])}")

    # Plot lifetime histograms
    plot_lifetime_histograms(measurements, folder)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
    plt.show()  # Show the plots interactively


# %%
