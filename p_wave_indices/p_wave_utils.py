import os
import numpy as np
from scipy.io import loadmat
import neurokit2 as nk
import pandas as pd

# -----------------------------------------
# Utility: Extract P-wave indices from ECG
# -----------------------------------------
def extract_pwave_indices(ecg_signal, sampling_rate=500):
    """
    Extracts P wave indices from a 12-lead ECG.
    Parameters:
        ecg_signal (np.ndarray): Shape (n_samples, 12)
        sampling_rate (int): Sampling rate of ECG in Hz
    Returns:
        dict: {"P_amp_leadII": float, "P_duration": float,
               "PR_interval": float, "P_axis": float}
    """
    

    # Take Lead II for P-wave amplitude
    lead_I = ecg_signal[0,:]
    lead_II = ecg_signal[1, :] 
    # Run ECG delineation on Lead II
    try:
        _, info = nk.ecg_process(lead_I, sampling_rate=sampling_rate)
    except:
        return {"P_amp_leadII": None,
                            "P_duration": None,
                            "PR_interval": None,
                            "P_axis": None}
    
    r_onsets = info["ECG_R_Onsets"]
    p_onsets = info["ECG_P_Onsets"]
    p_offsets = info["ECG_P_Offsets"]
    p_peaks = info["ECG_P_Peaks"]
    
    p_times = {"p_onsets": p_onsets, "p_peaks": p_peaks, "p_offsets": p_offsets, "r_onsets": r_onsets }
    p_times = pd.DataFrame(p_times)
    p_times = p_times.dropna(axis = 0)
    p_times = p_times.astype(int)


    if (p_times.count(axis = 0) < 2).any():
        return {"P_amp_leadII": None,
                            "P_duration": None,
                            "PR_interval": None,
                            "P_axis": None}


    # --- P Wave Amplitude (Lead II) ---
    p_max = np.array(lead_II[p_times["p_peaks"]])
    p_base = np.array(lead_II[p_times["p_onsets"]])
    p_amp = p_max - p_base

    
    P_amp_leadII = float(np.mean(p_amp))
    
    # --- P Wave Duration ---
    
    durations = (p_times["p_offsets"] - p_times["p_onsets"]) / sampling_rate * 1000  # ms
    P_duration = float(np.nanmean(durations))


    # --- PR Interval ---
    r_onsets = info["ECG_R_Onsets"]
    pr_intervals = (p_times["r_onsets"] - p_times["p_onsets"]) / sampling_rate * 1000  # ms
    PR_interval = float(np.nanmean(pr_intervals))


    # --- P Axis ---
    # Compute mean vector of P waves across leads I and aVF
    lead_aVF = ecg_signal[5, :]  # Assuming column 5 = aVF
    p_max_I = np.array(lead_I[p_times["p_peaks"]])
    p_base_I = np.array(lead_I[p_times["p_onsets"]])
    p_amp_I = p_max_I - p_base_I
    
    p_max_aVF = np.array(lead_aVF[p_times["p_peaks"]])
    p_base_aVF = np.array(lead_aVF[p_times["p_onsets"]])
    p_amp_aVF = p_max_aVF - p_base_aVF
    
    mean_P_I = np.mean(p_amp_I)
    mean_P_aVF = np.mean(p_amp_aVF)
    P_axis = np.degrees(np.arctan2(mean_P_aVF, mean_P_I))
   

    
    return {
        "P_amp_leadII": P_amp_leadII,
        "P_duration": P_duration,
        "PR_interval": PR_interval,
        "P_axis": P_axis,
    }


# -----------------------------------------
# Loop through CODE15% dataset folder
# -----------------------------------------
def process_dataset(ecg_directory, sampling_rate=500):
    total_num_ecgs = len(os.listdir(ecg_directory))
    results = []
    for i, file in enumerate(os.listdir(ecg_directory)):
        if i%1000 == 100:
            print("Working on ECG " + str(i) + " of " + str(total_num_ecgs))
            break
        if file.endswith(".mat"):
            mat_data = loadmat(os.path.join(ecg_directory, file))
            ecg = mat_data["feats"]  # (n_samples, 12)
            indices = extract_pwave_indices(ecg, sampling_rate)
            indices["filename"] = file
            results.append(indices)

    return results


# Example usage
# ecg_dir = "/path/to/CODE15_folder"
# features = process_code15_dataset(ecg_dir)
# print(features[:5])
