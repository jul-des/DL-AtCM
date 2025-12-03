import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d


def compute_vanilla_gradient_saliency(model, input_data, target_output_idx=0, unfreeze_for_saliency=True):

    # Set model to evaluation mode but enable gradient computation
    model.eval()
    
    # Store original backbone state
    was_frozen = getattr(model, 'backbone_frozen', False)
    
    # Temporarily unfreeze backbone if requested
    if unfreeze_for_saliency and hasattr(model, 'unfreeze_backbone'):
        if was_frozen:
            model.unfreeze_backbone()
    
    try:
        # Get the device of the model
        device = next(model.parameters()).device
        
        # Ensure input is on the same device as model and requires gradients
        input_data = input_data.clone().detach().to(device)
        input_data.requires_grad_(True)
        
        # Clear any existing gradients
        if input_data.grad is not None:
            input_data.grad.zero_()
        
        # Enable gradient computation explicitly
        with torch.enable_grad():
            # Forward pass
            outputs = model(input_data)
            
            # Get the target output (specific regression head)
            if target_output_idx >= outputs.shape[1]:
                raise ValueError(f"target_output_idx {target_output_idx} is out of range for outputs with shape {outputs.shape}")
            
            target_output = outputs[:, target_output_idx]
            
            # For regression, we typically want gradients w.r.t. the sum of outputs
            target_scalar = target_output.sum()
            
            # Backward pass
            target_scalar.backward(retain_graph=True)
        
        # Check if gradients were computed
        if input_data.grad is None:
            raise RuntimeError("Gradients were not computed. Check if model parameters require gradients.")
        
        # Get gradients w.r.t. input
        saliency_map = input_data.grad.clone()
        
    finally:
        # Restore original backbone state
        if unfreeze_for_saliency and hasattr(model, 'freeze_backbone') and was_frozen:
            model.freeze_backbone()
    
    return saliency_map


def visualize_ecg_saliency(
        original_ecg,
        saliency_map,
        lead_names=None,
        sample_idx: int = 0,
        save_path: str | None = None,
        title: str = "ECG Saliency Map",
        # ---- NEW smoothing parameters --------------------------------------
        smooth: bool = True,
        kernel: str = "gaussian",        # 'gaussian' or 'moving_average'
        kernel_width: int = 51,          # points; should be odd
        gaussian_sigma: float = 7.0,     # only for gaussian
):
    """
    Visualize an ECG with an optional smoothed saliency map overlay.

    Args
    ----
    original_ecg : Tensor (B, 12, 2500)
    saliency_map : Tensor (B, 12, 2500)
    lead_names   : list[str] of ECG lead labels
    sample_idx   : which batch element to show
    save_path    : optional output file
    title        : plot title
    smooth       : whether to smooth saliency values
    kernel       : 'gaussian' | 'moving_average'
    kernel_width : window length (odd integer)
    gaussian_sigma : std‑dev for gaussian kernel (ignored if kernel != 'gaussian')
    """

    if lead_names is None:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Convert to NumPy
    ecg_data      = original_ecg[sample_idx].cpu().numpy()   # (12, 2500)
    saliency_data = saliency_map[sample_idx].cpu().numpy()   # (12, 2500)

    # ---------------- Smoothing helper -------------------------------------
    def smooth_series(x: np.ndarray) -> np.ndarray:
        """Return a smoothed 1‑D array the same length as x."""
        if not smooth:
            return x

        if kernel == "gaussian":
            return gaussian_filter1d(x, sigma=gaussian_sigma, mode="reflect")
        elif kernel == "moving_average":
            # Simple causal MA; pad symmetrically to keep length unchanged
            k = kernel_width | 1          # ensure odd
            pad = k // 2
            window = np.ones(k) / k
            padded = np.pad(x, pad, mode="reflect")
            return np.convolve(padded, window, mode="valid")
        else:
            raise ValueError(f"Unknown kernel '{kernel}'")

    # Pre‑smooth saliency
    saliency_smoothed = np.vstack([smooth_series(lead) for lead in saliency_data])

    # ----------------- Plot -------------------------------------------------
    time_axis = np.linspace(0, 5, 2500)
    fig, axes = plt.subplots(12, 1, figsize=(15, 20), sharex=True)

    for i in range(12):
        ax = axes[i]

        # ECG trace (light gray so saliency stands out)
        ax.plot(time_axis, ecg_data[i], linewidth=1, color="lightgray", label="ECG")

        # Prepare saliency colouring
        saliency_abs  = np.abs(saliency_smoothed[i])
        saliency_norm = (saliency_abs - saliency_abs.min()) / (np.ptp(saliency_abs) + 1e-8)

        scatter = ax.scatter(time_axis, ecg_data[i],
                             c=saliency_norm,
                             cmap='Reds',
                             s=1, alpha=0.9,
                             label='Saliency')

        ax.set_title(f'Lead {lead_names[i]}', loc='left')
        ax.grid(True, alpha=0.3)
        if i == 11:
            ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amp.')

        # Show global colour bar only once
        #if i == 0:
            #plt.colorbar(scatter, ax=ax, label='Saliency magnitude')

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # make room for suptitle

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



def detect_cardiac_cycles(P_on, P_off, Q, S, T_on, T_off):
    """
    Return a list of dicts, each holding the six markers that make up
    one complete cardiac cycle in the correct temporal order.
    """
    cycles = []
    i = 0                       # index into P_on

    while i < len(P_on):
        cycle = {}

        # -------- P_on ----------------------------------------------------
        cycle["P_on"] = start = P_on[i]

        # -------- P_off ---------------------------------------------------
        j = np.searchsorted(P_off, start, side="left")
        if j == len(P_off):
            break
        cycle["P_off"] = p_off = P_off[j]

        # -------- Q -------------------------------------------------------
        k = np.searchsorted(Q, p_off, side="left")
        if k == len(Q):
            break
        cycle["Q"] = q = Q[k]

        # -------- S -------------------------------------------------------
        l = np.searchsorted(S, q, side="left")
        if l == len(S):
            break
        cycle["S"] = s = S[l]

        # -------- T_on ----------------------------------------------------
        m = np.searchsorted(T_on, s, side="left")
        if m == len(T_on):
            break
        cycle["T_on"] = t_on = T_on[m]

        # -------- T_off ---------------------------------------------------
        n = np.searchsorted(T_off, t_on, side="left")
        if n == len(T_off):
            break
        cycle["T_off"] = t_off = T_off[n]

        c = cycle
        pr  = c["Q"]    - c["P_on"]     # PR interval
        qrs = c["S"]    - c["Q"]        # QRS duration
        qt  = c["T_off"]- c["Q"]        # QT interval

        if 50 <= 2*pr <= 300 and 2*qrs < 250 and 2*qt < 750:   # multiply by 2 to convert to ms
            cycles.append(cycle)

        # Advance `i` to the first P_on that occurs *after* this T_off
        new_i = np.searchsorted(P_on, t_off, side="left")

        if new_i <= i:
            i += 1       # skip the current P_on and move on
        else:
            i = new_i

    return cycles


def average_saliency_per_segment(P_on, P_off, Q, S, T_on, T_off, saliency):
    total_saliency = {"P_wave" : [], "PQ" : [], "QRS" : [], "ST":[], "T_wave" : [], "TP": [], "Total_saliency": []}
    cycles = detect_cardiac_cycles(P_on, P_off, Q, S, T_on, T_off)
    num_cycles = len(cycles)
    for i, cycle in enumerate(cycles):
        total_saliency["P_wave"].append(saliency[cycle["P_on"]:cycle["P_off"]])
        total_saliency["PQ"].append(saliency[cycle["P_off"]:cycle["Q"]])
        total_saliency["QRS"].append(saliency[cycle["Q"]:cycle["S"]])
        total_saliency["ST"].append(saliency[cycle["S"]:cycle["T_on"]])
        total_saliency["T_wave"].append(saliency[cycle["T_on"]:cycle["T_off"]])
        try:
            total_saliency["TP"].append(saliency[cycle["T_off"]:cycles[i+1]["P_on"]])
        except:
            pass
    total_saliency["Total_saliency"] = list(saliency)
    for key in total_saliency.keys():
        if key == "Total_saliency":
            total_saliency[key] =  np.mean(np.abs(total_saliency[key]))
        else:
            total_saliency[key] = list(np.concatenate(total_saliency[key]))
            total_saliency[key] =  np.mean(np.abs(total_saliency[key]))
    return total_saliency, num_cycles
