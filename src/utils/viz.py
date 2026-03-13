import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from utils.dir_file import DirFileManager

from configs.constants import PTB_ORDER

def plot_ecg(ecg, leads = PTB_ORDER, sf = 250, file_name = None, plot_title = None, save_dir = None):
    n_leads, T = ecg.shape
    t = np.arange(T) / sf

    fig, axes = plt.subplots(n_leads, 1, figsize=(12, n_leads * 0.8), sharex = True)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        ax.plot(t, ecg[i], color = 'k', linewidth = 0.5)
        ax.set_ylabel(leads[i], fontsize=8, rotation=0, 
                      ha = "right", va = "center")
        # ax.set_ylim([0, 1])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
    
    axes[-1].set_xlabel("Time (s)")
    if plot_title:
        fig.suptitle(plot_title, fontsize=12)
    plt.tight_layout()
    DirFileManager.ensure_directory_exists(folder = f"{save_dir}/pngs")
    plt.savefig(f"{save_dir}/pngs/{file_name}.png", dpi = 150, bbox_inches = "tight")
    plt.close()

def plot_forecast(full_gt, full_pred, n_ctx_flat, n_gt_end, n_pred_end,
                  report, save_path, segment_len=2500, leads=PTB_ORDER, sf=250, ctx_per_lead=None):
    n_leads = full_gt.shape[0]
    t = np.arange(segment_len) / sf

    fig, axes = plt.subplots(n_leads, 1, figsize=(20, max(n_leads * 1.2, 3)), sharex=True)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        lead_start = i * segment_len
        bnd = ctx_per_lead if ctx_per_lead is not None else np.clip(n_ctx_flat - lead_start, 0, segment_len)
        gt_end = np.clip(n_gt_end - lead_start, 0, segment_len)
        pred_end = np.clip(n_pred_end - lead_start, 0, segment_len)
        pad_start = min(gt_end, pred_end)
        if pad_start < segment_len:
            ax.axvspan(t[pad_start], t[-1], color="lavender", alpha=0.5)
        if bnd > 0:
            ax.plot(t[:bnd], full_gt[i, :bnd], color="black", linewidth=1.0)
        if bnd < segment_len:
            ax.plot(t[bnd:gt_end], full_gt[i, bnd:gt_end], color="tab:blue", linewidth=1.0)
            ax.plot(t[bnd:pred_end], full_pred[i, bnd:pred_end], color="tab:red", linewidth=1.0)
        if 0 < bnd < segment_len:
            ax.axvline(t[bnd], color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel(leads[i], fontsize=8, rotation=0, ha="right", va="center")
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    handles = [
        mlines.Line2D([], [], color="black", linewidth=1.0, label="Context"),
        mlines.Line2D([], [], color="tab:blue", linewidth=1.0, label="Ground Truth"),
        mlines.Line2D([], [], color="tab:red", linewidth=1.0, label="Prediction"),
        mpatches.Patch(color="lavender", alpha=0.5, label="Padding"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(report, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()