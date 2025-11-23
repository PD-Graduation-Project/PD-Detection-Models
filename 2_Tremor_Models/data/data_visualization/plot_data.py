import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def save_movement_plots_to_pdf(movement_dataloaders,
                               pdf_path="signals_per_movement.pdf",
                               max_plots_per_movement=3):
    """
    Generate multi-page PDF of signal plots for each movement.
    """
    with PdfPages(pdf_path) as pdf:
        for movement_name, splits in movement_dataloaders.items():

            if len(splits["train"]) > 0:
                dl = splits["train"]
            elif len(splits["val"]) > 0:
                dl = splits["val"]
            else:
                print(f"[WARN] No data for movement: {movement_name}")
                continue

            # Fix: Handle extra return values (metadata etc) using *rest
            batch = next(iter(dl))
            signal_batch, handedness_batch, movement_batch, label_batch, *rest = batch

            B, _, T, C = signal_batch.shape
            num_plots = min(B, max_plots_per_movement)

            print(f"[INFO] Saving {num_plots} samples for movement: {movement_name}")

            for i in range(num_plots):
                signal = signal_batch[i]        
                handedness = handedness_batch[i].item()
                label = label_batch[i].item()

                fig, axs = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
                
                # Standard Title for this function
                fig.suptitle(
                    f"Movement: {movement_name} | Sample {i}\n"
                    f"Handedness: {'Left' if handedness == 0 else 'Right'} | Label: {label}",
                    fontsize=14
                )

                for ch in range(C):
                    axs[ch].plot(signal[0, :, ch].numpy(), label="Left wrist")
                    axs[ch].plot(signal[1, :, ch].numpy(), label="Right wrist", linestyle="--")
                    axs[ch].set_ylabel(f"Ch {ch}")
                    axs[ch].grid(True)
                    if ch == 0:
                        axs[ch].legend()

                axs[-1].set_xlabel("Time (samples)")
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        print(f"\n[INFO] Saved all movement plots to: {pdf_path}\n")


# =================================================================================== #
def plot_movement_comparison(movement_dataloaders,
                             pdf_path="movement_comparison.pdf",
                             max_samples_per_class=None):
    """
    Standard comparison (Averages wrists together).
    """
    c_healthy = 'teal'
    c_pd = 'crimson'

    with PdfPages(pdf_path) as pdf:
        for movement_name, splits in movement_dataloaders.items():
            print(f"[INFO] Processing movement: {movement_name}")

            healthy_samples = []
            pd_samples = []

            for split_name in ["train", "val"]:
                dl = splits[split_name]
                for batch in dl:
                    signals, handedness, movement, labels, *rest = batch # Fixed unpacking

                    mask_healthy = labels == 0
                    mask_pd = labels == 1

                    if mask_healthy.any(): healthy_samples.append(signals[mask_healthy])
                    if mask_pd.any(): pd_samples.append(signals[mask_pd])

            if len(healthy_samples) == 0 or len(pd_samples) == 0:
                print(f"[WARN] Movement {movement_name} missing healthy or PD samples.")
                continue

            healthy = torch.cat(healthy_samples, dim=0)
            pd = torch.cat(pd_samples, dim=0)

            if max_samples_per_class:
                healthy = healthy[:max_samples_per_class]
                pd = pd[:max_samples_per_class]

            healthy = healthy.mean(dim=1) # Average wrists
            pd = pd.mean(dim=1)

            healthy_mean = healthy.mean(dim=0)
            healthy_std = healthy.std(dim=0)
            pd_mean = pd.mean(dim=0)
            pd_std = pd.std(dim=0)

            T, channels = healthy_mean.shape
            x = torch.arange(T)

            fig, axs = plt.subplots(6, 1, figsize=(13, 14), sharex=True)
            
            # --- FIXED TITLE SECTION ---
            # 1. Movement Name in BLUE
            fig.text(0.5, 0.96, f"Movement: {movement_name}", 
                     color='#1A3D64', ha='center', fontsize=20, weight='bold')

            # 2. Description in BLACK
            fig.text(0.5, 0.94, "Healthy vs PD (mean ± std) | Combined Wrists", 
                     color='black', ha='center', fontsize=14)
            # ---------------------------

            for ch in range(channels):
                ax = axs[ch]
                
                # Healthy
                ax.fill_between(x, healthy_mean[:, ch]-healthy_std[:, ch], healthy_mean[:, ch]+healthy_std[:, ch],
                                alpha=0.25, color=c_healthy, label="Healthy" if ch == 0 else None)
                ax.plot(x, healthy_mean[:, ch], label="Healthy Mean" if ch == 0 else None, color=c_healthy)

                # PD
                ax.fill_between(x, pd_mean[:, ch]-pd_std[:, ch], pd_mean[:, ch]+pd_std[:, ch],
                                alpha=0.25, color=c_pd, label="PD" if ch == 0 else None)
                ax.plot(x, pd_mean[:, ch], linestyle="--", label="PD Mean" if ch == 0 else None, color=c_pd)
                
                ax.set_ylabel(f"Ch {ch}")
                ax.grid(True)
                
                # --- FIXED LEGEND POSITION ---
                if ch == 0: 
                    ax.legend(loc="upper right")
                # -----------------------------

            axs[-1].set_xlabel("Time (samples)")
            
            # --- FIXED LAYOUT OVERLAP ---
            # Reserves space at top for the custom titles
            plt.tight_layout(rect=[0, 0.00, 1, 0.91])
            
            pdf.savefig(fig)
            plt.close(fig)

        print(f"\n[INFO] Saved PDF: {pdf_path}\n")


# =================================================================================== #
# =================================================================================== #
# =================================================================================== #

def plot_movement_comparison_by_dominance(movement_dataloaders,
                                          pdf_path="movement_comparison_dominance.pdf",
                                          max_samples_per_class=None):
    """
    Plots Dominant vs Non-Dominant with Custom Title Colors:
    - Movement Name: Blue
    - Dominant: Red
    - Non-Dominant: Green
    - General Title: Black
    """
    
    # --- PLOT LINE COLORS ---
    c_healthy = 'teal' 
    c_pd = 'crimson'      
    # ------------------------

    with PdfPages(pdf_path) as pdf:

        for movement_name, splits in movement_dataloaders.items():
            print(f"[INFO] Processing movement: {movement_name}")

            h_dom_list, h_nondom_list = [], []
            pd_dom_list, pd_nondom_list = [], []

            for split_name in ["train", "val"]:
                dl = splits[split_name]
                for batch in dl:
                    # FIX: *rest handles extra return values safely
                    signals, handedness, movement, labels, *rest = batch
                    
                    B = signals.shape[0]
                    batch_indices = torch.arange(B)
                    dom_indices = handedness.long()
                    nondom_indices = 1 - dom_indices

                    sig_dom = signals[batch_indices, dom_indices]
                    sig_nondom = signals[batch_indices, nondom_indices]

                    mask_healthy = labels == 0
                    mask_pd = labels == 1

                    if mask_healthy.any():
                        h_dom_list.append(sig_dom[mask_healthy])
                        h_nondom_list.append(sig_nondom[mask_healthy])

                    if mask_pd.any():
                        pd_dom_list.append(sig_dom[mask_pd])
                        pd_nondom_list.append(sig_nondom[mask_pd])

            if not h_dom_list or not pd_dom_list:
                print(f"[WARN] Movement {movement_name} missing data. Skipping.")
                continue

            # -------- Inner Function to Plot One Side --------
            def process_and_plot(h_list, p_list, side_name):
                healthy = torch.cat(h_list, dim=0)
                pd = torch.cat(p_list, dim=0)

                if max_samples_per_class:
                    healthy = healthy[:max_samples_per_class]
                    pd = pd[:max_samples_per_class]

                h_mean = healthy.mean(dim=0)
                h_std = healthy.std(dim=0)
                p_mean = pd.mean(dim=0)
                p_std = pd.std(dim=0)

                T, channels = h_mean.shape
                x = torch.arange(T)
                
                fig, axs = plt.subplots(6, 1, figsize=(13, 14), sharex=True)

                # =========================================================
                # CUSTOM MULTI-COLOR TITLE IMPLEMENTATION
                # =========================================================
                
                # 1. Movement Name in BLUE
                fig.text(0.5, 0.96, f"Movement: {movement_name}", 
                         color='#1A3D64', ha='center', fontsize=20, weight='bold')

                # 2. Dominant (RED) or Non-Dominant (GREEN)
                if side_name == "Dominant":
                    side_color = 'red'
                else:
                    side_color = 'green'
                    
                fig.text(0.5, 0.94, f"{side_name} Hand", 
                         color=side_color, ha='center', fontsize=16, weight='bold')

                # 3. Description in BLACK
                fig.text(0.5, 0.915, "Healthy vs PD (mean ± std)", 
                         color='black', ha='center', fontsize=14)

                # =========================================================

                for ch in range(channels):
                    ax = axs[ch]
                    
                    # Healthy
                    ax.fill_between(x, h_mean[:, ch] - h_std[:, ch], h_mean[:, ch] + h_std[:, ch],
                                    alpha=0.25, color=c_healthy, label="Healthy" if ch==0 else None)
                    ax.plot(x, h_mean[:, ch], color=c_healthy, label="Healthy Mean" if ch==0 else None)

                    # PD
                    ax.fill_between(x, p_mean[:, ch] - p_std[:, ch], p_mean[:, ch] + p_std[:, ch],
                                    alpha=0.25, color=c_pd, label="PD" if ch==0 else None)
                    ax.plot(x, p_mean[:, ch], color=c_pd, linestyle="--", label="PD Mean" if ch==0 else None)

                    ax.set_ylabel(f"Ch {ch}")
                    ax.grid(True)
                    
                    # --- UPDATED LEGEND HERE ---
                    if ch == 0: 
                        ax.legend(loc="upper right")
                    # ---------------------------

                axs[-1].set_xlabel("Time (samples)")
                
                # Important: 'rect' reserves space at the top so the title doesn't get cut off
                plt.tight_layout(rect=[0, 0.00, 1, 0.91])
                
                pdf.savefig(fig)
                plt.close(fig)

            # Generate Plots
            process_and_plot(h_dom_list, pd_dom_list, "Dominant")
            process_and_plot(h_nondom_list, pd_nondom_list, "Non-Dominant")

        print(f"\n[INFO] Saved Dominance comparison PDF to: {pdf_path}\n")