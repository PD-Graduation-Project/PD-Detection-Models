import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def save_movement_plots_to_pdf(movement_dataloaders,
                                pdf_path="signals_per_movement.pdf",
                                max_plots_per_movement=3):
    """
    Generate multi-page PDF of signal plots for each movement.
    
    Args:
        movement_dataloaders (dict):
            { movement_name: {"train": DataLoader, "val": DataLoader} }
        pdf_path (str): output PDF file path
        max_plots_per_movement (int): number of samples plotted per movement
    """

    with PdfPages(pdf_path) as pdf:
        for movement_name, splits in movement_dataloaders.items():

            # Prefer train split, fallback to val
            if len(splits["train"]) > 0:
                dl = splits["train"]
            elif len(splits["val"]) > 0:
                dl = splits["val"]
            else:
                print(f"[WARN] No data for movement: {movement_name}")
                continue

            # Get first batch only
            signal_batch, handedness_batch, movement_batch, label_batch = next(iter(dl))
            # signal_batch: (B, 2, T, 6)

            B, _, T, C = signal_batch.shape
            num_plots = min(B, max_plots_per_movement)

            print(f"[INFO] Saving {num_plots} samples for movement: {movement_name}")

            for i in range(num_plots):
                signal = signal_batch[i]         # (2, T, 6)
                handedness = handedness_batch[i].item()
                label = label_batch[i].item()

                fig, axs = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
                fig.suptitle(
                    f"Movement: {movement_name} | Sample {i}\n"
                    f"Handedness: {'Left' if handedness == 0 else 'Right'} | Label: {label}",
                    fontsize=14
                )

                # Plot all 6 channels
                for ch in range(C):
                    axs[ch].plot(signal[0, :, ch].numpy(), label="Left wrist")
                    axs[ch].plot(signal[1, :, ch].numpy(), label="Right wrist", linestyle="--")
                    axs[ch].set_ylabel(f"Ch {ch}")
                    axs[ch].grid(True)
                    if ch == 0:
                        axs[ch].legend()

                axs[-1].set_xlabel("Time (samples)")
                plt.tight_layout()

                # Save this figure as a new page in the PDF
                pdf.savefig(fig)
                plt.close(fig)

        print(f"\n[INFO] Saved all movement plots to: {pdf_path}\n")


# =================================================================================== #
# =================================================================================== #
# =================================================================================== #


def plot_movement_comparison(movement_dataloaders,
                            pdf_path="movement_comparison.pdf",
                            max_samples_per_class=None):
    """
    For each movement:
        - Load ALL data from train + val dataloaders
        - Separate Healthy (label 0) and PD (label 1)
        - Average across samples
        - Plot per-channel (6 channels), where each plot overlays:
            - Healthy mean ± std
            - PD mean ± std
    Saves all 11 movements into ONE PDF.

    Args:
        movement_dataloaders (dict): {
            movement_name: {"train": DataLoader, "val": DataLoader}
        }
        pdf_path (str): output PDF
        max_samples_per_class (int or None): limit for debugging
    """

    with PdfPages(pdf_path) as pdf:

        for movement_name, splits in movement_dataloaders.items():
            print(f"[INFO] Processing movement: {movement_name}")

            healthy_samples = []
            pd_samples = []

            # -------- Collect all data (train + val) --------
            for split_name in ["train", "val"]:
                dl = splits[split_name]
                for batch in dl:
                    signals, handedness, movement, labels = batch
                    # signals: (B, 2, T, 6)

                    # Only label 0 (Healthy) and 1 (PD)
                    mask_healthy = labels == 0
                    mask_pd = labels == 1

                    if mask_healthy.any():
                        healthy_samples.append(signals[mask_healthy])

                    if mask_pd.any():
                        pd_samples.append(signals[mask_pd])

            if len(healthy_samples) == 0 or len(pd_samples) == 0:
                print(f"[WARN] Movement {movement_name} missing healthy or PD samples.")
                continue

            # -------- Stack into big tensors --------
            healthy = torch.cat(healthy_samples, dim=0)   # (N_h, 2, T, 6)
            pd = torch.cat(pd_samples, dim=0)             # (N_p, 2, T, 6)

            # Optionally shrink for speed
            if max_samples_per_class is not None:
                healthy = healthy[:max_samples_per_class]
                pd = pd[:max_samples_per_class]

            # -------- Average across left/right wrists --------
            # New shape: (N, T, 6)
            healthy = healthy.mean(dim=1)
            pd = pd.mean(dim=1)

            # -------- Compute mean/std for each class --------
            healthy_mean = healthy.mean(dim=0)          # (T, 6)
            healthy_std = healthy.std(dim=0)            # (T, 6)

            pd_mean = pd.mean(dim=0)                    # (T, 6)
            pd_std = pd.std(dim=0)                      # (T, 6)

            T = healthy_mean.shape[0]
            channels = 6

            # -------- Plot 6 channels --------
            fig, axs = plt.subplots(6, 1, figsize=(13, 14), sharex=True)
            fig.suptitle(f"Movement: {movement_name}\nHealthy vs PD (mean ± std)", fontsize=16)

            x = torch.arange(T)

            for ch in range(channels):
                ax = axs[ch]

                # Healthy shaded curve
                ax.fill_between(x,
                                healthy_mean[:, ch] - healthy_std[:, ch],
                                healthy_mean[:, ch] + healthy_std[:, ch],
                                alpha=0.25,
                                label="Healthy ± std" if ch == 0 else None)

                ax.plot(x, healthy_mean[:, ch],
                        label="Healthy mean" if ch == 0 else None)

                # PD shaded curve
                ax.fill_between(x,
                                pd_mean[:, ch] - pd_std[:, ch],
                                pd_mean[:, ch] + pd_std[:, ch],
                                alpha=0.25,
                                label="PD ± std" if ch == 0 else None)

                ax.plot(x, pd_mean[:, ch],
                        linestyle="--",
                        label="PD mean" if ch == 0 else None)

                ax.set_ylabel(f"Ch {ch}")
                ax.grid(True)

            axs[-1].set_xlabel("Time (samples)")
            axs[0].legend()
            plt.tight_layout()

            # Save figure as a page
            pdf.savefig(fig)
            plt.close(fig)

        print(f"\n[INFO] Saved PDF: {pdf_path}\n")
