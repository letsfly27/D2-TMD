import os
import numpy as np
import matplotlib.pyplot as plt

TARGET_ID = "6"  

BASE_DIR = "./data/validation"
PRED_DIR = "./predictions"
SHAPE = (128, 128, 128)

def show_popup(idx):
    seis_path = os.path.join(BASE_DIR, "seis", f"{idx}.dat")
    fault_path = os.path.join(BASE_DIR, "fault", f"{idx}.dat")
    pred_path = os.path.join(PRED_DIR, f"{idx}.dat")

    if not os.path.exists(pred_path):
        print(f"Error: Prediction file not found: {pred_path}")
        return
    if not os.path.exists(seis_path):
        print(f"Error: Seismic data not found: {seis_path}")
        return

    print(f"Loading Sample ID: {idx} ...")

    try:
        seis = np.fromfile(seis_path, dtype=np.float32).reshape(SHAPE)
        gt = np.fromfile(fault_path, dtype=np.float32).reshape(SHAPE)
        pred = np.fromfile(pred_path, dtype=np.float32).reshape(SHAPE)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    pred_bin = (pred > 0.5).astype(np.float32)
    z_idx, y_idx, x_idx = SHAPE[0] // 4, SHAPE[1] // 4, SHAPE[2] // 4

    print("Generating visualization...")

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.canvas.manager.set_window_title(f'Detailed Visualization - ID: {idx}')

    cols = ["Seismic", "Ground Truth", "Pred (Probability)", "Pred (Binary)"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12, fontweight='bold')

    axes[0, 0].imshow(seis[z_idx], cmap='gray')
    axes[0, 0].set_ylabel("Z-Slice", fontsize=12, fontweight='bold')
    axes[0, 1].imshow(gt[z_idx], cmap='gray', vmin=0, vmax=1)
    axes[0, 2].imshow(pred[z_idx], cmap='jet', vmin=0, vmax=1)
    axes[0, 3].imshow(pred_bin[z_idx], cmap='gray', vmin=0, vmax=1)

    axes[1, 0].imshow(seis[:, y_idx, :], cmap='gray')
    axes[1, 0].set_ylabel("Y-Slice", fontsize=12, fontweight='bold')
    axes[1, 1].imshow(gt[:, y_idx, :], cmap='gray', vmin=0, vmax=1)
    axes[1, 2].imshow(pred[:, y_idx, :], cmap='jet', vmin=0, vmax=1)
    axes[1, 3].imshow(pred_bin[:, y_idx, :], cmap='gray', vmin=0, vmax=1)

    axes[2, 0].imshow(seis[:, :, x_idx], cmap='gray')
    axes[2, 0].set_ylabel("X-Slice", fontsize=12, fontweight='bold')
    axes[2, 1].imshow(gt[:, :, x_idx], cmap='gray', vmin=0, vmax=1)
    axes[2, 2].imshow(pred[:, :, x_idx], cmap='jet', vmin=0, vmax=1)
    axes[2, 3].imshow(pred_bin[:, :, x_idx], cmap='gray', vmin=0, vmax=1)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    print("Visualization ready.")
    plt.show()

if __name__ == "__main__":
    show_popup(TARGET_ID)