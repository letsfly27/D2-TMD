import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from model_student import MobileNetV4_UNet3D

CHECKPOINT_PATH = "./weights/best_student.pth"
VAL_SEIS_DIR = "./data/validation/seis"
SAVE_DIR = "./predictions"

CHUNK_SIZE = 128
OVERLAP = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    print("Loading Student Model (MobileNetV4-3D)...")
    model = MobileNetV4_UNet3D(in_channels=1, num_classes=2).to(device)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if "total_ops" not in k and "total_params" not in k:
            new_state_dict[k] = v

    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Checkpoint loaded successfully.")
    except RuntimeError as e:
        print(f"Warning: Strict loading failed, falling back to strict=False: {e}")
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    return model

def infer_volume(model, volume_path):
    data = np.fromfile(volume_path, dtype=np.float32).reshape(128, 128, 128)
    data = (data - data.mean()) / (data.std() + 1e-6)

    full_pred = np.zeros(data.shape, dtype=np.float32)
    count_map = np.zeros(data.shape, dtype=np.float32)

    d, h, w = data.shape
    step = CHUNK_SIZE - OVERLAP

    coords = []
    for z in range(0, d, step):
        for y in range(0, h, step):
            for x in range(0, w, step):
                z_end = min(z + CHUNK_SIZE, d)
                y_end = min(y + CHUNK_SIZE, h)
                x_end = min(x + CHUNK_SIZE, w)

                z_start = max(0, z_end - CHUNK_SIZE)
                y_start = max(0, y_end - CHUNK_SIZE)
                x_start = max(0, x_end - CHUNK_SIZE)
                coords.append((z_start, z_end, y_start, y_end, x_start, x_end))

    with torch.no_grad():
        for (z1, z2, y1, y2, x1, x2) in coords:
            chunk = data[z1:z2, y1:y2, x1:x2]
            tensor = torch.tensor(chunk).unsqueeze(0).unsqueeze(0).float().to(device)

            logits, _ = model(tensor)
            prob = F.softmax(logits, dim=1)[:, 1].squeeze().cpu().numpy()

            full_pred[z1:z2, y1:y2, x1:x2] += prob
            count_map[z1:z2, y1:y2, x1:x2] += 1

    avg_pred = full_pred / count_map
    return avg_pred

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    model = load_model()

    files = sorted([f for f in os.listdir(VAL_SEIS_DIR) if f.endswith('.dat')])
    print(f"Starting inference on {len(files)} volumes...")

    for fname in tqdm(files):
        seis_path = os.path.join(VAL_SEIS_DIR, fname)
        save_path = os.path.join(SAVE_DIR, fname)

        pred_prob = infer_volume(model, seis_path)
        pred_prob.tofile(save_path)

    print(f"Inference completed. Results saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()