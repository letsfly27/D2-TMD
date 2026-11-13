import os
import argparse

import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm

from model_student import MobileNetV4_UNet3D


def load_model(
    checkpoint_path,
    device
):

    print(
        "Loading MobileNetV4-3D Student..."
    )

    model = MobileNetV4_UNet3D(
        in_channels=1,
        num_classes=2
    ).to(device)

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if "student" in checkpoint:

        state_dict = checkpoint["student"]

    elif "model" in checkpoint:

        state_dict = checkpoint["model"]

    elif "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    else:

        state_dict = checkpoint

    state_dict = {

        k.replace("module.", ""): v

        for k, v in state_dict.items()

        if "total_ops" not in k
        and "total_params" not in k

    }

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    print(
        "Student model loaded successfully."
    )

    return model


def get_starts(
    length,
    chunk_size,
    overlap
):

    if length <= chunk_size:

        return [0]

    stride = chunk_size - overlap

    starts = list(
        range(
            0,
            length - chunk_size + 1,
            stride
        )
    )

    last_start = length - chunk_size

    if starts[-1] != last_start:

        starts.append(last_start)

    return starts


@torch.no_grad()
def infer_volume(
    model,
    volume,
    device,
    chunk_size=128,
    overlap=16
):

    volume = np.asarray(
        volume,
        dtype=np.float32
    )

    # Z-score normalization
    mean = volume.mean()

    std = volume.std() + 1e-6

    volume = (
        volume - mean
    ) / std

    D, H, W = volume.shape

    pred_sum = np.zeros(
        (D, H, W),
        dtype=np.float32
    )

    count_map = np.zeros(
        (D, H, W),
        dtype=np.float32
    )

    z_starts = get_starts(
        D,
        chunk_size,
        overlap
    )

    y_starts = get_starts(
        H,
        chunk_size,
        overlap
    )

    x_starts = get_starts(
        W,
        chunk_size,
        overlap
    )

    for z in z_starts:

        for y in y_starts:

            for x in x_starts:

                z2 = min(
                    z + chunk_size,
                    D
                )

                y2 = min(
                    y + chunk_size,
                    H
                )

                x2 = min(
                    x + chunk_size,
                    W
                )

                patch = volume[
                    z:z2,
                    y:y2,
                    x:x2
                ]

                pd = chunk_size - patch.shape[0]

                ph = chunk_size - patch.shape[1]

                pw = chunk_size - patch.shape[2]

                tensor = torch.from_numpy(
                    patch.copy()
                ).unsqueeze(0).unsqueeze(0).float()

                tensor = tensor.to(device)

                if pd > 0 or ph > 0 or pw > 0:

                    tensor = F.pad(
                        tensor,
                        (
                            0, pw,
                            0, ph,
                            0, pd
                        )
                    )

                logits, _ = model(tensor)

                prob = F.softmax(
                    logits,
                    dim=1
                )[:, 1]

                prob = prob[
                    :,
                    :patch.shape[0],
                    :patch.shape[1],
                    :patch.shape[2]
                ]

                prob = prob.squeeze(0).cpu().numpy()

                pred_sum[
                    z:z2,
                    y:y2,
                    x:x2
                ] += prob

                count_map[
                    z:z2,
                    y:y2,
                    x:x2
                ] += 1.0

    pred_prob = (
        pred_sum /
        np.maximum(count_map, 1.0)
    )

    return pred_prob


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        default="./runs/d2tmd/best_student.pth"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/test/seis"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./predictions"
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=16
    )

    parser.add_argument(
        "--shape",
        nargs=3,
        type=int,
        default=[128, 128, 128]
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = load_model(
        args.weights,
        device
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    files = sorted([

        f for f in os.listdir(
            args.input_dir
        )

        if f.endswith(".dat")

    ])

    if len(files) == 0:

        raise RuntimeError(
            "No .dat files found."
        )

    print(
        f"Starting inference on {len(files)} volumes..."
    )

    for fname in tqdm(files):

        seis_path = os.path.join(
            args.input_dir,
            fname
        )

        data = np.fromfile(
            seis_path,
            dtype=np.float32
        )

        if data.size != int(np.prod(args.shape)):

            raise ValueError(
                f"Invalid data size: {fname}"
            )

        volume = data.reshape(
            tuple(args.shape)
        )

        pred_prob = infer_volume(
            model,
            volume,
            device,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )

        pred_binary = (
            pred_prob >= 0.5
        ).astype(np.uint8)

        prob_path = os.path.join(
            args.output_dir,
            fname.replace(
                ".dat",
                "_prob.dat"
            )
        )

        binary_path = os.path.join(
            args.output_dir,
            fname.replace(
                ".dat",
                "_pred.dat"
            )
        )

        pred_prob.astype(
            np.float32
        ).tofile(prob_path)

        pred_binary.tofile(
            binary_path
        )

    print(
        "Inference completed."
    )


if __name__ == "__main__":

    main()
