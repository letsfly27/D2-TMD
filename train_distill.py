import os
import time
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeismicDataset3D
from model_student import MobileNetV4_UNet3D
from model_teacher import GeoLG_3DFaultNet
from loss_distill import D2TMD_Loss

from utils import (
    dice_score,
    iou_score,
    precision_recall,
    cldice_loss
)

from utils_light import count_parameters


# =========================================
# Paper Experimental Configuration
# =========================================

BATCH_SIZE = 1

LEARNING_RATE = 1e-4

NUM_EPOCHS = 300

DATA_SHAPE = (128, 128, 128)

CROP_SIZE = (64, 64, 64)

TEMPERATURE = 2.0

CLDICE_WEIGHT = 1.5

RANDOM_SEED = 2026


# =========================================
# Reproducibility
# =========================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


# =========================================
# Random 3D Cropping
# =========================================

def random_crop_3d(
    seismic,
    label,
    crop_size=(64, 64, 64)
):

    _, _, D, H, W = seismic.shape

    cd, ch, cw = crop_size

    if D < cd or H < ch or W < cw:

        raise ValueError(
            "Input volume is smaller than crop size."
        )

    z = random.randint(0, D - cd)

    y = random.randint(0, H - ch)

    x = random.randint(0, W - cw)

    seismic_crop = seismic[
        :,
        :,
        z:z + cd,
        y:y + ch,
        x:x + cw
    ]

    label_crop = label[
        :,
        :,
        z:z + cd,
        y:y + ch,
        x:x + cw
    ]

    return seismic_crop, label_crop


# =========================================
# Ground Truth Supervision
# =========================================

def soft_dice_loss(
    pred_logits,
    true_labels,
    smooth=1e-5
):

    pred_probs = F.softmax(
        pred_logits,
        dim=1
    )[:, 1]

    true_labels = true_labels.squeeze(1).float()

    intersection = torch.sum(
        pred_probs * true_labels
    )

    union = (
        torch.sum(pred_probs)
        + torch.sum(true_labels)
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        union + smooth
    )

    return 1.0 - dice


def ground_truth_loss(
    pred_logits,
    label
):

    label_long = label.squeeze(1).long()

    # Cross entropy
    l_ce = F.cross_entropy(
        pred_logits,
        label_long
    )

    # Dice
    l_dice = soft_dice_loss(
        pred_logits,
        label
    )

    # clDice
    pred_prob = F.softmax(
        pred_logits,
        dim=1
    )[:, 1:2]

    l_cldice = cldice_loss(
        pred_prob,
        label.float()
    )

    # Paper equation:
    # L_GT = L_CE + L_Dice + 1.5 * L_clDice

    l_gt = (
        l_ce
        + l_dice
        + CLDICE_WEIGHT * l_cldice
    )

    return l_gt, l_ce, l_dice, l_cldice


# =========================================
# Load Model Checkpoint
# =========================================

def load_teacher_weights(
    teacher,
    checkpoint_path,
    device
):

    if not os.path.isfile(checkpoint_path):

        raise FileNotFoundError(
            f"Teacher checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    elif "model" in checkpoint:

        state_dict = checkpoint["model"]

    else:

        state_dict = checkpoint

    new_state_dict = {}

    for key, value in state_dict.items():

        new_key = key.replace(
            "module.",
            ""
        )

        new_state_dict[new_key] = value

    teacher.load_state_dict(
        new_state_dict,
        strict=True
    )

    return teacher


# =========================================
# Validation
# =========================================

@torch.no_grad()
def validate(
    model,
    loader,
    device
):

    model.eval()

    dice_list = []

    iou_list = []

    precision_list = []

    recall_list = []

    for seismic, label in loader:

        seismic = seismic.to(
            device,
            non_blocking=True
        )

        label = label.to(
            device,
            non_blocking=True
        )

        pred, _ = model(seismic)

        d = dice_score(
            pred,
            label
        )

        iou = iou_score(
            pred,
            label
        )

        precision, recall = precision_recall(
            pred,
            label
        )

        dice_list.append(d.item())

        iou_list.append(iou.item())

        precision_list.append(
            precision.item()
        )

        recall_list.append(
            recall.item()
        )

    return {
        "dice": float(np.mean(dice_list)),
        "iou": float(np.mean(iou_list)),
        "precision": float(np.mean(precision_list)),
        "recall": float(np.mean(recall_list))
    }


# =========================================
# Main Training
# =========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data"
    )

    parser.add_argument(
        "--teacher_path",
        type=str,
        default="./weights/best_teacher.pth"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./runs/d2tmd"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS
    )

    parser.add_argument(
        "--crop_size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED
    )

    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=[
            "baseline",
            "nsa",
            "nsa_pod",
            "full"
        ]
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None
    )

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(
        args.save_dir,
        exist_ok=True
    )

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s | %(message)s",

        handlers=[

            logging.FileHandler(
                os.path.join(
                    args.save_dir,
                    "training.log"
                )
            ),

            logging.StreamHandler()
        ]
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    logging.info(
        f"Using device: {device}"
    )

    if device.type == "cuda":

        logging.info(
            torch.cuda.get_device_name(0)
        )

    crop_size = (
        args.crop_size,
        args.crop_size,
        args.crop_size
    )

    # =====================================
    # Dataset
    # =====================================

    logging.info("Loading training dataset...")

    train_dataset = SeismicDataset3D(

        os.path.join(
            args.data_dir,
            "train/seis"
        ),

        os.path.join(
            args.data_dir,
            "train/fault"
        ),

        shape=DATA_SHAPE,

        augment=True
    )

    val_dataset = SeismicDataset3D(

        os.path.join(
            args.data_dir,
            "validation/seis"
        ),

        os.path.join(
            args.data_dir,
            "validation/fault"
        ),

        shape=DATA_SHAPE,

        augment=False
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch_size,

        shuffle=True,

        num_workers=args.num_workers,

        pin_memory=(device.type == "cuda"),

        drop_last=False
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=1,

        shuffle=False,

        num_workers=args.num_workers,

        pin_memory=(device.type == "cuda")
    )

    logging.info(
        f"Training samples: {len(train_dataset)}"
    )

    logging.info(
        f"Validation samples: {len(val_dataset)}"
    )

    # =====================================
    # Teacher
    # =====================================

    use_teacher = (
        args.ablation != "baseline"
    )

    teacher = None

    if use_teacher:

        logging.info(
            "Loading GeoLG-3DFaultNet Teacher..."
        )

        teacher = GeoLG_3DFaultNet(
            in_channels=1,
            num_classes=2
        ).to(device)

        teacher = load_teacher_weights(
            teacher,
            args.teacher_path,
            device
        )

        teacher.eval()

        for param in teacher.parameters():

            param.requires_grad = False

    # =====================================
    # Student
    # =====================================

    logging.info(
        "Initializing MobileNetV4-3D Student..."
    )

    student = MobileNetV4_UNet3D(
        in_channels=1,
        num_classes=2
    ).to(device)

    # =====================================
    # D2-TMD Loss
    # =====================================

    d2tmd_loss_fn = D2TMD_Loss(

        s_channels=(48, 96, 128),

        t_channels=(64, 128, 256),

        lambda1=0.05,

        lambda2=1.0,

        lambda3=0.5,

        lambda_boundary=5.0,

        gamma_boundary=2.0

    ).to(device)

    logging.info(
        f"Teacher parameters: "
        f"{count_parameters(teacher) if teacher is not None else 0:.3f} M"
    )

    logging.info(
        f"Student parameters: "
        f"{count_parameters(student):.3f} M"
    )

    # =====================================
    # Optimizer
    # =====================================

    trainable_parameters = list(
        student.parameters()
    )

    if use_teacher and args.ablation in (
        "nsa",
        "nsa_pod",
        "full"
    ):

        # Important:
        # NSA projection layers must be optimized.
        trainable_parameters += list(
            d2tmd_loss_fn.adapt_layers.parameters()
        )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr
    )

    start_epoch = 0

    best_dice = 0.0

    # =====================================
    # Resume
    # =====================================

    if args.resume is not None:

        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False
        )

        student.load_state_dict(
            checkpoint["student"]
        )

        if "d2tmd_loss" in checkpoint:

            d2tmd_loss_fn.load_state_dict(
                checkpoint["d2tmd_loss"]
            )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        start_epoch = (
            checkpoint["epoch"] + 1
        )

        best_dice = checkpoint.get(
            "best_dice",
            0.0
        )

        logging.info(
            f"Resuming from epoch {start_epoch + 1}"
        )

    # =====================================
    # Ablation Configuration
    # =====================================

    use_nsa = args.ablation in (
        "nsa",
        "nsa_pod",
        "full"
    )

    use_pod = args.ablation in (
        "nsa_pod",
        "full"
    )

    use_mbc = (
        args.ablation == "full"
    )

    logging.info(
        f"Ablation: {args.ablation}"
    )

    # =====================================
    # Training Loop
    # =====================================

    for epoch in range(
        start_epoch,
        args.epochs
    ):

        student.train()

        if teacher is not None:

            teacher.eval()

        total_loss_sum = 0.0

        gt_loss_sum = 0.0

        kd_loss_sum = 0.0

        nsa_sum = 0.0

        pod_sum = 0.0

        mbc_sum = 0.0

        start_time = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}"
        )

        for seismic, label in pbar:

            seismic = seismic.float().to(
                device,
                non_blocking=True
            )

            label = label.float().to(
                device,
                non_blocking=True
            )

            # Random 64x64x64 crop
            seismic, label = random_crop_3d(
                seismic,
                label,
                crop_size
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # Student forward
            s_logits, s_feats = student(
                seismic
            )

            # Hard supervision
            l_gt, l_ce, l_dice, l_cldice = (
                ground_truth_loss(
                    s_logits,
                    label
                )
            )

            # =================================
            # Knowledge Distillation
            # =================================

            l_kd = s_logits.new_zeros(())

            l_topo = s_logits.new_zeros(())

            l_context = s_logits.new_zeros(())

            l_boundary = s_logits.new_zeros(())

            if teacher is not None:

                with torch.no_grad():

                    t_logits, t_feats = (
                        teacher.forward_with_features(
                            seismic
                        )
                    )

                if use_nsa:

                    l_topo = (
                        d2tmd_loss_fn.nsa_loss(
                            s_feats,
                            t_feats
                        )
                    )

                if use_pod:

                    l_context = (
                        d2tmd_loss_fn.pod_loss(
                            s_logits,
                            t_logits,
                            epoch + 1,
                            args.epochs,
                            temp=TEMPERATURE
                        )
                    )

                if use_mbc:

                    l_boundary = (
                        d2tmd_loss_fn.mbc_loss(
                            s_logits,
                            t_logits,
                            label
                        )
                    )

                # L_KD
                l_kd = (
                    0.05 * l_topo
                    + 1.0 * l_context
                    + 0.5 * l_boundary
                )

            # Paper:
            # L_Total = L_GT + L_KD
            loss = l_gt + l_kd

            loss.backward()

            optimizer.step()

            # =================================
            # Logging
            # =================================

            total_loss_sum += loss.item()

            gt_loss_sum += l_gt.item()

            kd_loss_sum += l_kd.item()

            nsa_sum += l_topo.item()

            pod_sum += l_context.item()

            mbc_sum += l_boundary.item()

            pbar.set_postfix({

                "Loss": f"{loss.item():.4f}",

                "GT": f"{l_gt.item():.4f}",

                "KD": f"{l_kd.item():.4f}",

                "NSA": f"{l_topo.item():.4f}",

                "POD": f"{l_context.item():.4f}",

                "MBC": f"{l_boundary.item():.4f}"

            })

        # =====================================
        # Validation
        # =====================================

        val_metrics = validate(
            student,
            val_loader,
            device
        )

        epoch_time = (
            time.time() - start_time
        )

        n_batches = len(train_loader)

        logging.info(

            f"Epoch [{epoch + 1}/{args.epochs}] "

            f"Time: {epoch_time:.1f}s | "

            f"Loss: {total_loss_sum / n_batches:.5f} | "

            f"GT: {gt_loss_sum / n_batches:.5f} | "

            f"KD: {kd_loss_sum / n_batches:.5f} | "

            f"NSA: {nsa_sum / n_batches:.5f} | "

            f"POD: {pod_sum / n_batches:.5f} | "

            f"MBC: {mbc_sum / n_batches:.5f} | "

            f"Val Dice: {val_metrics['dice']:.4f} | "

            f"Val IoU: {val_metrics['iou']:.4f} | "

            f"Precision: {val_metrics['precision']:.4f} | "

            f"Recall: {val_metrics['recall']:.4f}"

        )

        # =====================================
        # Save Checkpoint
        # =====================================

        state = {

            "epoch": epoch,

            "student": student.state_dict(),

            "d2tmd_loss": d2tmd_loss_fn.state_dict(),

            "optimizer": optimizer.state_dict(),

            "best_dice": max(
                best_dice,
                val_metrics["dice"]
            ),

            "metrics": val_metrics,

            "ablation": args.ablation

        }

        torch.save(

            state,

            os.path.join(
                args.save_dir,
                "latest_student.pth"
            )

        )

        if val_metrics["dice"] > best_dice:

            best_dice = val_metrics["dice"]

            state["best_dice"] = best_dice

            torch.save(

                state,

                os.path.join(
                    args.save_dir,
                    "best_student.pth"
                )

            )

            logging.info(
                f"New best model saved: {best_dice:.4f}"
            )

        if (epoch + 1) % 20 == 0:

            torch.save(

                state,

                os.path.join(
                    args.save_dir,
                    f"student_epoch_{epoch + 1}.pth"
                )

            )

    logging.info(
        "Training completed."
    )


if __name__ == "__main__":

    main()
