import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import argparse
import time
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import SeismicDataset3D
from model_teacher import GeoLG_3DFaultNet
from model_student import MobileNetV4_UNet3D
from loss_distill import D2TMD_Loss
from utils import dice_score, iou_score, cldice_loss
from utils_light import count_parameters

def soft_dice_loss(pred_logits, true_labels, smooth=1e-5):
    pred_probs = F.softmax(pred_logits, dim=1)[:, 1]
    true_labels = true_labels.float()

    intersection = torch.sum(pred_probs * true_labels, dim=(1, 2, 3))
    union = torch.sum(pred_probs, dim=(1, 2, 3)) + torch.sum(true_labels, dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return (1.0 - dice).mean()

parser = argparse.ArgumentParser()
my_teacher_path = './weights/best_teacher.pth'
parser.add_argument('--teacher_path', type=str, default=my_teacher_path, help='Path to teacher weights')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=350)
parser.add_argument('--save_dir', type=str,
                    default=./runs/distill_mobilev4_d2tmd')
parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(args.save_dir, 'training.log')),
        logging.StreamHandler()
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_features = []
def get_features_hook(module, input, output):
    teacher_features.append(output)

def register_teacher_hooks(teacher_model):
    teacher_model.down1.register_forward_hook(get_features_hook)
    teacher_model.down2.register_forward_hook(get_features_hook)
    teacher_model.down3.register_forward_hook(get_features_hook)
    logging.info("Successfully registered teacher feature hooks for NSA alignment")

def main():
    logging.info(f"Loading data from {args.data_dir}...")

    train_ds = SeismicDataset3D(os.path.join(args.data_dir, 'train/seis'),
                                os.path.join(args.data_dir, 'train/fault'), augment=True)
    val_ds = SeismicDataset3D(os.path.join(args.data_dir, 'validation/seis'),
                              os.path.join(args.data_dir, 'validation/fault'), augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    logging.info("Initializing Teacher (GeoLG-3DFaultNet)...")
    teacher = GeoLG_3DFaultNet(in_channels=1, num_classes=2).to(device)
    ckpt = torch.load(args.teacher_path, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in ckpt.get('model_state_dict', ckpt).items()}
    teacher.load_state_dict(state_dict)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    register_teacher_hooks(teacher)

    logging.info("Initializing Student (MobileNetV4-3D)...")
    student = MobileNetV4_UNet3D(in_channels=1, num_classes=2).to(device)

    logging.info("=" * 40)
    logging.info(f" Teacher Params: {count_parameters(teacher):.2f} M")
    logging.info(f" Student Params: {count_parameters(student):.2f} M")
    logging.info("=" * 40)

    d2tmd_loss_fn = D2TMD_Loss(
        s_channels=[48, 96, 128],
        t_channels=[64, 128, 256],
        lambda1=0.05, 
        lambda2=1.0,  
        lambda3=0.5   
    ).to(device)
    
    ce_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_dice = 0.0

    if args.resume and os.path.isfile(args.resume):
        logging.info(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        student.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('metrics', {}).get('dice', 0.0)
        logging.info(f"Resumed successfully! Start Epoch: {start_epoch + 1}, Best Dice: {best_dice:.4f}")

    for epoch in range(start_epoch, args.epochs):
        student.train()
        log_loss = {'total': 0, 'gt': 0, 'ce': 0, 'dice': 0, 'cldice': 0, 'kd_n': 0}

        pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}/{args.epochs}")

        for img, label in pbar:
            img = img.float().to(device)
            label_long = label.long().squeeze(1).to(device)

            optimizer.zero_grad()

            global teacher_features
            teacher_features = []
            with torch.no_grad():
                t_logits = teacher(img)
                if isinstance(t_logits, tuple): t_logits = t_logits[0]
            t_feats = teacher_features

            s_logits, s_feats = student(img)

            l_ce = ce_loss_fn(s_logits, label_long)
            l_dice = soft_dice_loss(s_logits, label_long)
            s_prob = F.softmax(s_logits, dim=1)[:, 1].unsqueeze(1)
            l_cldice = cldice_loss(s_prob, label.to(device).float())
            l_gt = l_ce + l_dice + 2.0 * l_cldice

            l_kd, (l_topo, l_context, l_boundary) = d2tmd_loss_fn(
                s_logits, t_logits, s_feats, t_feats, label.to(device), epoch, args.epochs, temp=2.0
            )

            loss = 0.5 * l_gt + 0.5 * l_kd

            loss.backward()
            optimizer.step()

            log_loss['total'] += loss.item()
            log_loss['gt'] += l_gt.item()
            log_loss['ce'] += l_ce.item()
            log_loss['dice'] += l_dice.item()
            log_loss['cldice'] += l_cldice.item()
            log_loss['kd_n'] += l_kd.item()

            pbar.set_postfix({
                'L': f"{loss.item():.3f}",
                'GT': f"{l_gt.item():.3f}",
                'D2TMD': f"{l_kd.item():.3f}"
            })

        scheduler.step()

        student.eval()
        val_dice, val_iou = 0.0, 0.0
        with torch.no_grad():
            for img, label in val_loader:
                img = img.float().to(device)
                label = label.float().to(device)
                pred, _ = student(img)
                val_dice += dice_score(pred, label)
                val_iou += iou_score(pred, label)

        avg_dice = val_dice / len(val_loader)
        avg_iou = val_iou / len(val_loader)

        msg = (f"Ep {epoch + 1} | "
               f"Loss: {log_loss['total'] / len(train_loader):.3f} | "
               f"GT: {log_loss['gt'] / len(train_loader):.3f} | "
               f"D2-TMD: {log_loss['kd_n'] / len(train_loader):.3f} | "
               f"Val Dice: {avg_dice:.4f}")
        logging.info(msg)

        state = {
            'epoch': epoch,
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'metrics': {'dice': avg_dice, 'iou': avg_iou}
        }

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(state, os.path.join(args.save_dir, 'best_student.pth'))
            logging.info(f"New Best Student Model Saved (Dice: {best_dice:.4f})")

        if (epoch + 1) % 20 == 0:
            torch.save(state, os.path.join(args.save_dir, f'student_epoch_{epoch + 1}.pth'))

if __name__ == '__main__':
    main()