import torch
import torch.nn as nn
import torch.nn.functional as F


class D2TMD_Loss(nn.Module):

    def __init__(
        self,
        s_channels=(48, 96, 128),
        t_channels=(64, 128, 256),
        lambda1=0.05,
        lambda2=1.0,
        lambda3=0.5,
        lambda_boundary=5.0,
        gamma_boundary=2.0
    ):

        super().__init__()

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.lambda_boundary = lambda_boundary
        self.gamma_boundary = gamma_boundary

        self.adapt_layers = nn.ModuleList()

        for s_c, t_c in zip(
            s_channels,
            t_channels
        ):

            self.adapt_layers.append(
                nn.Conv3d(
                    s_c,
                    t_c,
                    kernel_size=1
                )
            )

    # ==================================
    # Forward
    # ==================================

    def forward(
        self,
        stud_logits,
        tea_logits,
        stud_feats,
        tea_feats,
        target,
        current_epoch,
        max_epochs,
        temp=2.0
    ):

        l_context = self.pod_loss(
            stud_logits,
            tea_logits,
            current_epoch,
            max_epochs,
            temp
        )

        l_topo = self.nsa_loss(
            stud_feats,
            tea_feats
        )

        l_boundary = self.mbc_loss(
            stud_logits,
            tea_logits,
            target
        )

        total_kd_loss = (
            self.lambda1 * l_topo
            + self.lambda2 * l_context
            + self.lambda3 * l_boundary
        )

        return total_kd_loss, (
            l_topo.detach(),
            l_context.detach(),
            l_boundary.detach()
        )

    # ==================================
    # NSA
    # Numerical-Structural Alignment
    # ==================================

    def nsa_loss(
        self,
        s_feats,
        t_feats
    ):

        if len(s_feats) != len(self.adapt_layers):
            raise ValueError(
                "Student feature stage number mismatch."
            )

        if len(t_feats) != len(self.adapt_layers):
            raise ValueError(
                "Teacher feature stage number mismatch."
            )

        loss_topo = 0.0

        for i, (s, t) in enumerate(
            zip(s_feats, t_feats)
        ):

            t = t.detach()

            # Adaptive channel projection
            s_aligned = self.adapt_layers[i](s)

            # Spatial alignment
            if s_aligned.shape[2:] != t.shape[2:]:

                s_aligned = F.interpolate(
                    s_aligned,
                    size=t.shape[2:],
                    mode='trilinear',
                    align_corners=False
                )

            # Numerical alignment
            loss_mse = F.mse_loss(
                s_aligned,
                t
            )

            # Channel correlation matrices
            b, c, d, h, w = t.shape

            s_flat = s_aligned.reshape(
                b, c, -1
            )

            t_flat = t.reshape(
                b, c, -1
            )

            spatial_size = d * h * w

            s_corr = torch.bmm(
                s_flat,
                s_flat.transpose(1, 2)
            ) / spatial_size

            t_corr = torch.bmm(
                t_flat,
                t_flat.transpose(1, 2)
            ) / spatial_size

            # Structural alignment
            loss_struct = F.mse_loss(
                s_corr,
                t_corr
            )

            loss_topo = (
                loss_topo
                + loss_mse
                + loss_struct
            )

        return loss_topo

    # ==================================
    # POD
    # Process-aware Omni-granularity
    # Decoupling
    # ==================================

    def pod_loss(
        self,
        s_logits,
        t_logits,
        current_epoch,
        max_epochs,
        temp=2.0
    ):

        rho = (
            float(current_epoch)
            / float(max_epochs)
        )

        rho = max(
            0.0,
            min(1.0, rho)
        )

        # Dynamic weights from the paper
        w_macro = (
            0.1 + 0.5 * (1.0 - rho)
        )

        w_meso = 0.2

        w_micro = (
            0.1 + 0.5 * rho
        )

        scales = [4, 2, 1]

        weights = [
            w_macro,
            w_meso,
            w_micro
        ]

        with torch.no_grad():

            t_logits = t_logits.detach()

        loss_context = s_logits.new_zeros(())

        for k, weight in zip(
            scales,
            weights
        ):

            if k == 1:

                s_scale = s_logits

                t_scale = t_logits

            else:

                # Spatial average pooling
                s_scale = F.avg_pool3d(
                    s_logits,
                    kernel_size=k,
                    stride=k
                )

                t_scale = F.avg_pool3d(
                    t_logits,
                    kernel_size=k,
                    stride=k
                )

            # Temperature-softened probabilities
            s_log_prob = F.log_softmax(
                s_scale / temp,
                dim=1
            )

            t_prob = F.softmax(
                t_scale / temp,
                dim=1
            )

            # KL(P_T || P_S)
            kl_map = F.kl_div(
                s_log_prob,
                t_prob,
                reduction='none'
            ).sum(dim=1)

            # Average over spatial positions and batch
            loss_scale = kl_map.mean()

            loss_context = (
                loss_context
                + weight
                * (temp ** 2)
                * loss_scale
            )

        return loss_context

    # ==================================
    # MBC
    # Morphology-adaptive Boundary
    # Correction
    # ==================================

    def mbc_loss(
        self,
        s_logits,
        t_logits,
        target
    ):

        if target.dim() == 4:

            gt_float = target.unsqueeze(1).float()

        else:

            gt_float = target.float()

        gt_float = gt_float.detach()

        # 3D local smoothing
        y_smooth = F.avg_pool3d(
            gt_float,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Morphological boundary mask
        m_boundary = torch.abs(
            gt_float - y_smooth
        )

        # Exponential spatial weight
        w_spatial = (
            1.0
            + self.lambda_boundary
            * (
                torch.exp(
                    self.gamma_boundary * m_boundary
                ) - 1.0
            )
        )

        # Teacher probability
        with torch.no_grad():

            t_prob = F.softmax(
                t_logits.detach(),
                dim=1
            )

        # Student probability
        s_prob = F.softmax(
            s_logits,
            dim=1
        )

        # Squared L2 distance
        l2_map = (
            s_prob - t_prob
        ).pow(2).sum(
            dim=1,
            keepdim=True
        )

        # Boundary-weighted loss
        loss_boundary = (
            w_spatial * l2_map
        ).mean()

        return loss_boundary
