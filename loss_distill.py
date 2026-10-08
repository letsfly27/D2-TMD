import torch
import torch.nn as nn
import torch.nn.functional as F

class D2TMD_Loss(nn.Module):

    def __init__(self, s_channels=[48, 96, 128], t_channels=[64, 128, 256],
                 lambda1=0.05, lambda2=1.0, lambda3=0.5):
        super().__init__()
        self.lambda1 = lambda1  # Topo (NSA module) weight
        self.lambda2 = lambda2  # Context (POD module) weight
        self.lambda3 = lambda3  # Boundary (MBC module) weight

        
        self.adapt_layers = nn.ModuleList()
        for s_c, t_c in zip(s_channels, t_channels):
            self.adapt_layers.append(nn.Conv3d(s_c, t_c, 1))

    def forward(self, stud_logits, tea_logits, stud_feats, tea_feats, target, current_epoch, max_epochs, temp=2.0):
        
       
        l_context = self.pod_loss(stud_logits, tea_logits, target, current_epoch, max_epochs, temp)

        
        l_topo = self.nsa_loss(stud_feats, tea_feats)

       	l_boundary = self.mbc_loss(stud_logits, tea_logits, target, temp)

        
        total_kd_loss = self.lambda1 * l_topo + self.lambda2 * l_context + self.lambda3 * l_boundary

        return total_kd_loss, (l_topo.item(), l_context.item(), l_boundary.item())

        def pod_loss(self, s_logits, t_logits, target, current_epoch, max_epochs, temp=2.0):
       	rho = current_epoch / max_epochs
        
       	w_macro = 0.1 + 0.5 * (1.0 - rho)  # scale k=4
        w_meso = 0.2                       # scale k=2
        w_micro = 0.1 + 0.5 * rho          # scale k=1

        s_probs = F.softmax(s_logits / temp, dim=1)
        with torch.no_grad():
            t_probs = F.softmax(t_logits / temp, dim=1)

        
        if target.dim() == 4:
            gt_mask = target.unsqueeze(1).float()
        else:
            gt_mask = target.float()
        focus_mask = F.max_pool3d(gt_mask, kernel_size=3, stride=1, padding=1)

        loss_context = 0.0

        
        scales = [4, 2, 1]
        weights = [w_macro, w_meso, w_micro]

        for k, w in zip(scales, weights):
            
            kernel_size = 2 * k - 1 
            pad = kernel_size // 2

            if kernel_size > 1:
                s_scale_prob = F.avg_pool3d(s_probs, kernel_size=kernel_size, stride=1, padding=pad)
                t_scale_prob = F.avg_pool3d(t_probs, kernel_size=kernel_size, stride=1, padding=pad)
            else:
                s_scale_prob = s_probs
                t_scale_prob = t_probs

            
            kl_map = F.kl_div(torch.log(s_scale_prob + 1e-8), t_scale_prob, reduction='none').sum(dim=1, keepdim=True)
            kl_masked = kl_map * focus_mask
            
            loss_context += w * kl_masked.mean() * (temp ** 2)

        return loss_context

    
    def nsa_loss(self, s_feats, t_feats):
        loss_topo = 0.0
        for i, (s, t) in enumerate(zip(s_feats, t_feats)):
            
            s_aligned = self.adapt_layers[i](s)

            if s_aligned.shape != t.shape:
                s_aligned = F.interpolate(s_aligned, size=t.shape[2:], mode='trilinear', align_corners=False)

            
            loss_mse = F.mse_loss(s_aligned, t)

            
            b, c, d, h, w = t.shape
            t_flat = t.view(b, c, -1)
            s_flat = s_aligned.view(b, c, -1)

            t_corr = torch.bmm(t_flat, t_flat.transpose(1, 2)) / (d * h * w)
            s_corr = torch.bmm(s_flat, s_flat.transpose(1, 2)) / (d * h * w)

            
            loss_struct = F.mse_loss(s_corr, t_corr)
            
            
            loss_topo += (loss_mse + loss_struct)

        return loss_topo

    
    def mbc_loss(self, s_logits, t_logits, target, temp=2.0):
        s_log_probs = F.log_softmax(s_logits / temp, dim=1)
        with torch.no_grad():
            t_probs = F.softmax(t_logits / temp, dim=1)

        
        kl_map = F.kl_div(s_log_probs, t_probs, reduction='none').sum(dim=1, keepdim=True)

        if target.dim() == 4:
            gt_float = target.unsqueeze(1).float()
        else:
            gt_float = target.float()

        
        y_smooth = F.avg_pool3d(gt_float, kernel_size=3, stride=1, padding=1)
        m_boundary = torch.abs(gt_float - y_smooth)
        
        
        if m_boundary.max() > 0:
            m_boundary = m_boundary / (m_boundary.max() * 2.0 + 1e-8)

        
        lambda_val = 5.0  
        gamma_val = 2.0   
        
        
        w_spatial = 1.0 + lambda_val * (torch.exp(gamma_val * m_boundary) - 1.0)

        
        loss_boundary = (w_spatial * kl_map).mean() * (temp ** 2)

        return loss_boundary