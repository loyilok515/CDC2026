import torch
from torch import nn
from torch.autograd import grad
import numpy as np

# States that are used in the model
effective_dim_start = 0
effective_dim_end = 12

# States in manifold
dim_manifold_start = 0
dim_manifold_end = 9

# Effective states in flat space
dim_flat_start = 9
dim_flat_end = 12

# # Control constraints (should match config_MUAV.py)
f_bound = 5.0

# # Control saturation factor - increased to avoid premature saturation
saturation_factor = 0.3  # Changed from 0.01 to 0.1, further changed to 0.3

class U_FUNC(nn.Module):
    """docstring for U_FUNC."""

    def __init__(self, model_u_w1, model_u_w2, num_dim_x, num_dim_manifold, num_dim_control):
        super(U_FUNC, self).__init__()
        self.model_u_w1 = model_u_w1
        self.model_u_w2 = model_u_w2
        self.num_dim_x = num_dim_x
        self.num_dim_manifold = num_dim_manifold
        self.num_dim_control = num_dim_control

    def forward(self, x, xstar, uref):
        # x: B x n x 1
        # u: B x m x 1
        bs = x.shape[0]

        w1 = self.model_u_w1(torch.cat([x[:,effective_dim_start:effective_dim_end,:],xstar[:,effective_dim_start:effective_dim_end,:]],dim=1).squeeze(-1)).reshape(bs, -1, self.num_dim_manifold)
        w2 = self.model_u_w2(torch.cat([x[:,effective_dim_start:effective_dim_end,:],xstar[:,effective_dim_start:effective_dim_end,:]],dim=1).squeeze(-1)).reshape(bs, self.num_dim_control, -1)
        
        # Determine error state in manifold
        R = x[:, dim_manifold_start:dim_manifold_end, :].reshape(bs, 3, 3)
        Rstar = xstar[:, dim_manifold_start:dim_manifold_end, :].reshape(bs, 3, 3)
        Re = torch.bmm(Rstar.transpose(1, 2), R)  # R_e = R_star^T * R
        Re = 0.5 * (Re - Re.transpose(1, 2))
        
        # Method 1 (Element-wise difference, R-R*)
        # xe_manifold = x[:, dim_manifold_start:dim_manifold_end, :] - xstar[:, dim_manifold_start:dim_manifold_end, :]
        
        # Method 2 (Rotation error in Lie algebra)
        xe_manifold = 0.5 * torch.stack([Re[:, 2, 1],
                                         Re[:, 0, 2],
                                         Re[:, 1, 0]], dim=1).reshape(bs, 3, 1)

        xe_flat = x[:, dim_flat_start:dim_flat_end, :] - xstar[:, dim_flat_start:dim_flat_end, :]
        xe = torch.cat([xe_manifold, xe_flat], dim=1)

        u_raw = w2.matmul(torch.tanh(w1.matmul(xe)))
        
        # # Apply control bounds with smoother saturation
        # bounds = torch.tensor([f_bound] * self.num_dim_control, dtype=x.dtype, device=x.device).view(1, -1, 1).expand(bs, -1, -1)
        
        # Use softer saturation to avoid sudden control cutoffs
        # u = torch.tanh(u_raw * saturation_factor) * bounds + uref
        u = u_raw + uref
        
        return u


def get_model(num_dim_x, num_dim_manifold, num_dim_control, w_lb, use_cuda=False):
    dim = effective_dim_end - effective_dim_start
    
    model_Wbot = torch.nn.Sequential(
        torch.nn.Linear(dim-num_dim_control, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, (num_dim_manifold-num_dim_control) ** 2, bias=False))

    model_W = torch.nn.Sequential(
        torch.nn.Linear(dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, num_dim_manifold * num_dim_manifold, bias=False))

    c = 3 * num_dim_x
    model_u_w1 = torch.nn.Sequential(
        torch.nn.Linear(2*dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, c*num_dim_manifold, bias=True))

    model_u_w2 = torch.nn.Sequential(
        torch.nn.Linear(2*dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, num_dim_control*c, bias=True))

    if use_cuda:
        model_W = model_W.cuda()
        model_Wbot = model_Wbot.cuda()
        model_u_w1 = model_u_w1.cuda()
        model_u_w2 = model_u_w2.cuda()

    def W_func(x):
        bs = x.shape[0]
        x = x.squeeze(-1)

        W = model_W(x[:, effective_dim_start:effective_dim_end]).view(bs, num_dim_manifold, num_dim_manifold)
        
        # Assuming the B(x) is structured as follows:
        # B(x) = [0, b(x)], where b(x) is invertible
        # Wbot = model_Wbot(x[:, effective_dim_start:effective_dim_end-num_dim_control]).view(bs, num_dim_manifold-num_dim_control, num_dim_manifold-num_dim_control)
        # W[:, 0:num_dim_manifold-num_dim_control, 0:num_dim_manifold-num_dim_control] = Wbot
        # W[:, num_dim_manifold-num_dim_control::, 0:num_dim_manifold-num_dim_control] = 0

        W = W.transpose(1,2).matmul(W)
        W = W + w_lb * torch.eye(num_dim_manifold).view(1, num_dim_manifold, num_dim_manifold).type(x.type())
        
        return W

    u_func = U_FUNC(model_u_w1, model_u_w2, num_dim_x, num_dim_manifold, num_dim_control)

    return model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func