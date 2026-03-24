import torch
from torch import nn
from torch.autograd import grad
import numpy as np

# States that are used in the model
effective_dim_start = 3
effective_dim_end = 16

# States in flat space
dim_flat_start = 0
dim_flat_end = 7

# States in manifold
dim_manifold_start = 7
dim_manifold_end = 16

# Control constraints in u_ccm
T_bound = 5
omega_bound = 0.35

# Control saturation factor
saturation_factor = 0.1

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
        
        # Determine error state in flat space
        xe_flat = x[:, dim_flat_start:dim_flat_end, :] - xstar[:, dim_flat_start:dim_flat_end, :]

        # Determine error state in manifold
        R = x[:, dim_manifold_start:dim_manifold_end, :].reshape(bs, 3, 3)
        Rstar = xstar[:, dim_manifold_start:dim_manifold_end, :].reshape(bs, 3, 3)
        Re = torch.bmm(Rstar.transpose(1, 2), R)  
        Re = 0.5 * (Re - Re.transpose(1, 2))  # R_e = 1/2 * (R_star^T @ R - R^T @ R_star)
        
        # Rotation error in Lie algebra
        xe_manifold = torch.stack([Re[:, 2, 1],
                                   Re[:, 0, 2],
                                   Re[:, 1, 0]], dim=1).reshape(bs, 3, 1)

        
        xe = torch.cat([xe_flat, xe_manifold], dim=1)

        u_raw = w2.matmul(torch.tanh(w1.matmul(xe)))
        
        # Apply control bounds with smoother saturation
        bounds = torch.tensor([T_bound, omega_bound, omega_bound, omega_bound]).type(x.type()).view(1, -1, 1).expand(bs, -1, -1)

        # Use softer saturation to avoid sudden control cutoffs
        u = bounds * torch.tanh(saturation_factor * u_raw / bounds) + uref
        # u = u_raw + uref
        
        return u


def get_model(num_dim_x, num_dim_manifold, num_dim_control, w_lb, use_cuda=False):
    dim = effective_dim_end - effective_dim_start
    
    # Neural control contraction metric
    model_Wbot = torch.nn.Sequential(
        torch.nn.Linear(dim-num_dim_control, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, (num_dim_manifold-num_dim_control) ** 2, bias=False))

    model_W = torch.nn.Sequential(
        torch.nn.Linear(dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, num_dim_manifold * num_dim_manifold, bias=False))

    # Neural controller
    c = 3 * num_dim_x
    model_u_w1 = torch.nn.Sequential(
        torch.nn.Linear(2*dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, c*num_dim_manifold, bias=True))

    model_u_w2 = torch.nn.Sequential(
        torch.nn.Linear(2*dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, num_dim_control*c, bias=True))

    # RCCM parameters
    device = torch.device("cuda" if use_cuda else "cpu")
    param_alpha_init = torch.log(torch.exp(torch.tensor(0.7, device=device)) - 1.0)  # Inverse of softplus
    param_miu_init = torch.log(torch.exp(torch.tensor(0.2, device=device)) - 1.0)
    param_lambda_init = torch.log(torch.exp(torch.tensor(0.5, device=device)) - 1.0)
    param_alpha = torch.nn.Parameter(param_alpha_init)
    param_miu   = torch.nn.Parameter(param_miu_init)
    param_lambda = torch.nn.Parameter(param_lambda_init)

    if use_cuda:
        model_W = model_W.cuda()
        model_Wbot = model_Wbot.cuda()
        model_u_w1 = model_u_w1.cuda()
        model_u_w2 = model_u_w2.cuda()

    def W_func(x):
        bs = x.shape[0]
        x = x.squeeze(-1)

        # W = Theta^T @ Theta + w_lb @ I
        W = model_W(x[:, effective_dim_start:effective_dim_end]).view(bs, num_dim_manifold, num_dim_manifold)
        
        # E_func has the structure [0; 1]
        Wbot = model_Wbot(x[:, effective_dim_start:effective_dim_end-num_dim_control]).view(bs, num_dim_manifold-num_dim_control, num_dim_manifold-num_dim_control)
        W[:, 0:num_dim_manifold-num_dim_control, 0:num_dim_manifold-num_dim_control] = Wbot
        W[:, num_dim_manifold-num_dim_control::, 0:num_dim_manifold-num_dim_control] = 0
        
        W = W.transpose(1,2).matmul(W)
        W = W + w_lb * torch.eye(num_dim_manifold).view(1, num_dim_manifold, num_dim_manifold).type(x.type())
        
        return W

    u_func = U_FUNC(model_u_w1, model_u_w2, num_dim_x, num_dim_manifold, num_dim_control)

    return model_W, model_Wbot, model_u_w1, model_u_w2, param_alpha, param_miu, param_lambda, W_func, u_func