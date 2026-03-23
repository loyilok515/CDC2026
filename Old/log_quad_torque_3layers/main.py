import torch
from torch.autograd import grad
import torch.nn.functional as F

import importlib
import numpy as np
import time
from tqdm import tqdm
from utils import euler_xyz_to_rotmat

import os
import sys
sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')

# Hyperparameters
bs = 1024  # Batch size
num_train = 1024*128  # Number of samples for training
num_test = 1024*32  # Number of samples for testing
learning_rate = 0.001  # Base learning rate
epochs = 30  # Number of training epochs
lr_step = 10  # Learning rate step
detach_epoch = 10  # Number of epochs to detach the computation graph
RCCM_start_epoch = 0  # Number of epochs to start RCCM training
_lambda_fixed = 0.5  # Convergence rate: lambda
w_ub = 10  # Upper bound of the eigenvalue of the dual metric
w_lb = 0.1  # Lower bound of the eigenvalue of the dual metric
alpha_lb = 0.9  # Lower bound of the alpha parameter in RCCM
epsilon = _lambda_fixed * 0.1  # Ensure definiteness of LMI

# Configuration variables
task = 'quad_torque' #'quad_torque'  # Name of the model
log = 'log_quad_torque_3layers' #'log_quad_RCCM_torque'  # Path to a directory for storing the training log
log_record_file = log+'/training_log.txt'
use_cuda = True  # Set to False to disable CUDA

np.random.seed(1024)

# Ensure the log directory exists
os.makedirs(log, exist_ok=True)
# Copies all files into log for reference
os.system('cp *.py '+log)
os.system('cp -r models/ '+log)
os.system('cp -r configs/ '+log)
os.system('cp -r systems/ '+log)
# Creates text file for recording training data
with open(log_record_file, 'w') as f:
    pass

# Get training bounds from config file
config = importlib.import_module('config_'+task)
X_MIN = config.X_MIN
X_MAX = config.X_MAX
U_MIN = config.UREF_MIN
U_MAX = config.UREF_MAX
XE_MIN = config.XE_MIN
XE_MAX = config.XE_MAX
w_MIN = config.w_MIN
w_MAX = config.w_MAX

# Get system functions from system file
system = importlib.import_module('system_'+task)
S_func = system.S_func
f_func = system.f_func
B_func = system.B_func
B_w_func = system.B_w_func
C_func = system.C_func
D_func = system.D_func
E_func = system.E_func
E_w_func = system.E_w_func
Ebot_func = system.Ebot_func
num_dim_x = system.num_dim_x
num_dim_control = system.num_dim_control
num_dim_manifold = system.num_dim_manifold
num_dim_noise = system.num_dim_noise
num_dim_z = system.num_dim_z

# Get neural network configurations from model file
model = importlib.import_module('model_'+task)
get_model = model.get_model

model_W, model_Wbot, model_u_w1, model_u_w2, param_alpha, param_miu, param_lambda, W_func, u_func = get_model(num_dim_x, num_dim_manifold, num_dim_control, w_lb=w_lb, use_cuda=use_cuda)

# Generate training samples
def sample_xef():
    # Random sample xstar based on sampling bounds
    XREF = (X_MAX-X_MIN) * np.random.rand(num_dim_manifold, 1) + X_MIN

    # Get reference attitude from euler angles
    euler_ref = XREF[num_dim_manifold-3:, :]
    Rref = euler_xyz_to_rotmat(euler_ref[0,0], euler_ref[1,0], euler_ref[2,0])
    rref = Rref.flatten().reshape(-1, 1)

    # Combine states
    xref = np.zeros(num_dim_x).reshape(-1, 1)
    xref[0:num_dim_manifold-3, 0] =  XREF[0:num_dim_manifold-3, 0] # Flat space
    xref[num_dim_manifold-3:, 0] = rref[:, 0] # Manifold
    return xref

def sample_x(xref):
    # Random sample xe based on sampling bounds
    XE = (XE_MAX-XE_MIN) * np.random.rand(num_dim_manifold, 1) + XE_MIN

    # Get attitude error from euler angles
    euler_e = XE[num_dim_manifold-3:, :]
    R_e = euler_xyz_to_rotmat(euler_e[0,0], euler_e[1,0], euler_e[2,0])
    Rref = xref[num_dim_manifold-3:, 0].reshape(3, 3)
    R = Rref @ R_e
    r = R.flatten().reshape(-1, 1)

    # Combine states
    x = np.zeros(num_dim_x).reshape(-1, 1)
    x[0:num_dim_manifold-3, 0] = xref[0:num_dim_manifold-3, 0] + XE[0:num_dim_manifold-3, 0] # Flat space
    x[num_dim_manifold-3:, 0] = r[:, 0] # Manifold
    return x

def sample_uref():
    # Random sample ustar based on sampling bounds
    return (U_MAX-U_MIN) * np.random.rand(num_dim_control, 1) + U_MIN

def sample_w():
    return (w_MAX-w_MIN) * np.random.rand(num_dim_noise, 1) + w_MIN

def sample_full():
    xref = sample_xef()
    uref = sample_uref()
    x = sample_x(xref)
    w = sample_w()
    return (x, xref, uref, w)

X_tr = [sample_full() for _ in range(num_train)]
X_te = [sample_full() for _ in range(num_test)]

# Define functions for Jacobians
def Jacobian_Matrix(M, x):
    # NOTE that this function assume that data are independent of each other
    # along the batch dimension.
    # M: B x m x p
    # x: B x n x 1
    # ret: B x m x p x n
    bs = x.shape[0]
    m = M.size(1)
    p = M.size(2)
    n = x.size(1)
    J = torch.zeros(bs, m, p, n).type(x.type())
    for i in range(m):
        for j in range(p):
            J[:, i, j, :] = grad(M[:, i, j].sum(), x, create_graph=True)[0].squeeze(-1)
    return J

def Jacobian(f, x):
    # NOTE that this function assume that data are independent of each other
    f = f + 0. * x.sum() # to avoid the case that f is independent of x
    # f: B x m x 1
    # x: B x n x 1
    # ret: B x m x n
    bs = x.shape[0]
    m = f.size(1)
    n = x.size(1)
    J = torch.zeros(bs, m, n).type(x.type())
    for i in range(m):
        J[:, i, :] = grad(f[:, i, 0].sum(), x, create_graph=True)[0].squeeze(-1)
    return J

def weighted_gradients(W, v, x, detach=False):
    # v, x: bs x n x 1
    # DWDx: bs x n x n x n
    assert v.size() == x.size()
    bs = x.shape[0]
    if detach:
        return (Jacobian_Matrix(W, x).detach() * v.view(bs, 1, 1, -1)).sum(dim=3)
    else:
        return (Jacobian_Matrix(W, x) * v.view(bs, 1, 1, -1)).sum(dim=3)

# Loss functions for LMIs, penalizing matrix for its negative definiteness
K = 1024  # number of random unit vectors for checking positive definiteness
def loss_pos_matrix_random_sampling(A):
    # A: bs x d x d
    # z: K x d
    z = torch.randn(K, A.size(-1)).cuda()
    z = z / z.norm(dim=1, keepdim=True)
    zTAz = (z.matmul(A) * z.view(1,K,-1)).sum(dim=2).view(-1)
    negative_index = zTAz.detach().cpu().numpy() < 0
    if negative_index.sum()>0:
        negative_zTAz = zTAz[negative_index]
        return -1.0 * (negative_zTAz.mean())
    else:
        return torch.tensor(0.).type(z.type()).requires_grad_()

def loss_pos_matrix_eigen_values(A):
    # A: bs x d x d
    eigv = torch.linalg.eigvalsh(A, UPLO='L').view(-1)
    negative_index = eigv.detach().cpu().numpy() < 0
    negative_eigv = eigv[negative_index]
    return negative_eigv.norm()

def forward(x, xref, uref, w, verbose=False, acc=False, detach=False):
    # x: bs x n x 1
    bs = x.shape[0]
    W = W_func(x) # W: bs x num_dim_manifold x num_dim_manifold
    M = torch.inverse(W) # M: bs x num_dim_manifold x num_dim_manifold
    S = S_func(x)
    P_s = S # For specific systems
    P_s_T = P_s.transpose(1,2)
    f = f_func(x)
    B = B_func(x)
    B_w = B_w_func(x)
    C = C_func(x)
    D = D_func(x)
    E = E_func(x)
    E_w = E_w_func(x)
    _Ebot = Ebot_func(x)
    DfDx = Jacobian(f, x)
    DBDx = torch.zeros(bs, num_dim_x, num_dim_x, num_dim_control).type(x.type())
    for i in range(num_dim_control):
        DBDx[:,:,:,i] = Jacobian(B[:,:,i].unsqueeze(-1), x)
    DBwDx = torch.zeros(bs, num_dim_x, num_dim_x, num_dim_noise).type(x.type())
    for i in range(num_dim_noise):
        DBwDx[:,:,:,i] = Jacobian(B_w[:,:,i].unsqueeze(-1), x)

    u = u_func(x, xref, uref) # u: bs x m x 1
    K = Jacobian(u, x)

    # compute RCCM parameters (force positive constants)
    _alpha = F.softplus(param_alpha)
    _miu = F.softplus(param_miu)
    _lambda_nn = F.softplus(param_lambda)
    _lambda = _lambda_fixed #  _lambda = _lambda_nn

    A = ( DfDx + sum([u[:, i, 0].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i] for i in range(num_dim_control)])
             + sum([w[:, i, 0].unsqueeze(-1).unsqueeze(-1) * DBwDx[:, :, :, i] for i in range(num_dim_noise)])  )
    dot_x = f + B.matmul(u) + B_w.matmul(w)
    dot_M = weighted_gradients(M, dot_x, x, detach=detach) # DMDt
    dot_W = weighted_gradients(W, dot_x, x, detach=detach) # DWDt
    dot_P_s_T = weighted_gradients(P_s_T, dot_x, x, detach=detach)  # DPsTdt

    S_f = weighted_gradients(P_s_T, f, x).matmul(S) + P_s_T.matmul(DfDx).matmul(S)
    S_B = torch.zeros(bs, num_dim_manifold, num_dim_manifold, num_dim_control).type(x.type())
    for i in range(num_dim_control):
        S_B[:, :, :, i] = weighted_gradients(P_s_T, B[:, :, i].unsqueeze(-1), x).matmul(S) + P_s_T.matmul(DBDx[:, :, :, i]).matmul(S)
    S_Bw = torch.zeros(bs, num_dim_manifold, num_dim_manifold, num_dim_noise).type(x.type())
    for i in range(num_dim_noise):
        S_Bw[:, :, :, i] = weighted_gradients(P_s_T, B_w[:, :, i].unsqueeze(-1), x).matmul(S) + P_s_T.matmul(DBwDx[:, :, :, i]).matmul(S)
   
    
    # Contraction condition (detach for first few epochs to let C1 and C2 find a CCM first)
    if detach:
        Contraction = dot_M + ((dot_P_s_T + P_s_T.matmul(A) + E.matmul(K)).matmul(S)).transpose(1,2).matmul(M.detach()) + M.detach().matmul((dot_P_s_T + P_s_T.matmul(A) + E.matmul(K)).matmul(S)) + 2 * _lambda * M.detach()
        Contraction_dual = -dot_W + W.detach().matmul(((dot_P_s_T + P_s_T.matmul(A) + E.matmul(K)).matmul(S)).transpose(1,2)) + ((dot_P_s_T + P_s_T.matmul(A) + E.matmul(K)).matmul(S)).matmul(W.detach()) + 2 * _lambda * W.detach()
    else:
        Contraction = dot_M + ((dot_P_s_T + P_s_T.matmul(A) + E.matmul(K)).matmul(S)).transpose(1,2).matmul(M) + M.matmul((dot_P_s_T + P_s_T.matmul(A) + E.matmul(K)).matmul(S)) + 2 * _lambda * M
        Contraction_dual = -dot_W + W.matmul(((dot_P_s_T + P_s_T.matmul(A) + E.matmul(K)).matmul(S)).transpose(1,2)) + ((dot_P_s_T + P_s_T.matmul(A) + E.matmul(K)).matmul(S)).matmul(W) + 2 * _lambda * W

    # # RCCM_1 (Condition (15)) (dual)
    # RCCM_1_11 = Contraction_dual
    # RCCM_1_12 = E_w # P_s_T.matmul(B_w)
    # RCCM_1_21 = RCCM_1_12.transpose(1,2)
    # RCCM_1_22 = - _miu * torch.eye(num_dim_noise).repeat(bs, 1, 1).type(x.type())
    # RCCM_1 = torch.cat([
    #     torch.cat([RCCM_1_11, RCCM_1_12], dim=2),
    #     torch.cat([RCCM_1_21, RCCM_1_22], dim=2)
    # ], dim=1)

    # # RCCM_2 (Condition (16))
    # RCCM_2_11 = 2*_lambda*M - (S.transpose(1,2)).matmul((C + D.matmul(K)).transpose(1,2)).matmul(C + D.matmul(K)).matmul(S) / _alpha
    # RCCM_2_12 = torch.zeros(bs, num_dim_manifold, num_dim_noise).type(x.type())
    # RCCM_2_21 = RCCM_2_12.transpose(1,2)
    # RCCM_2_22 = (_alpha - _miu) * torch.eye(num_dim_noise).repeat(bs, 1, 1).type(x.type())
    # RCCM_2 = torch.cat([
    #     torch.cat([RCCM_2_11, RCCM_2_12], dim=2),
    #     torch.cat([RCCM_2_21, RCCM_2_22], dim=2)
    # ], dim=1)

    # RCCM_1 (Condition (15))
    RCCM_1_11 = Contraction
    RCCM_1_12 = torch.matmul(M, E_w)
    RCCM_1_21 = RCCM_1_12.transpose(1,2)
    RCCM_1_22 = - _miu * torch.eye(num_dim_noise).repeat(bs, 1, 1).type(x.type())
    RCCM_1 = torch.cat([
        torch.cat([RCCM_1_11, RCCM_1_12], dim=2),
        torch.cat([RCCM_1_21, RCCM_1_22], dim=2)
    ], dim=1)

    # RCCM_2 (Condition (16))
    RCCM_2 = 2*_lambda*M - (S.transpose(1,2)).matmul((C + D.matmul(K)).transpose(1,2)).matmul(C + D.matmul(K)).matmul(S) / _alpha

    # C1
    C1_inner = - weighted_gradients(W, f, x) + S_f.matmul(W) + W.matmul(S_f.transpose(1,2)) + 2 * _lambda * W
    C1_LHS_1 = _Ebot.transpose(1,2).matmul(C1_inner).matmul(_Ebot)

    # C2
    C2_inners = []
    C2s = []
    for j in range(num_dim_control):
        C2_inner = weighted_gradients(W, B[:,:,j].unsqueeze(-1), x) - (S_B[:,:,:,j].matmul(W) + W.matmul(S_B[:,:,:,j].transpose(1,2)))
        C2 = _Ebot.transpose(1,2).matmul(C2_inner).matmul(_Ebot)
        C2_inners.append(C2_inner)
        C2s.append(C2)

    # C2w
    C2w_inners = []
    C2ws = []
    for j in range(num_dim_noise):
        C2w_inner = weighted_gradients(W, B_w[:,:,j].unsqueeze(-1), x) - (S_Bw[:,:,:,j].matmul(W) + W.matmul(S_Bw[:,:,:,j].transpose(1,2)))
        C2w = _Ebot.transpose(1,2).matmul(C2w_inner).matmul(_Ebot)
        C2w_inners.append(C2w_inner)
        C2ws.append(C2w)

    loss_CCM = loss_pos_matrix_random_sampling(-Contraction - epsilon * torch.eye(Contraction.shape[-1]).unsqueeze(0).type(x.type()))
    loss_w_ub = loss_pos_matrix_random_sampling(w_ub * torch.eye(W.shape[-1]).unsqueeze(0).type(x.type()) - W)
    loss_C1 = loss_pos_matrix_random_sampling(-C1_LHS_1 - epsilon * torch.eye(C1_LHS_1.shape[-1]).unsqueeze(0).type(x.type()))
    loss_C2 = 4. * sum([1.*(C2**2).reshape(bs,-1).sum(dim=1).mean() for C2 in C2s])
    # loss_C2w = 1. * sum([1.*(C2w_inner**2).reshape(bs,-1).sum(dim=1).mean() for C2w_inner in C2w_inners])
    # loss_RCCM1 = loss_pos_matrix_random_sampling(-RCCM_1 - epsilon * torch.eye(RCCM_1.shape[-1]).unsqueeze(0).type(x.type()))
    # loss_RCCM2 = loss_pos_matrix_random_sampling(RCCM_2 - epsilon * torch.eye(RCCM_2.shape[-1]).unsqueeze(0).type(x.type()))
    # loss_alpha = 0.5 * torch.relu(_alpha - alpha_lb)
    # loss_miu = 2. * torch.relu(_miu - _alpha)

    if epoch < RCCM_start_epoch: 
        loss = loss_CCM + loss_w_ub + loss_C1 + loss_C2 # + loss_C2w
    else:
        loss = loss_CCM + loss_w_ub + loss_C1 + loss_C2 # + loss_C2w + loss_RCCM1 + loss_RCCM2 + loss_alpha + loss_miu

    if verbose:
        #print("loss_CCM: %.6f, loss_w_ub: %.6f, loss_C1: %.6f, loss_C2: %.6f, loss_C2w: %.6f, loss_RCCM1: %.6f, loss_RCCM2: %.6f, loss_alpha: %.6f, loss_miu: %.6f"%(loss_CCM.item(), loss_w_ub.item(), loss_C1.item(), loss_C2.item(), loss_C2w.item(), loss_RCCM1.item(), loss_RCCM2.item(), loss_alpha.item(), loss_miu.item()))
        print(torch.linalg.eigvalsh(Contraction).min(dim=1)[0].mean().item(), torch.linalg.eigvalsh(Contraction).max(dim=1)[0].mean().item(), torch.linalg.eigvalsh(Contraction).mean().item())
        print(torch.linalg.eigvalsh(RCCM_1).min(dim=1)[0].mean().item(), torch.linalg.eigvalsh(RCCM_1).max(dim=1)[0].mean().item(), torch.linalg.eigvalsh(RCCM_1).mean().item())
        print(torch.linalg.eigvalsh(RCCM_2).min(dim=1)[0].mean().item(), torch.linalg.eigvalsh(RCCM_2).max(dim=1)[0].mean().item(), torch.linalg.eigvalsh(RCCM_2).mean().item())
    if acc:
        return (loss, 
               ((torch.linalg.eigvalsh(Contraction, UPLO='L') >= 0).sum(dim=1) == 0).cpu().detach().numpy(), 
               ((torch.linalg.eigvalsh(C1_LHS_1, UPLO='L') >= 0).sum(dim=1) == 0).cpu().detach().numpy(), 
               sum([4.*(C2**2).reshape(bs,-1).sum(dim=1).mean() for C2 in C2s]).item(), 
               sum([1.*(C2w**2).reshape(bs,-1).sum(dim=1).mean() for C2w in C2ws]).item(), 
               ((torch.linalg.eigvalsh(RCCM_1, UPLO='L') >= 0).sum(dim=1) == 0).cpu().detach().numpy(),
               ((torch.linalg.eigvalsh(RCCM_2, UPLO='L') <= 0).sum(dim=1) == 0).cpu().detach().numpy()
               )
    else:
        return loss, None, None, None, None, None, None

optimizer = torch.optim.Adam([param_alpha, param_miu, param_lambda] + list(model_W.parameters()) + list(model_Wbot.parameters()) + list(model_u_w1.parameters()) + list(model_u_w2.parameters()), lr=learning_rate)

def trainval(X, bs=bs, train=True, acc=False, detach=False): # trainval a set of x
    # torch.autograd.set_detect_anomaly(True)

    if train:
        indices = np.random.permutation(len(X))
    else:
        indices = np.array(list(range(len(X))))

    total_loss = 0
    total_p1 = 0
    total_p2 = 0
    total_l3 = 0
    total_l4 = 0
    total_r5 = 0
    total_r6 = 0

    if train:
        _iter = tqdm(range(len(X) // bs))
    else:
        _iter = range(len(X) // bs)
    for b in _iter:
        start = time.time()
        x = []; xref = []; uref = []; w = []
        for id in indices[b*bs:(b+1)*bs]:
            if use_cuda:
                x.append(torch.from_numpy(X[id][0]).float().cuda())
                xref.append(torch.from_numpy(X[id][1]).float().cuda())
                uref.append(torch.from_numpy(X[id][2]).float().cuda())
                w.append(torch.from_numpy(X[id][3]).float().cuda())
            else:
                x.append(torch.from_numpy(X[id][0]).float())
                xref.append(torch.from_numpy(X[id][1]).float())
                uref.append(torch.from_numpy(X[id][2]).float())
                w.append(torch.from_numpy(X[id][3]).float())
        x, xref, uref, w = (torch.stack(d).detach() for d in (x, xref, uref, w))
        x = x.requires_grad_()

        loss, p1, p2, l3, l4, r5, r6 = forward(x, xref, uref, w, verbose=True if not train and b == 0 else False, acc=acc, detach=detach)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.shape[0]
        if acc:
            total_p1 += p1.sum()
            total_p2 += p2.sum()
            total_l3 += l3 * x.shape[0]
            total_l4 += l4 * x.shape[0]
            total_r5 += r5.sum()
            total_r6 += r6.sum()
    return total_loss / len(X), total_p1 / len(X), total_p2 / len(X), total_l3 / len(X), total_l4 / len(X), total_r5 / len(X), total_r6 / len(X)


best_acc = 0

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by every lr_step epochs"""
    lr = learning_rate * (0.1 ** (epoch // lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


for epoch in range(0, epochs):
    adjust_learning_rate(optimizer, epoch)
    loss, _, _, _, _, _, _ = trainval(X_tr, train=True, acc=False, detach=True if epoch < detach_epoch else False)
    print("Training loss: ", loss)
    loss, p1, p2, l3, l4, r5, r6 = trainval(X_te, train=False, acc=True, detach=False)
    print("Epoch %d: Testing loss/p1/p2/l3/l4/r5/r6: "%epoch, loss, p1, p2, l3, l4, r5, r6)  # Ideal: 0 / 1 / 1 / 0 / 0 / 1 / 1
    alpha_curr = F.softplus(param_alpha)
    miu_curr = F.softplus(param_miu)
    lambda_curr = F.softplus(param_lambda)
    print("alpha: %.3f, miu: %.3f, lambda: %.3f"%(alpha_curr.item(), miu_curr.item(), lambda_curr.item()))

    with open(log_record_file, 'a') as f:
        f.write(
            f"Epoch {epoch}: loss = {loss:.6f} \n"
            f"p1 = {p1:.6f}, p2 = {p2:.6f}, l3 = {l3:.6f}, "
            f"l4 = {l4:.6f}, r5 = {r5:.6f}, r6 = {r6:.6f}\n"
            f"alpha = {alpha_curr.item():.6f}, miu = {miu_curr.item():.6f}, lambda = {lambda_curr.item():.6f}\n"
        )

    if p1+p2 >= best_acc:
        best_acc = p1 + p2
        filename = log+'/model_best.pth.tar'
        filename_controller = log+'/controller_best.pth.tar'
        torch.save({
            'precs': (loss, p1, p2, l3, l4, r5, r6),
            'model_W': model_W.state_dict(),
            'model_Wbot': model_Wbot.state_dict(),
            'model_u_w1': model_u_w1.state_dict(),
            'model_u_w2': model_u_w2.state_dict(),
            'alpha': alpha_curr.item(),
            'miu': miu_curr.item(),
            'lambda': lambda_curr.item(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, filename)
        torch.save(u_func, filename_controller)
        
