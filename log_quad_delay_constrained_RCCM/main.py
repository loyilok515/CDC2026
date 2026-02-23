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
epochs = 45  # Number of training epochs
lr_step = 15  # Learning rate step
detach_epoch = 10  # Number of epochs to detach the computation graph
_lambda = 0.5  # Convergence rate: lambda
w_ub = 10  # Upper bound of the eigenvalue of the dual metric
w_lb = 0.1  # Lower bound of the eigenvalue of the dual metric

# Configuration variables
task = 'quad'  # Name of the model
log = 'log_quad_constrained'  # Path to a directory for storing the training log
use_cuda = True  # Set to False to disable CUDA

np.random.seed(1024)

# Ensure the log directory exists
os.makedirs(log, exist_ok=True)
# Copies all files into log for reference
os.system('cp *.py '+log)
os.system('cp -r models/ '+log)
os.system('cp -r configs/ '+log)
os.system('cp -r systems/ '+log)

epsilon = _lambda * 0.1  # Ensure definiteness of LMI

# Get training bounds from config file
config = importlib.import_module('config_'+task)
X_MIN = config.X_MIN
X_MAX = config.X_MAX
U_MIN = config.UREF_MIN
U_MAX = config.UREF_MAX
XE_MIN = config.XE_MIN
XE_MAX = config.XE_MAX

# Get system functions from system file
system = importlib.import_module('system_'+task)
S_func = system.S_func
f_func = system.f_func
B_func = system.B_func
E_func = system.E_func
Ebot_func = system.Ebot_func
num_dim_x = system.num_dim_x
num_dim_control = system.num_dim_control
num_dim_manifold = system.num_dim_manifold

# Get neural network configurations from model file
model = importlib.import_module('model_'+task)
get_model = model.get_model

model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func = get_model(num_dim_x, num_dim_manifold, num_dim_control, w_lb=w_lb, use_cuda=use_cuda)

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

def sample_full():
    xref = sample_xef()
    uref = sample_uref()
    x = sample_x(xref)
    return (x, xref, uref)

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

def forward(x, xref, uref, _lambda, verbose=False, acc=False, detach=False):
    # x: bs x n x 1
    bs = x.shape[0]
    W = W_func(x) # W: bs x num_dim_manifold x num_dim_manifold
    M = torch.inverse(W) # M: bs x num_dim_manifold x num_dim_manifold
    S = S_func(x)
    P_s = S # For specific systems
    P_s_T = P_s.transpose(1,2)
    f = f_func(x)
    B = B_func(x)
    E = E_func(x)
    _Ebot = Ebot_func(x)
    DfDx = Jacobian(f, x)
    DBDx = torch.zeros(bs, num_dim_x, num_dim_x, num_dim_control).type(x.type())
    for i in range(num_dim_control):
        DBDx[:,:,:,i] = Jacobian(B[:,:,i].unsqueeze(-1), x)

    u = u_func(x, xref, uref) # u: bs x m x 1
    K = Jacobian(u, x)

    A = DfDx + sum([u[:, i, 0].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i] for i in range(num_dim_control)])   
    dot_x = f + B.matmul(u)
    dot_M = weighted_gradients(M, dot_x, x, detach=detach) # DMDt
    dot_W = weighted_gradients(W, dot_x, x, detach=detach) # DWDt
    dot_P_s_T = weighted_gradients(P_s_T, dot_x, x, detach=detach)  # DPsTdt

    S_f = weighted_gradients(P_s_T, f, x).matmul(S) + P_s_T.matmul(DfDx).matmul(S)
    S_B = torch.zeros(bs, num_dim_manifold, num_dim_manifold, num_dim_control).type(x.type())
    for i in range(num_dim_control):
        S_B[:, :, :, i] = weighted_gradients(P_s_T, B[:, :, i].unsqueeze(-1), x).matmul(S) + P_s_T.matmul(DBDx[:, :, :, i]).matmul(S)   
    
    # Contraction condition (detach for first few epochs to let C1 and C2 find a CCM first)
    if detach:
        Contraction = dot_M + ((dot_P_s_T + P_s_T.matmul(A) + E.matmul(K)).matmul(S)).transpose(1,2).matmul(M.detach()) + M.detach().matmul((dot_P_s_T + P_s_T.matmul(A) + E.matmul(K)).matmul(S)) + 2 * _lambda * M.detach()
    else:
        Contraction = dot_M + ((dot_P_s_T + P_s_T.matmul(A) + E.matmul(K)).matmul(S)).transpose(1,2).matmul(M) + M.matmul((dot_P_s_T + P_s_T.matmul(A) + E.matmul(K)).matmul(S)) + 2 * _lambda * M

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

    loss = 0
    loss += loss_pos_matrix_random_sampling(-Contraction - epsilon * torch.eye(Contraction.shape[-1]).unsqueeze(0).type(x.type()))
    loss += loss_pos_matrix_random_sampling(-C1_LHS_1 - epsilon * torch.eye(C1_LHS_1.shape[-1]).unsqueeze(0).type(x.type()))
    loss += loss_pos_matrix_random_sampling(w_ub * torch.eye(W.shape[-1]).unsqueeze(0).type(x.type()) - W)
    loss += 4. * sum([1.*(C2**2).reshape(bs,-1).sum(dim=1).mean() for C2 in C2s])

    # Add a penalty for control inputs larger than 3
    control_penalty = torch.relu((u-uref).abs() - 10).sum(dim=1).mean()
    # loss += 1. * control_penalty

    if verbose:
        print(torch.symeig(Contraction)[0].min(dim=1)[0].mean(), torch.symeig(Contraction)[0].max(dim=1)[0].mean(), torch.symeig(Contraction)[0].mean())
    if acc:
        return loss, ((torch.linalg.eigvalsh(Contraction, UPLO='L') >= 0).sum(dim=1) == 0).cpu().detach().numpy(), ((torch.linalg.eigvalsh(C1_LHS_1, UPLO='L') >= 0).sum(dim=1) == 0).cpu().detach().numpy(), sum([1.*(C2**2).reshape(bs,-1).sum(dim=1).mean() for C2 in C2s]).item(), control_penalty.item()
    else:
        return loss, None, None, None, None

optimizer = torch.optim.Adam(list(model_W.parameters()) + list(model_Wbot.parameters()) + list(model_u_w1.parameters()) + list(model_u_w2.parameters()), lr=learning_rate)

def trainval(X, bs=bs, train=True, _lambda=_lambda, acc=False, detach=False): # trainval a set of x
    # torch.autograd.set_detect_anomaly(True)

    if train:
        indices = np.random.permutation(len(X))
    else:
        indices = np.array(list(range(len(X))))

    total_loss = 0
    total_p1 = 0
    total_p2 = 0
    total_l3 = 0
    total_c4 = 0

    if train:
        _iter = tqdm(range(len(X) // bs))
    else:
        _iter = range(len(X) // bs)
    for b in _iter:
        start = time.time()
        x = []; xref = []; uref = []; 
        for id in indices[b*bs:(b+1)*bs]:
            if use_cuda:
                x.append(torch.from_numpy(X[id][0]).float().cuda())
                xref.append(torch.from_numpy(X[id][1]).float().cuda())
                uref.append(torch.from_numpy(X[id][2]).float().cuda())
            else:
                x.append(torch.from_numpy(X[id][0]).float())
                xref.append(torch.from_numpy(X[id][1]).float())
                uref.append(torch.from_numpy(X[id][2]).float())

        x, xref, uref = (torch.stack(d).detach() for d in (x, xref, uref))
        x = x.requires_grad_()

        start = time.time()

        loss, p1, p2, l3, c4 = forward(x, xref, uref, _lambda=_lambda, verbose=False if not train else False, acc=acc, detach=detach)

        start = time.time()
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('backward(): %.3f s'%(time.time() - start))

        total_loss += loss.item() * x.shape[0]
        if acc:
            total_p1 += p1.sum()
            total_p2 += p2.sum()
            total_l3 += l3 * x.shape[0]
            total_c4 += c4 * x.shape[0]
    return total_loss / len(X), total_p1 / len(X), total_p2 / len(X), total_l3/ len(X), total_c4 / len(X)


best_acc = 0

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by every lr_step epochs"""
    lr = learning_rate * (0.1 ** (epoch // lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


for epoch in range(0, epochs):
    adjust_learning_rate(optimizer, epoch)
    loss, _, _, _, _ = trainval(X_tr, train=True, _lambda=_lambda, acc=False, detach=True if epoch < detach_epoch else False)
    print("Training loss: ", loss)
    loss, p1, p2, l3, c4 = trainval(X_te, train=False, _lambda=0., acc=True, detach=False)
    print("Epoch %d: Testing loss/p1/p2/l3/c4: "%epoch, loss, p1, p2, l3, c4)  # Ideal: 0 (total loss), 1 (probability of contraction condition being satisfied), 1 (probability of C1 being satisfied), 0 (mean of C2), 0 (mean of control penalty)

    if p1+p2 >= best_acc:
        best_acc = p1 + p2
        filename = log+'/model_best.pth.tar'
        filename_controller = log+'/controller_best.pth.tar'
        torch.save({
            'precs': (loss, p1, p2, l3, c4),
            'model_W': model_W.state_dict(),
            'model_Wbot': model_Wbot.state_dict(),
            'model_u_w1': model_u_w1.state_dict(),
            'model_u_w2': model_u_w2.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, filename)
        torch.save(u_func, filename_controller)
        
