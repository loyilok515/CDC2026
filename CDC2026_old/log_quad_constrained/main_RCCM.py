import torch
from torch.autograd import grad
import torch.nn.functional as F

import importlib
import numpy as np
import time
from tqdm import tqdm

import os
import sys
sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')
import argparse

np.random.seed(1024)

parser = argparse.ArgumentParser(description="")
parser.add_argument('--task', type=str,
                        default='CAR', help='Name of the model.')
parser.add_argument('--no_cuda', dest='use_cuda', action='store_false', help='Disable cuda.')
parser.set_defaults(use_cuda=True)
parser.add_argument('--bs', type=int, default=1024, help='Batch size.')
parser.add_argument('--num_train', type=int, default=131072, help='Number of samples for training.') # 4096 * 32
parser.add_argument('--num_test', type=int, default=32768, help='Number of samples for testing.') # 1024 * 32
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='Base learning rate.')
parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
parser.add_argument('--lr_step', type=int, default=5, help='')
parser.add_argument('--lambda', type=float, dest='_lambda', default=0.5, help='Convergence rate: lambda')
parser.add_argument('--w_ub', type=float, default=10, help='Upper bound of the eigenvalue of the dual metric.')
parser.add_argument('--w_lb', type=float, default=0.1, help='Lower bound of the eigenvalue of the dual metric.')
parser.add_argument('--log', type=str, help='Path to a directory for storing the log.')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from.')
parser.add_argument('--alpha_lb', type=float, dest='alpha_lb', default=0.5, help='Noise to deviation gain')
parser.add_argument('--miu_lb', type=float, dest='miu_lb', default=0.25, help='Tube radius: miu/lambda')
parser.add_argument('--beta_RCCM_2', type=float, dest='beta_RCCM_2', default=2.0, help='Weight for RCCM_2 loss')
parser.add_argument('--beta_update', type=float, dest='beta_update', default=2.0, help='penalty update rate for beta_RCCM_2')

args = parser.parse_args()

os.system('cp *.py '+args.log)
os.system('cp -r models/ '+args.log)
os.system('cp -r configs/ '+args.log)
os.system('cp -r systems/ '+args.log)

epsilon = args._lambda * 0.1

config = importlib.import_module('config_'+args.task)
X_MIN = config.X_MIN
X_MAX = config.X_MAX
U_MIN = config.UREF_MIN
U_MAX = config.UREF_MAX
XE_MIN = config.XE_MIN
XE_MAX = config.XE_MAX
w_MIN = config.w_MIN
w_MAX = config.w_MAX

system = importlib.import_module('system_'+args.task)
f_func = system.f_func
B_func = system.B_func
B_w_func = system.B_w_func
C_func = system.C_func
D_func = system.D_func
num_dim_x = system.num_dim_x
num_dim_control = system.num_dim_control
num_dim_noise = system.num_dim_noise
num_dim_z = system.num_dim_z
if hasattr(system, 'Bbot_func'):
    Bbot_func = system.Bbot_func

model = importlib.import_module('model_'+args.task)
get_model = model.get_model

model_W, model_Wbot, model_u_w1, model_u_w2, param_alpha, param_miu, W_func, u_func = get_model(num_dim_x, num_dim_control, w_lb=args.w_lb, use_cuda=args.use_cuda)

# constructing datasets
def sample_xef():
    return (X_MAX-X_MIN) * np.random.rand(num_dim_x, 1) + X_MIN

def sample_x(xref):
    xe = (XE_MAX-XE_MIN) * np.random.rand(num_dim_x, 1) + XE_MIN
    x = xref + xe
    x[x>X_MAX] = X_MAX[x>X_MAX]
    x[x<X_MIN] = X_MIN[x<X_MIN]
    return x

def sample_uref():
    return (U_MAX-U_MIN) * np.random.rand(num_dim_control, 1) + U_MIN

def sample_w():
    return (w_MAX-w_MIN) * np.random.rand(num_dim_noise, 1) + w_MIN

def sample_full():
    xref = sample_xef()
    uref = sample_uref()
    x = sample_x(xref)
    w = sample_w()
    return (x, xref, uref, w)

X_tr = [sample_full() for _ in range(args.num_train)]
X_te = [sample_full() for _ in range(args.num_test)]

if 'Bbot_func' not in locals():
    def Bbot_func(B): # columns of Bbot forms a basis of the null space of B^T
        bs = B.shape[0]
        Bbot = torch.cat((torch.eye(num_dim_x-num_dim_control, num_dim_x-num_dim_control),
            torch.zeros(num_dim_control, num_dim_x-num_dim_control)), dim=0)
        if args.use_cuda:
            Bbot = Bbot.cuda()
        Bbot.unsqueeze(0)
        return Bbot.repeat(bs, 1, 1)

def Jacobian_Matrix(M, x):
    # NOTE that this function assume that data are independent of each other
    # along the batch dimension.
    # M: B x m x m
    # x: B x n x 1
    # ret: B x m x m x n
    bs = x.shape[0]
    m = M.size(-1)
    n = x.size(1)
    J = torch.zeros(bs, m, m, n).type(x.type())
    for i in range(m):
        for j in range(m):
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

K = 1024
# K = 4096  # For more states?
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
    # eigv = torch.symeig(A, eigenvectors=True)[0].view(-1)
    eigv = torch.linalg.eigvalsh(A, UPLO='L').view(-1)
    negative_index = eigv.detach().cpu().numpy() < 0
    negative_eigv = eigv[negative_index]
    return negative_eigv.norm()

def forward(x, xref, uref, w, _lambda, verbose=False, acc=False, detach=False):

    # x: bs x n x 1
    bs = x.shape[0]
    W = W_func(x)
    M = torch.inverse(W)
    f = f_func(x)
    B = B_func(x)
    B_w = B_w_func(x)
    C = C_func(x)
    D = D_func(x)
    DfDx = Jacobian(f, x)
    DBDx = torch.zeros(bs, num_dim_x, num_dim_x, num_dim_control).type(x.type())
    for i in range(num_dim_control):
        DBDx[:,:,:,i] = Jacobian(B[:,:,i].unsqueeze(-1), x)
    DB_wDx = torch.zeros(bs, num_dim_x, num_dim_x, num_dim_noise).type(x.type())
    _Bbot = Bbot_func(B)
    # _Bbot = Bbot_func(x)  # Bbot: bs x n x (n-m)  # For usual training
    u = u_func(x, x - xref, uref) # u: bs x m x 1 # TODO: x - xref
    K = Jacobian(u, x)

    # compute RCCM parameters (force positive constants)
    _alpha = F.softplus(param_alpha)
    _miu = F.softplus(param_miu)

    A = DfDx + sum([u[:, i, 0].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i] for i in range(num_dim_control)]) + sum([w[:, i, 0].unsqueeze(-1).unsqueeze(-1) * DB_wDx[:, :, :, i] for i in range(num_dim_noise)])
    dot_x = f + B.matmul(u) + B_w.matmul(w)
    dot_M = weighted_gradients(M, dot_x, x, detach=detach) # DMDt
    dot_W = weighted_gradients(W, dot_x, x, detach=detach) # DWDt

    if detach:
        Contraction = dot_M + M.detach().matmul(A + B.matmul(K)) + (A + B.matmul(K)).transpose(1,2).matmul(M.detach()) + 2 * _lambda * M.detach()
        Contraction_dual = - dot_W + (A + B.matmul(K)).matmul(W.detach()) + W.detach().matmul((A + B.matmul(K)).transpose(1,2)) + 2 * _lambda * W.detach()
        RCCM_2_11 = 2 * _lambda * M.detach() # W
    else:
        Contraction = dot_M + M.matmul(A + B.matmul(K)) + (A + B.matmul(K)).transpose(1,2).matmul(M) + 2 * _lambda * M
        Contraction_dual = - dot_W + (A + B.matmul(K)).matmul(W) + W.matmul((A + B.matmul(K)).transpose(1,2)) + 2 * _lambda * W
        RCCM_2_11 = 2 * _lambda * M # W
 

    RCCM_1_11 = Contraction_dual
    RCCM_1_12 = B_w
    RCCM_1_21 = B_w.transpose(1,2)
    RCCM_1_22 = - _miu * torch.eye(num_dim_noise).repeat(bs, 1, 1).type(x.type())

    
    RCCM_2_21 = torch.zeros(bs, num_dim_noise, num_dim_x).type(x.type())
    RCCM_2_22 = (_alpha - _miu) * torch.eye(num_dim_noise).repeat(bs, 1, 1).type(x.type())
    RCCM_2_31 = (C + D.matmul(K))#.matmul(W)
    RCCM_2_32 = torch.zeros(bs, (num_dim_z), num_dim_noise).type(x.type())
    RCCM_2_33 = _alpha * torch.eye(num_dim_z).repeat(bs, 1, 1).type(x.type())
    RCCM_2_12 = RCCM_2_21.transpose(1,2)
    RCCM_2_13 = RCCM_2_31.transpose(1,2)
    RCCM_2_23 = RCCM_2_32.transpose(1,2)

    R2_11 = RCCM_2_11 - ((C + D.matmul(K)).transpose(1,2)).matmul(C + D.matmul(K)) / _alpha
    R2_12 = torch.zeros(bs, num_dim_x, num_dim_noise).type(x.type())
    R2_21 = R2_12.transpose(1,2)
    R2_22 = (_alpha - _miu) * torch.eye(num_dim_noise).repeat(bs, 1, 1).type(x.type())
    
    # RCCM_1 (Condition (15))
    RCCM_1 = torch.cat([
        torch.cat([RCCM_1_11, RCCM_1_12], dim=2),
        torch.cat([RCCM_1_21, RCCM_1_22], dim=2)
    ], dim=1)
    
    # RCCM_2 (Condition (16))
    RCCM_2 = torch.cat([
        torch.cat([RCCM_2_11, RCCM_2_12, RCCM_2_13], dim=2), 
        torch.cat([RCCM_2_21, RCCM_2_22, RCCM_2_23], dim=2), 
        torch.cat([RCCM_2_31, RCCM_2_32, RCCM_2_33], dim=2)
    ], dim=1)
    
    # R2 (Condition (16), Schur complement of RCCM_2)
    R2 = torch.cat([
        torch.cat([R2_11, R2_12], dim=2),
        torch.cat([R2_21, R2_22], dim=2)
    ], dim=1)

    # C1
    C1_inner = - weighted_gradients(W, f, x) + DfDx.matmul(W) + W.matmul(DfDx.transpose(1,2)) + 2 * _lambda * W
    C1_LHS_1 = _Bbot.transpose(1,2).matmul(C1_inner).matmul(_Bbot) # this has to be a negative definite matrix

    # C2
    C2_inners = []
    C2s = []
    for j in range(num_dim_control):
        C2_inner = weighted_gradients(W, B[:,:,j].unsqueeze(-1), x) - (DBDx[:,:,:,j].matmul(W) + W.matmul(DBDx[:,:,:,j].transpose(1,2)))
        C2 = _Bbot.transpose(1,2).matmul(C2_inner).matmul(_Bbot)
        C2_inners.append(C2_inner)
        C2s.append(C2)

    loss = 0
    loss_1 = loss_pos_matrix_random_sampling(-Contraction - epsilon * torch.eye(Contraction.shape[-1]).unsqueeze(0).type(x.type()))
    loss_2 = loss_pos_matrix_random_sampling(-RCCM_1 - epsilon * torch.eye(RCCM_1.shape[-1]).unsqueeze(0).type(x.type()))
    # loss_3 = loss_pos_matrix_eigen_values(RCCM_2 - epsilon * torch.eye(RCCM_2.shape[-1]).unsqueeze(0).type(x.type()))
    loss_3 = loss_pos_matrix_eigen_values(R2 - epsilon * torch.eye(R2.shape[-1]).unsqueeze(0).type(x.type()))
    loss_4 = loss_pos_matrix_random_sampling(-C1_LHS_1 - epsilon * torch.eye(C1_LHS_1.shape[-1]).unsqueeze(0).type(x.type()))
    loss_5 = loss_pos_matrix_random_sampling(args.w_ub * torch.eye(W.shape[-1]).unsqueeze(0).type(x.type()) - W)
    loss_6 = 1. * sum([1.*(C2**2).reshape(bs,-1).sum(dim=1).mean() for C2 in C2s])
    
    loss += loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6
    print(loss_1.item(), loss_2.item(), loss_3.item(), loss_4.item(), loss_5.item(), loss_6.item())

    # Add a penalty for control inputs larger than u_max
    u_max = 0
    control_penalty = torch.relu(u.abs() - u_max).sum(dim=1).mean()

    # Add alpha penalty to minimize gain for disturbance
    alpha_penalty = torch.relu(_alpha - args.alpha_lb)
    loss += alpha_penalty
    

    if verbose:
        print(torch.linalg.eigvalsh(Contraction).min(dim=1)[0].mean().item(), torch.linalg.eigvalsh(Contraction).max(dim=1)[0].mean().item(), torch.linalg.eigvalsh(Contraction).mean().item())
        print(torch.linalg.eigvalsh(RCCM_1).min(dim=1)[0].mean().item(), torch.linalg.eigvalsh(RCCM_1).max(dim=1)[0].mean().item(), torch.linalg.eigvalsh(RCCM_1).mean().item())
        print(torch.linalg.eigvalsh(RCCM_2).min(dim=1)[0].mean().item(), torch.linalg.eigvalsh(RCCM_2).max(dim=1)[0].mean().item(), torch.linalg.eigvalsh(RCCM_2).mean().item())

    if acc:
        return loss, ((torch.linalg.eigvalsh(Contraction, UPLO='L') >= 0).sum(dim=1) == 0).cpu().detach().numpy(), ((torch.linalg.eigvalsh(RCCM_1, UPLO='L') >= 0).sum(dim=1) == 0).cpu().detach().numpy(), ((torch.linalg.eigvalsh(RCCM_2, UPLO='L') <= 0).sum(dim=1) == 0).cpu().detach().numpy(), ((torch.linalg.eigvalsh(C1_LHS_1, UPLO='L') >= 0).sum(dim=1) == 0).cpu().detach().numpy(), sum([1.*(C2**2).reshape(bs,-1).sum(dim=1).mean() for C2 in C2s]).item(), control_penalty.item()
    else:
        return loss, None, None, None, None, None, None

optimizer = torch.optim.Adam([param_alpha, param_miu] + list(model_W.parameters()) + list(model_Wbot.parameters()) + list(model_u_w1.parameters()) + list(model_u_w2.parameters()), lr=args.learning_rate)

def trainval(X, bs=args.bs, train=True, _lambda=args._lambda, acc=False, detach=False): # trainval a set of x
    # torch.autograd.set_detect_anomaly(True)

    if train:
        indices = np.random.permutation(len(X))
    else:
        indices = np.array(list(range(len(X))))

    total_loss = 0
    total_p1 = 0
    total_p1_1 = 0
    total_p1_2 = 0
    total_p2 = 0
    total_l3 = 0
    total_c4 = 0

    if train:
        _iter = tqdm(range(len(X) // bs))
    else:
        _iter = range(len(X) // bs)
    for b in _iter:
        start = time.time()
        x = []; xref = []; uref = []; w = []; 
        for id in indices[b*bs:(b+1)*bs]:
            if args.use_cuda:
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
        start = time.time()

        loss, p1, p1_1, p1_2, p2, l3, c4 = forward(x, xref, uref, w, _lambda=_lambda, verbose=True if not train else False, acc=acc, detach=detach)

        start = time.time()
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('backward(): %.3f s'%(time.time() - start))

        total_loss += loss.item() * x.shape[0]
        if acc:
            total_p1 += p1.sum()
            total_p1_1 += p1_1.sum()
            total_p1_2 += p1_2.sum()
            total_p2 += p2.sum()
            total_l3 += l3 * x.shape[0]
            total_c4 += c4 * x.shape[0]
    return total_loss / len(X), total_p1 / len(X), total_p1_1 / len(X), total_p1_2 / len(X), total_p2 / len(X), total_l3 / len(X), total_c4 / len(X)


best_acc = 0

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by every args.lr_step epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // args.lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


start_epoch = 0
if args.resume is not None:
    print(f"Resuming from checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')
    model_W.load_state_dict(checkpoint['model_W'])
    model_Wbot.load_state_dict(checkpoint['model_Wbot'])
    model_u_w1.load_state_dict(checkpoint['model_u_w1'])
    model_u_w2.load_state_dict(checkpoint['model_u_w2'])
    param_alpha.data = checkpoint['param_alpha']
    param_miu.data = checkpoint['param_miu']
    if args.use_cuda:
        param_alpha.data = param_alpha.data.cuda()
        param_miu.data = param_miu.data.cuda()
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    precs = checkpoint['precs']
    print("Loaded model with loss/p1_1/p1_2/p2/l3/c4: ", precs)

for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch)
    beta_RCCM_2 = args.beta_RCCM_2 * (args.beta_update * (epoch // args.lr_step + 1))
    loss, _, _, _, _, _, _ = trainval(X_tr, train=True, _lambda=args._lambda, acc=False, detach=True if epoch < args.lr_step else False)
    print("Training loss: ", loss)
    loss, p1, p1_1, p1_2, p2, l3, c4 = trainval(X_te, train=False, _lambda=args._lambda, acc=True, detach=False)
    print("Epoch %d: Testing loss/p1/p1_1/p1_2/p2/l3/c4: "%epoch, loss, p1, p1_1, p1_2, p2, l3, c4)  # Ideal: 0 (total loss), 1, 1 (probability of contraction conditions being satisfied), 1 (probability of C1 being satisfied), 0 (mean of C2), 0 (mean of control penalty)
    
    alpha = F.softplus(param_alpha)
    miu = F.softplus(param_miu)
    print(alpha.item(), miu.item())

    if p1+p1_1+p1_2+p2 >= best_acc:
        best_acc = p1 + p1_1 + p1_2 + p2
        filename = args.log+'/model_best.pth.tar'
        filename_controller = args.log+'/controller_best.pth.tar'
        torch.save({
            'args': args,
            'precs': (loss, p1, p1_1, p1_2, p2, l3, c4),
            'model_W': model_W.state_dict(),
            'model_Wbot': model_Wbot.state_dict(),
            'model_u_w1': model_u_w1.state_dict(),
            'model_u_w2': model_u_w2.state_dict(),
            'alpha': alpha.item(),
            'miu': miu.item(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, filename)
        torch.save(u_func, filename_controller)
        
