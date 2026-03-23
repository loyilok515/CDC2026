import torch

# Load the .pth.tar file
checkpoint = torch.load('log_quad_constrained_RCCM/model_best.pth.tar', weights_only=False)

# Print the keys in the checkpoint
epoch = checkpoint['epoch']
precs = checkpoint['precs']
alpha = checkpoint['alpha']
miu = checkpoint['miu']
lambda_ = checkpoint['lambda']
print("Loaded model with loss/p1/p2/l3/c4: ", precs)
print("Epoch: ", epoch)
print("alpha: ", alpha)
print("miu: ", miu)
print("lambda: ", lambda_)