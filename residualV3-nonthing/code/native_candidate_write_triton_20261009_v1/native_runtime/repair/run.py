import hashlib
import torch

def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def feature_signature(weight,basis):
    ids=torch.tensor([0,1,42,weight.shape[0]//2,151669,weight.shape[0]-1],device=weight.device)
    return dict(ids=ids.cpu().tolist(),weight_rows_sha256=digest(weight[ids]),basis_rows_sha256=digest(basis[ids]))
