import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from utils import block_matching, gather_groups, aggregate, split2d, estimate_patch_size
import time

def __less2d_solve(Z: torch.Tensor, cuda: bool) -> tuple[torch.Tensor, torch.Tensor]:
    N, Href, Wref, k, n = Z.shape
    Z_dtype = Z.dtype

    R = torch.matmul(Z, Z.transpose(-1, -2))
    theta = torch.zeros(N, Href, Wref, k, k, dtype=Z_dtype)

    for j in range(k):
        idx_excl = list(range(j)) + list(range(j + 1, k))
        R_excl = R[..., idx_excl, :][..., idx_excl]
        z_j_excl = R[..., j][..., idx_excl].unsqueeze(-1)
        if cuda:
            beta_j = torch.linalg.lstsq(R_excl.cuda(), z_j_excl.cuda()).solution.cpu()
        else:
            beta_j = torch.linalg.lstsq(R_excl, z_j_excl).solution
        theta[..., idx_excl, j] = beta_j.squeeze(-1)

    theta = theta.transpose(-1, -2)
    Z_hat = torch.matmul(theta, Z)
    weights = 1 / torch.sum(theta ** 2, dim=-1, keepdim=True).clip(1 / k, 1)
    return Z_hat, weights

def __less2d_step(input_tensor: torch.Tensor, k: int, p: int, w: int, s: int, cuda: bool) -> torch.Tensor:
    _, C, H, W = input_tensor.size()
    grayscale_tensor = torch.mean(input_tensor, dim=1, keepdim=True) if C != 1 else input_tensor
    indices = block_matching(grayscale_tensor, k, p, w, s)
    Z = gather_groups(input_tensor, indices, p)
    Z_hat, weights = __less2d_solve(Z, cuda)
    z_hat = aggregate(Z_hat, weights, indices, H, W, p)
    return z_hat

def _less2d_denoise(input_tensor: torch.Tensor, cuda: bool, k: int = 18, p: int = 5, w: int = 37, s: int = 4) -> torch.Tensor:
    return __less2d_step(input_tensor, k, p, w, s, cuda)

def _less1d_denoise(input_vector: torch.Tensor, max_k: int = 1000, pat: int = 5) -> torch.Tensor:
    x = input_vector.view(1, 1, -1)
    l = x.shape[-1]
    min_loss = float('inf')
    best_k = 3
    counter = pat

    for k in range(3, max_k):
        if k % 2 == 0 or k > l:
            continue
        pad = k // 2
        x_pad = F.pad(x, (pad, pad), mode='reflect')
        windows = x_pad.unfold(-1, k, 1)
        mask = torch.ones(k, dtype=torch.bool)
        mask[k // 2] = 0
        windows_bs = windows[..., mask]
        denoised = windows_bs.median(dim=-1).values.squeeze()
        loss = torch.mean((denoised - x.squeeze()) ** 2).item()
        if loss < min_loss:
            min_loss = loss
            best_k = k
            counter = pat
        else:
            counter -= 1
        if counter <= 0:
            break

    pad = best_k // 2
    x_pad = F.pad(x, (pad, pad), mode='reflect')
    windows = x_pad.unfold(-1, best_k, 1)
    denoised = windows.median(dim=-1).values.squeeze()
    return denoised

def compute_loss(y_tensor, x_hat_tensor):
    y_list = split2d(y_tensor)
    x_hat_list = split2d(x_hat_tensor)
    loss = 0.
    valid_pairs = 0
    adjacency = {
        0: [1, 2, 3],
        1: [0, 2, 3],
        2: [0, 1, 3],
        3: [0, 1, 2]
    }
    for i in range(4):
        f_i = x_hat_list[i]
        y_i = y_list[i]
        for j in adjacency[i]:
            y_j = y_list[j]
            _loss = f_i**2 - 2*y_j*f_i + y_i*y_j
            loss += _loss.mean()
            valid_pairs += 1
    return loss / valid_pairs

def compute_ortho_loss(y_tensor, x_hat_tensor):
    y_list = split2d(y_tensor)
    x_hat_list = split2d(x_hat_tensor)
    loss = 0.
    valid_pairs = 0
    adjacency = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3],
        3: [1, 2]
    }
    for i in range(4):
        f_i = x_hat_list[i]
        y_i = y_list[i]
        for j in adjacency[i]:
            y_j = y_list[j]
            _loss = f_i**2 - 2*y_j*f_i + y_i*y_j
            loss += _loss.mean()
            valid_pairs += 1
    return loss / valid_pairs

def denoise(
    data: torch.Tensor, 
    gt: Optional[torch.Tensor] = None,
    cuda: bool = True,
    pat: int = 10,
    verbose: bool = True,
    patch_size: Optional[int] = None, 
    top_k: int = 18, 
    window_size: int = 37, 
    stride: int = 4,
    verbose_prefix: str = "",
    return_loss: bool = False,
    ):
    assert data.ndim == 3, "Input data must be 3D (C, H, W)"
    if gt is not None:
        assert data.shape == gt.shape, "Input data and ground truth must have the same shape"
    
    if patch_size is None:
        patch_size = estimate_patch_size(data)
        print(f'Estimate patch_size={patch_size} from given data.')

    c, h, w = data.shape
    m = h * w
    y_tensor = data.view(1, c, h, w)
    Y = data.view(c, m).T
    X = gt.view(c, m).T if gt is not None else None

    if cuda:
        Y = Y.cuda()
        if X is not None:
            X = X.cuda()

    if verbose:
        print(f"{verbose_prefix}Start decomp...")
        _start_time = time.time()

    U, S, Vh = torch.linalg.svd(Y, full_matrices=False)

    if verbose:
        print(f"{verbose_prefix}Decomp. time: {time.time() - _start_time:.4f} s")

    if cuda:
        U, S, Vh = U.cpu(), S.cpu(), Vh.cpu()

    num_basis = S.shape[0]
    X_hat_opt = None
    i_opt = None
    X_hat = torch.zeros_like(Y, dtype=Y.dtype)
    mse_list = []
    loss_list = []

    for i in range(num_basis):
        vh = Vh[i, :]
        vh_hat = _less1d_denoise(vh.cuda()) if cuda else _less1d_denoise(vh)
        vh_hat = vh_hat.reshape(1, -1)
        if not cuda:
            vh_hat = vh_hat.cpu()

        u = U[:, i].view(1, 1, h, w)
        u_hat = _less2d_denoise(input_tensor=u, cuda=cuda, p=patch_size, k=top_k, w=window_size, s=stride).reshape(-1, 1)
        if cuda:
            u_hat = u_hat.cuda()

        s = S[i].item()
        x_hat = s * torch.matmul(u_hat, vh_hat)
        X_hat = X_hat + x_hat

        if X is not None:
            mse = F.mse_loss(X, X_hat, reduction='none').mean().item()
            mse_list.append(mse)
        else:
            mse = None

        loss = compute_loss(y_tensor, X_hat.T.view(1, c, h, w)).item()
        loss_list.append(loss)

        if verbose:
            msg_str = f"{verbose_prefix}Accumulating {i+1}-th basis, loss: {loss:.4f}" if mse is None else f"Accumulating {i+1}-th basis, mse: {mse:.4f}, loss: {loss:.4f}"
            print(msg_str)

        if i == 0:
            X_hat_opt = X_hat
            i_opt = i
            counter = pat
        else:
            if loss < np.min(loss_list[:-1]):
                X_hat_opt = X_hat
                counter = pat
                i_opt = i
            else:
                counter -= 1
            if counter <= 0:
                if verbose:
                    if i_opt == 0:
                        print(f"{verbose_prefix}Early stopping at {i+1}-th basis, top {i+1} bases are included in final denoised result.")
                    else:
                        print(f"{verbose_prefix}Early stopping at {i+1}-th basis, top {i_opt+1} bases are included in final denoised result.")
                        if len(mse_list) > 0:
                            print(f"{verbose_prefix}min mse: {np.min(mse_list):.4f} with top {np.argmin(mse_list) + 1} bases, min loss: {np.min(loss_list):.4f} with top {np.argmin(loss_list) + 1} bases")
                break

    if return_loss:
        return X_hat_opt.T.view(c, h, w), loss_list, mse_list
    return X_hat_opt.T.view(c, h, w)
