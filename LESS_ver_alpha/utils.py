from tifffile import TiffFile
import numpy as np
from typing import Optional
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import random
import sys
# import torchvision

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def one_phase_decay(t, Y0, Plateau, K):
    """
    One-phase decay model.

    Parameters:
        t (array-like): Time or independent variable.
        Y0 (float): Initial value at t=0.
        Plateau (float): The asymptotic value as t approaches infinity.
        K (float): Rate constant.

    Returns:
        array-like: Dependent variable values at each time point.
    """
    return Plateau + (Y0 - Plateau) * np.exp(-K * t)


def estimate_patch_size(data: torch.Tensor):
    """
    data: [C, H, W]
    """ 
    mean_proj = data.mean(0, keepdims=True) # [1, H, W]

    relative_avg_var = ((data - mean_proj)/mean_proj).var(0).mean().item()

    # use the fit params
    # Define known parameters
    Y0 = 5.666       # Initial value
    Plateau = 14.50  # Asymptotic value
    K = 0.2070       # Decay rate constant

    # Compute the decay curve
    est_patch_size = one_phase_decay(relative_avg_var, Y0, Plateau, K)

    # cast to closed odd int
    est_patch_size = int(np.floor(est_patch_size))
    if est_patch_size % 2 == 0:
        est_patch_size += 1

    est_patch_size = max(est_patch_size, 5)

    return est_patch_size


def print_info(args):
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    # print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))


def split_data(data, s):
    """
    Split a 3D array [C, H, W] into groups based on 2D indices.

    Args:
        data (np.ndarray): The input array of shape [C, H, W].
        s (int): Step size for splitting.

    Returns:
        tuple: (groups, h_indices, w_indices)
            - groups: List of split groups, each of shape [C, h_group, w_group].
            - h_indices: List of row indices for each group.
            - w_indices: List of column indices for each group.
    """
    C, H, W = data.shape
    groups = []
    h_indices = []
    w_indices = []
    
    for offset_h in range(s):
        for offset_w in range(s):
            # Generate indices
            h_idx = np.arange(offset_h, H, s)
            w_idx = np.arange(offset_w, W, s)
            
            # Save indices
            h_indices.append(h_idx)
            w_indices.append(w_idx)
            
            # Extract group data
            group = data[:, h_idx[:, None], w_idx]
            groups.append(group)
    
    return groups, h_indices, w_indices


def split_indices(c, h, w, s):
    """
    Split a 3D array [C, H, W] into groups based on 2D indices.

    Args:
        data (np.ndarray): The input array of shape [C, H, W].
        s (int): Step size for splitting.

    Returns:
        tuple: (groups, h_indices, w_indices)
            - groups: List of split groups, each of shape [C, h_group, w_group].
            - h_indices: List of row indices for each group.
            - w_indices: List of column indices for each group.
    """
    h_indices = []
    w_indices = []
    
    indices = []
    i = 0
    for offset_h in range(s):
        for offset_w in range(s):
            # Generate indices
            h_idx = np.arange(offset_h, h, s)
            w_idx = np.arange(offset_w, w, s)
            
            # Save indices
            h_indices.append(h_idx)
            w_indices.append(w_idx)

            # print(len(h_idx), len(w_idx))
            
            # Extract group data
            # group = data[:, h_idx[:, None], w_idx]
            # groups.append(group)

            indices.append(i)
            i = i+1
    
    return h_indices, w_indices, indices

def merge_groups(groups, H, W, h_indices, w_indices):
    """
    Merge split groups back into the original array shape [C, H, W].

    Args:
        groups (list of np.ndarray): List of split groups, each of shape [C, h_group, w_group].
        H (int): Height of the original array.
        W (int): Width of the original array.
        h_indices (list of np.ndarray): List of row indices for each group.
        w_indices (list of np.ndarray): List of column indices for each group.

    Returns:
        np.ndarray: Merged array of shape [C, H, W].
    """
    # Initialize an empty array
    C = groups[0].shape[0]  # Number of channels
    merged = np.zeros((C, H, W))
    
    # Iterate over groups and merge them
    for group, h_idx, w_idx in zip(groups, h_indices, w_indices):
        for i, hi in enumerate(h_idx):
            for j, wj in enumerate(w_idx):
                merged[:, hi, wj] = group[:, i, j]  # Directly assign values since no overlap

    return merged


def merge_groups_new(groups, H, W, h_indices, w_indices):
    """
    Efficiently merge split groups back into the original array shape [C, H, W].

    Args:
        groups (list of np.ndarray): List of split groups, each of shape [C, h_group, w_group].
        H (int): Height of the original array.
        W (int): Width of the original array.
        h_indices (list of np.ndarray): List of row indices for each group.
        w_indices (list of np.ndarray): List of column indices for each group.

    Returns:
        np.ndarray: Merged array of shape [C, H, W].
    """
    C = groups[0].shape[0]
    merged = np.zeros((C, H, W), dtype=groups[0].dtype)

    for group, h_idx, w_idx in zip(groups, h_indices, w_indices):
        # Use np.ix_ to create a meshgrid for advanced indexing
        merged[:, np.ix_(h_idx, w_idx)] = group

    return merged

def get_shape_dtype_itemsize(filepath: str) -> list:
    """
    Get metadata from a TIFF file, including spatial dimensions and number of frames.
    Args:
        filepath (str): Path to the TIFF file.
    Returns:
        list: A list containing the number of frames and spatial dimensions (height, width).
    """ 

    with TiffFile(filepath) as tif:
        # Get shape: (frames, height, width) or (height, width) for single frame
        num_frames = len(tif.pages)
        data = tif.pages[0].asarray()
        dtype = data.dtype
        itemsize = data.itemsize
        shape = list(tif.pages[0].shape)
        data = [num_frames] + shape
        return tuple(data), dtype, itemsize

def load_frames(filepath: str, start_index: Optional[int] = None, end_index:Optional[int] = None) -> np.ndarray:
    """
    Load frames from a TIFF file.
    Args:
        filepath (str): Path to the TIFF file.
        start_index (int, optional): Starting index of frames to load. If None, starts from the first frame.
        end_index (int, optional): Ending index of frames to load. If None, loads until the last frame.
    Returns:
        np.ndarray: A numpy array containing the loaded frames.
    """
    if start_index is not None and end_index is not None:
        assert start_index < end_index, "start_index must be less than end_index"
        
    with TiffFile(filepath) as tif:
        if start_index is not None and end_index is not None:
            total_frames = len(tif.pages)
            assert start_index < total_frames, "start_index exceeds number of frames in the TIFF file"
            assert end_index <= total_frames, "end_index exceeds number of frames in the TIFF file"

            indices = range(start_index, end_index)
        else:
            indices = range(len(tif.pages))
        # Load specified frames
        frames = [tif.pages[i].asarray() for i in indices]
        return np.stack(frames)

def block_matching(input_x, k, p, w, s):
    """
    Finds similar patches within a specified window around each reference patch.

    Args:
        input_x (torch.FloatTensor): Input image tensor of shape (N, C, H, W).
        k (int): Number of most similar patches to find.
        p (int): Patch size.
        w (int): Search window size.
        s (int): Stride for moving between reference patches.

    Returns:
        torch.LongTensor: Indices of shape (N, Href, Wref, k) of similar patches for each reference patch.
    """
    if w % 2 != 1:
        raise ValueError(f"Invalid input: w ({w}) must be an odd integer.")
    if (w - p + 1) ** 2 < k:
        raise ValueError(
            f"Invalid input: k ({k}) must be less than or equal to the number of overlapping patches per window, that is {(w - p + 1) ** 2}.")

    def block_matching_aux(input_x_pad, k, p, v, s):
        """
        Auxiliary function to perform block matching in a padded input tensor.

        Args:
            input_x_pad (torch.FloatTensor): Padded input tensor of shape (N, C, H, W).
            k (int): Number of similar patches to find.
            p (int): Patch size.
            v (int): Half of the search window size.
            s (int): Stride for moving between reference patches.

        Returns:
            torch.LongTensor: Indices of shape (N, Href, Wref, k) of similar patches for each reference patch.
        """
        N, C, H, W = input_x_pad.size()
        assert C == 1
        Href, Wref = -((H - (2 * v + p) + 1) // -s), -((W - (
                    2 * v + p) + 1) // -s)  # ceiling division, represents the number of reference patches along each axis for unfold with stride=s
        norm_patches = F.avg_pool2d(input_x_pad ** 2, p, stride=1)
        norm_patches = F.unfold(norm_patches, 2 * v + 1, stride=s)
        norm_patches = rearrange(norm_patches, 'n (p1 p2) l -> 1 (n l) p1 p2', p1=2 * v + 1)
        local_windows = F.unfold(input_x_pad, 2 * v + p, stride=s) / p
        local_windows = rearrange(local_windows, 'n (p1 p2) l -> 1 (n l) p1 p2', p1=2 * v + p)
        ref_patches = rearrange(local_windows[..., v:-v, v:-v], '1 b p1 p2 -> b 1 p1 p2')
        scalar_product = F.conv2d(local_windows, ref_patches, groups=N * Href * Wref)
        distances = norm_patches - 2 * scalar_product  # (up to a constant)
        distances[:, :, v, v] = float('-inf')  # the reference patch is always taken
        distances = rearrange(distances, '1 (n h w) p1 p2 -> n h w (p1 p2)', n=N, h=Href, w=Wref)
        indices = torch.topk(distances, k, dim=-1, largest=False,
                                sorted=False).indices  # float('nan') is considered to be the highest value for topk
        return indices

    v = w // 2
    input_x_pad = F.pad(input_x, [v] * 4, mode='constant', value=float('nan'))
    N, C, H, W = input_x.size()
    Href, Wref = -((H - p + 1) // -s), -(
                (W - p + 1) // -s)  # ceiling division, represents the number of reference patches along each axis for unfold with stride=s
    ind_H_ref = torch.arange(0, H - p + 1, step=s, device=input_x.device)
    ind_W_ref = torch.arange(0, W - p + 1, step=s, device=input_x.device)
    if (H - p + 1) % s != 1:
        ind_H_ref = torch.cat((ind_H_ref, torch.tensor([H - p], device=input_x.device)), dim=0)
    if (W - p + 1) % s != 1:
        ind_W_ref = torch.cat((ind_W_ref, torch.tensor([W - p], device=input_x.device)), dim=0)

    indices = torch.empty(N, ind_H_ref.size(0), ind_W_ref.size(0), k, dtype=ind_H_ref.dtype, device=ind_H_ref.device)
    indices[:, :Href, :Wref, :] = block_matching_aux(input_x_pad, k, p, v, s)
    if (H - p + 1) % s != 1:
        indices[:, Href:, :Wref, :] = block_matching_aux(input_x_pad[:, :, -(2 * v + p):, :], k, p, v, s)
    if (W - p + 1) % s != 1:
        indices[:, :Href, Wref:, :] = block_matching_aux(input_x_pad[:, :, :, -(2 * v + p):], k, p, v, s)
        if (H - p + 1) % s != 1:
            indices[:, Href:, Wref:, :] = block_matching_aux(input_x_pad[:, :, -(2 * v + p):, -(2 * v + p):], k, p, v, s)

    # (ind_row, ind_col) is a 2d-representation of indices
    ind_row = torch.div(indices, 2 * v + 1, rounding_mode='floor') - v
    ind_col = torch.fmod(indices, 2 * v + 1) - v

    # from 2d to 1d representation of indices
    indices = (ind_row + rearrange(ind_H_ref, 'h -> 1 h 1 1')) * (W - p + 1) + (ind_col + rearrange(ind_W_ref, 'w -> 1 1 w 1'))
    return indices

def gather_groups(input_y, indices, p):
    """
    Gathers groups of patches based on the indices from block-matching.

    Args:
        input_y (torch.FloatTensor): Input image tensor of shape (N, C, H, W).
        indices (torch.LongTensor): Indices of similar patches of shape (N, Href, Wref, k).
        k (int): Number of similar patches.
        p (int): Patch size.

    Returns:
        torch.FloatTensor: Grouped patches of shape (N, Href, Wref, k, p**2).
    """
    unfold_Y = F.unfold(input_y, p)
    _, n, _ = unfold_Y.shape
    _, Href, Wref, k = indices.shape
    Y = torch.gather(unfold_Y, dim=2, index=repeat(indices, 'N h w k -> N n (h w k)', n=n))
    return rearrange(Y, 'N n (h w k) -> N h w k n', k=k, h=Href, w=Wref)

def aggregate(X_hat, weights, indices, H, W, p):
    """
    Aggregates groups of patches back into the image grid.

    Args:
        X_hat (torch.FloatTensor): Grouped denoised patches of shape (N, Href, Wref, k, p**2).
        weights (torch.FloatTensor): Weights of each patch of shape (N, Href, Wref, k, 1).
        indices (torch.LongTensor): Indices of the patches in the original image of shape (N, Href, Wref, k).
        H (int): Height of the original image.
        W (int): Width of the original image.
        p (int): Patch size.

    Returns:
        torch.FloatTensor: Reconstructed image tensor.
    """
    N, _, _, _, n = X_hat.size()
    X = rearrange(X_hat * weights, 'N h w k n -> (N h w k) n')
    weights = repeat(weights, 'N h w k 1 -> (N h w k) n', n=n)
    offset = (H - p + 1) * (W - p + 1) * torch.arange(N, device=X.device).view(-1, 1, 1, 1)
    indices = rearrange(indices + offset, 'N h w k -> (N h w k)')

    X_sum = torch.zeros(N * (H - p + 1) * (W - p + 1), n, dtype=X.dtype, device=X.device)
    weights_sum = torch.zeros_like(X_sum)

    X_sum.index_add_(0, indices, X)
    weights_sum.index_add_(0, indices, weights)
    X_sum = rearrange(X_sum, '(N hw) n -> N n hw', N=N)
    weights_sum = rearrange(weights_sum, '(N hw) n -> N n hw', N=N)

    return F.fold(X_sum, (H, W), p) / F.fold(weights_sum, (H, W), p)

def split2d(x):
    # x.shape: [B, C, H, W]
    c = x.shape[1]

    filter1 = torch.FloatTensor([[[[1, 0], [0, 0]]]]).to(x.device)
    filter1 = filter1.repeat(c, 1, 1, 1)

    filter2 = torch.FloatTensor([[[[0, 1], [0, 0]]]]).to(x.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    filter3 = torch.FloatTensor([[[[0, 0], [1, 0]]]]).to(x.device)
    filter3 = filter3.repeat(c, 1, 1, 1)

    filter4 = torch.FloatTensor([[[[0, 0], [0, 1]]]]).to(x.device)
    filter4 = filter4.repeat(c, 1, 1, 1)

    output1 = F.conv2d(x, filter1, stride=2, groups=c)
    output2 = F.conv2d(x, filter2, stride=2, groups=c)
    output3 = F.conv2d(x, filter3, stride=2, groups=c)
    output4 = F.conv2d(x, filter4, stride=2, groups=c)

    return [output1, output2, output3, output4]