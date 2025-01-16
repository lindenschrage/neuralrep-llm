import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def mean_excluding_diagonal(matrix):
    return np.mean([matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix)) if i != j])

def get_error_5shot_vectorized(Xa, Xb, num_trials, shots=5):
    """
    Compute the error rate using a 5-shot approach in a fully vectorized manner.
    
    Xa, Xb: Tensors of shape (num_samples, dim_feature)
    num_trials: number of random trials
    shots: how many shots per class
    """

    num_samples = Xa.shape[0]
    
    # Generate all random permutations for A and B at once
    perm_a = torch.stack([torch.randperm(num_samples) for _ in range(num_trials)])  # (num_trials, num_samples)
    perm_b = torch.stack([torch.randperm(num_samples) for _ in range(num_trials)])
    
    # Select the shots and queries
    Xa_shots = Xa[perm_a[:, :shots]]   # (num_trials, shots, dim)
    Xb_shots = Xb[perm_b[:, :shots]]   # (num_trials, shots, dim)
    Xa_queries = Xa[perm_a[:, shots:]] # (num_trials, num_samples - shots, dim)

    # Compute centroids
    xa_centroids = Xa_shots.mean(dim=1) # (num_trials, dim)
    xb_centroids = Xb_shots.mean(dim=1) # (num_trials, dim)

    # Compute h-values for all queries in all trials:
    # h = (distance to B centroid)^2 - (distance to A centroid)^2
    xa_centroids_exp = xa_centroids.unsqueeze(1) # (num_trials, 1, dim)
    xb_centroids_exp = xb_centroids.unsqueeze(1) # (num_trials, 1, dim)

    diff_xa = Xa_queries - xa_centroids_exp  # (num_trials, num_samples - shots, dim)
    diff_xb = Xa_queries - xb_centroids_exp

    # Sum of squared distances
    dist_xa = (diff_xa**2).sum(dim=2)  # (num_trials, num_samples - shots)
    dist_xb = (diff_xb**2).sum(dim=2)

    h_values = dist_xb - dist_xa

    # Flatten all h-values from all trials
    h_true = h_values.flatten()

    # Compute error rate
    err_true = (h_true <= 0).float().mean()
    return err_true


def get_error_list_5shot_vectorized(feature, num_trials=200, shots=5): 
    """
    Compute the error matrix for all pairs of classes using the vectorized 5-shot method.

    feature: Tensor or array of shape (num_classes, num_samples, dim_feature)
    num_trials: number of random trials
    shots: number of shots per class
    """
    # Convert to torch tensors if not already
    if not isinstance(feature, torch.Tensor):
        feature = torch.tensor(feature, dtype=torch.float32)

    M = feature.shape[0]  # number of classes
    err_list = np.full((M, M), np.nan)

    # Example: CATEGORIES might be a list of class names, adjust as needed
    CATEGORIES = [f"Class_{i}" for i in range(M)]

    for i in range(M): 
        print(f"Processing class {CATEGORIES[i]}...")
        for j in range(M): 
            if i != j:
                err_list[i, j] = get_error_5shot_vectorized(
                    feature[i], feature[j], num_trials, shots=shots
                ).item()

    return err_list
