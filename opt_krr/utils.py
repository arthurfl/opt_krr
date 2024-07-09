import torch

def compute_whitening_parameters(X):
    """
    Compute the whitening parameters (mean and whitening matrix) from the input data.
    
    Args:
    X (torch.Tensor): The input data tensor with shape (n_samples, n_features).
    
    Returns:
    torch.Tensor, torch.Tensor: The mean and whitening matrix.
    """
    # Compute the mean
    mean = torch.mean(X, dim=0)
    
    # Center the data
    X_centered = X - mean
    
    # Compute the covariance matrix
    cov_matrix = torch.mm(X_centered.t(), X_centered) / (X_centered.size(0) - 1)
    
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = torch.eig(cov_matrix, eigenvectors=True)
    
    # Construct the whitening matrix
    whitening_matrix = torch.mm(eigenvectors, torch.diag(1.0 / torch.sqrt(eigenvalues[:, 0] + 1e-5)))
    
    return mean, whitening_matrix

def whiten_data(X, mean, whitening_matrix):
    """
    Apply the whitening operation to the input data using precomputed mean and whitening matrix.
    
    Args:
    X (torch.Tensor): The input data tensor with shape (n_samples, n_features).
    mean (torch.Tensor): The mean tensor computed from the training data.
    whitening_matrix (torch.Tensor): The whitening matrix computed from the training data.
    
    Returns:
    torch.Tensor: The whitened data tensor with the same shape as input.
    """
    # Center the data
    X_centered = X - mean
    
    # Whiten the data
    X_whitened = torch.mm(X_centered, whitening_matrix)
    
    return X_whitened
