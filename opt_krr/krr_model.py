import torch
import torch.nn as nn

class KernelRidgeRegression(nn.Module):
    def __init__(self, kernel='linear', lambda_=1.0, gamma=None, degree=3, coef0=1, input_dim=1):
        super(KernelRidgeRegression, self).__init__()
        self.kernel = kernel
        self.lambda_ = nn.Parameter(torch.tensor(lambda_, dtype=torch.float32), requires_grad=True)
        if gamma is None:
            gamma = torch.ones(input_dim, dtype=torch.float32)
        self.gamma = nn.Parameter(gamma, requires_grad=True)
        self.degree = degree
        self.coef0 = coef0
        self.X_ref = None
        self.alpha_ = None
    
    def _linear_kernel(self, X, Y):
        return torch.matmul(X, Y.T)
    
    def _polynomial_kernel(self, X, Y):
        return (self.coef0 + torch.matmul(X, Y.T)) ** self.degree
    
    def _rbf_kernel(self, X, Y):
        gamma = self.gamma
        gamma_expanded = gamma.view(1, -1)  # Expand gamma for broadcasting
        X_scaled = X / torch.sqrt(gamma_expanded)
        Y_scaled = Y / torch.sqrt(gamma_expanded)
        K = torch.cdist(X_scaled, Y_scaled) ** 2
        return torch.exp(-K)
    
    def _kernel_function(self, X, Y):
        if self.kernel == 'linear':
            return self._linear_kernel(X, Y)
        elif self.kernel == 'poly':
            return self._polynomial_kernel(X, Y)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X, Y)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X_ref, y_ref):
        self.X_ref = X_ref
        K = self._kernel_function(X_ref, X_ref)
        n = K.shape[0]
        I = torch.eye(n, device=K.device)
        self.alpha_ = torch.linalg.solve(K + self.lambda_ * I, y_ref)
    
    def predict(self, X):
        K = self._kernel_function(X, self.X_ref)
        return torch.matmul(K, self.alpha_)

    def forward(self, X):
        return self.predict(X)
