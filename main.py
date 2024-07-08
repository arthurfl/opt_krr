import torch
from opt_krr.krr_model import KernelRidgeRegression
from opt_krr.train import train_krr_model

if __name__ == "__main__":
    # Create synthetic data
    torch.manual_seed(0)
    X_ref = torch.randn(50, 3)  # Reference dataset
    y_ref = torch.randn(50)     # Reference labels
    X_train = torch.randn(30, 3)  # Training dataset
    y_train = torch.randn(30)     # Training labels
    X_test = torch.randn(20, 3)   # Test dataset
    y_test = torch.randn(20)      # Test labels

    # Initialize Kernel Ridge Regression model
    input_dim = X_ref.shape[1]  # Number of features
    krr = KernelRidgeRegression(kernel='rbf', lambda_=1.0, gamma=None, input_dim=input_dim)

    # Define initial values for gamma and lambda
    initial_gamma = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    initial_lambda = torch.tensor(0.1, dtype=torch.float32)

    # Train the model and optimize parameters
    trained_krr = train_krr_model(krr, (X_ref, y_ref), (X_train, y_train), (X_test, y_test), num_epochs=200, lr=0.01, optimize_lambda=True, initial_gamma=initial_gamma, initial_lambda=initial_lambda)

    # Make predictions with the trained model
    X_new = torch.randn(5, 3).to(trained_krr.lambda_.device)  # Ensure the new data is also on the same device
    y_new_pred = trained_krr.predict(X_new)
    print(y_new_pred)
