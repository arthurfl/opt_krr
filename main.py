import os
import torch
from opt_krr.krr_model import KernelRidgeRegression
from opt_krr.train import train_krr_model
from opt_krr.utils import compute_whitening_parameters, whiten_data
from opt_krr.plot_utils import plot_learning_curves, plot_predictions_vs_true

import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Creating synthetic datasets
    torch.manual_seed(0)
    X_ref = torch.randn(50, 3) # Reference dataset
    y_ref = torch.randn(50) # Reference labels
    X_train = torch.randn(30, 3) # Training dataset
    y_train = torch.randn(30) # Training labels
    X_test = torch.randn(20, 3) # Test dataset
    y_test = torch.randn(20) # Test labels

    # One can apply q whitening transformation to the datasets: 
    # mean, whitening_matrix = compute_whitening_parameters(X_train)

    # X_train_whitened = whiten_data(X_train, mean, whitening_matrix)
    # X_test_whitened = whiten_data(X_test, mean, whitening_matrix)

    # Initializing the KRR model
    input_dim = X_ref.shape[1]  # Number of features
    krr = KernelRidgeRegression(kernel='lap', lambda_=1.0, gamma=None, input_dim=input_dim)

    # Defining initial regularization strength and bandwidths
    initial_lambda = torch.tensor(0.1, dtype=torch.float32)
    # Bandwidths can either be a vector or a matrix
    # initial_gamma = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    initial_gamma = torch.tensor([[1.0, 0.5, 0.2], [0.5, 2.0, 0.1], [0.1, 0.9, 0.99]])

    print("Least-squares solver")

    # Training the model, through simultaneous optimization of lambda and gamma
    trained_krr, train_losses, test_losses = train_krr_model(
        krr, (X_ref, y_ref), (X_train, y_train), (X_test, y_test),
        num_epochs=200, lr=0.01, loss="l1", optimize_lambda=True,
        initial_gamma=initial_gamma, initial_lambda=initial_lambda, 
        solver="leastsquares", device=device
    )

    # Plot learning curves
    plot_learning_curves(train_losses, test_losses)

    # Make predictions with the trained model on new data
    X_new = torch.randn(5, 3).to(trained_krr.lambda_.device)
    y_new_pred = trained_krr.predict(X_new)
    print(y_new_pred)

    # Plot predictions vs true values, on training and test datasets
    y_train_pred = trained_krr.predict(X_train)
    plot_predictions_vs_true(y_train, y_train_pred, dataset_name="Training")

    y_test_pred = trained_krr.predict(X_test)
    plot_predictions_vs_true(y_test, y_test_pred, dataset_name="Test")

    trained_krr.save('model.pth')

    # Display the bandwidth matrix
    plt.imshow(trained_krr.gamma.detach().numpy())
    plt.show()
