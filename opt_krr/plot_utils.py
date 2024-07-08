import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_learning_curves(train_losses, test_losses):
    """
    Plot the learning curves for training and test losses.
    
    Args:
        train_losses (list of float): List of training losses.
        test_losses (list of float): List of test losses.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions_vs_true(y_true, y_pred, dataset_name=""):
    """
    Plot the predictions vs true values.
    
    Args:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.
        dataset_name (str): Name of the dataset (optional).
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, edgecolor='k', alpha=0.7, label='Predictions')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs True Values ({dataset_name})')
    plt.legend()
    plt.grid(True)
    plt.show()
