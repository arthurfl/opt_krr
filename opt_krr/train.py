import torch
import torch.optim as optim
from torch import nn

def train_krr_model(krr_model, ref_data, train_data, test_data, num_epochs=100, lr=0.01, optimize_lambda=True, initial_gamma=None, initial_lambda=None, device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    krr_model.to(device)
    
    X_ref, y_ref = ref_data
    X_train, y_train = train_data
    X_test, y_test = test_data

    X_ref, y_ref = X_ref.to(device), y_ref.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    if initial_gamma is not None:
        krr_model.gamma.data = initial_gamma.to(device)
    if initial_lambda is not None:
        krr_model.lambda_.data = initial_lambda.to(device)

    params = [krr_model.gamma]
    if optimize_lambda:
        params.append(krr_model.lambda_)

    optimizer = optim.Adam(params, lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        krr_model.train()
        optimizer.zero_grad()
        krr_model.fit(X_ref, y_ref)
        y_train_pred = krr_model.predict(X_train)
        train_loss = criterion(y_train_pred, y_train)
        train_loss.backward()
        optimizer.step()

        krr_model.eval()
        with torch.no_grad():
            y_test_pred = krr_model.predict(X_test)
            test_loss = criterion(y_test_pred, y_test)

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}')

    print("Optimization finished.")
    return krr_model, train_losses, test_losses
