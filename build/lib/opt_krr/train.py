import torch
import torch.optim as optim
from torch import nn

def train_krr_model(krr_model, ref_data, train_data, test_data, num_epochs=100, lr=0.01, optimize_lambda=True, initial_gamma=None, initial_lambda=None):
    X_ref, y_ref = ref_data
    X_train, y_train = train_data
    X_test, y_test = test_data

    if initial_gamma is not None:
        krr_model.gamma.data = initial_gamma
    if initial_lambda is not None:
        krr_model.lambda_.data = initial_lambda

    params = [krr_model.gamma]
    if optimize_lambda:
        params.append(krr_model.lambda_)

    optimizer = optim.Adam(params, lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        krr_model.train()
        optimizer.zero_grad()
        krr_model.fit(X_ref, y_ref)
        y_train_pred = krr_model.predict(X_train)
        train_loss = criterion(y_train_pred, y_train)
        train_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            krr_model.eval()
            with torch.no_grad():
                y_test_pred = krr_model.predict(X_test)
                test_loss = criterion(y_test_pred, y_test)
                print(f'Epoch {epoch}, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}')

    print("Optimization finished.")
    return krr_model
