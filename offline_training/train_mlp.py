import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from model import mlp_model
from utils import utils
import numpy as np
import sys

torch.manual_seed(0)

target = sys.argv[1]
strategy = "FCUFS"

# Hypre parameters
if strategy == "CFS":
    if target == "power":
        hidden_size = 5
        output_size = 1
        hidden_layer_num = 4
        lr = 0.001
        batch_size = 4
        epochs = 1996
        criterion = nn.MSELoss()
    elif target == "performance":
        hidden_size = 5
        output_size = 1
        hidden_layer_num = 4
        lr = 0.001
        batch_size = 8
        epochs = 2995
        criterion = nn.MSELoss()

if strategy == "UFS":
    if target == "power":
        hidden_size = 5
        output_size = 1
        hidden_layer_num = 4
        lr = 0.001
        batch_size = 8
        epochs = 800
        criterion = nn.MSELoss()
    elif target == "performance":
        hidden_size = 5
        output_size = 1
        hidden_layer_num = 4
        lr = 0.001
        batch_size = 8
        epochs = 1999
        criterion = nn.MSELoss()

if strategy == "FCUFS":
    if target == "power":
        hidden_size = 6
        output_size = 1
        hidden_layer_num = 4
        lr = 0.01
        batch_size = 32
        epochs = 314
        criterion = nn.MSELoss()
    elif target == "performance":
        hidden_size = 6
        output_size = 1
        hidden_layer_num = 4
        lr = 0.01
        batch_size = 32
        epochs = 912
        criterion = nn.MSELoss()

# load data
print("loading data")
input_size = mlp_model.get_features_num(f"trainset_{strategy}/dataset/train.csv", target)
print("feature num:", input_size)
model = mlp_model.MLP(input_size, hidden_size, output_size, hidden_layer_num)
x_data, y_data = model.load_features(f"trainset_{strategy}/dataset/train.csv", target)

# split
num_rows = x_data.shape[0]
select_rows = int(0.1 * num_rows)
random_indices = torch.randperm(num_rows)

x_test = x_data[random_indices[:select_rows]]
y_test = y_data[random_indices[:select_rows]]

x_train = x_data[random_indices[select_rows:]]
y_train = y_data[random_indices[select_rows:]]

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
print("training")
loss_history = list()
test_loss_history = list()

for epoch in range(epochs):
    model.train()
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        train_mean_error = round(torch.mean(torch.abs(outputs - labels)).item() * 100, 2)
        train_max_error = round(torch.quantile(torch.abs(outputs - labels), 1).item() * 100, 2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    loss_history.append(loss.item())

    # Test
    model.eval()
    y_pred = model(x_test)
    test_mean_error = round(torch.mean(torch.abs(y_pred - y_test)/y_test).item() * 100, 2)
    test_max_error = round(torch.quantile(torch.abs(y_pred - y_test)/y_test, 0.95).item() * 100, 2)

    test_loss = criterion(y_pred, y_test)
    test_loss_history.append(test_loss.item())

    # Log
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {round(loss.item(), 6)}, Test_Loss:{round(test_loss.item(), 6)}, Test_mean_error:{test_mean_error}%, max {test_max_error}%')

y_pred = model(x_data)
print(np.abs((y_data.detach().numpy() - y_pred.detach().numpy()) / y_data.detach().numpy()) * 100)

print("RMSE", np.sqrt(((y_pred.detach().numpy() - y_data.detach().numpy()) ** 2).mean()))
print("MAPE", np.mean(np.abs((y_data.detach().numpy() - y_pred.detach().numpy()) / y_data.detach().numpy())) * 100)

print("Training complete")

torch.save(model, f"{target}_model_{strategy}.pth")
