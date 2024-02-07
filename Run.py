import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Util import *
from Models import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as MinMaxScalerSK
import json
from sklearn.metrics import r2_score,mean_absolute_error
import random
import time



#unpack config
latent_dim = 86
num_epochs = 8400
learning_rate = .000531
epoch_to_start_regressor=2780
lambda_u=7.27
lambda_l_start=1e-16


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open('HydrocarbonDataProcessed.json') as f:
    data = json.load(f)


random.shuffle(data)

#grab all '13C_shift' values
X=[]
for i in range(len(data)):

    shift=np.array(data[i]['13C_shift'])[:,0]
    #put list of shifts onto a histogram
    hist, bin_edges = np.histogram(shift, bins=4000, range=(0,250))
    #nan check
    if np.isnan(hist).any():
        print('nan in Hisotgram')
    hist_normalized = hist / np.sum(hist)
    #nan check
    if np.isnan(hist_normalized).any():
        print('nan')
        #nan to num
        hist_normalized=np.nan_to_num(hist_normalized)


    X.append(hist_normalized)
print(len(X))
X_u = torch.tensor(X).float().to(device)

with open('ViscosityTrainingData.json') as f:
    data = json.load(f)

#grab all '13C_shift' values
X=[]
for i in range(len(data)):
    shift=np.array(data[i]['13C_shift'])[:,0]
    #put list of shifts onto a histogram
    hist, bin_edges = np.histogram(shift, bins=4000, range=(0,250))
    #nan check
    if np.isnan(hist).any():
        print('nan in Hisotgram')
    hist_normalized = hist / np.sum(hist)
    #nan check
    if np.isnan(hist_normalized).any():
        print('nan')
        #nan to num
        hist_normalized=np.nan_to_num(hist_normalized)


    X.append(hist_normalized)
print(len(X))
X_l_len=len(X)
X_l = torch.tensor(X).float().to(device)

A=[]
B=[]
for i in range(len(data)):
    A.append(data[i]['A'])
    B.append(data[i]['B'])
A,B=np.array(A),np.array(B)

#Norm A,B
AScaler=MinMaxScaler()
BScaler=MinMaxScaler()
A=AScaler.fit_transform(A)
B=BScaler.fit_transform(B)
A=torch.tensor(A).float().to(device)
B=torch.tensor(B).float().to(device)

print(torch.min(A),torch.min(B))

y_l = torch.stack((A, B), dim=1).to(device)

def train_test_validation_split_torch(X, Y, test_size=0.1, validation_size=0.125):
    """
    Splits the dataset into training, validation, and testing sets.
    
    Args:
        X (Tensor): The input features.
        Y (Tensor): The targets.
        test_size (float): The proportion of the dataset to include in the test split.
        validation_size (float): The proportion of the training set to include in the validation split.
        random_state (int): Seed for the random number generator.
    
    Returns:
        X_train, X_val, X_test, Y_train, Y_val, Y_test
    """
    num_samples = X.size(0)
    
    # Split into train+val and test
    num_test_samples = int(num_samples * test_size)
    indices = torch.randperm(num_samples)
    train_val_indices = indices[num_test_samples:]
    test_indices = indices[:num_test_samples]
    
    X_train_val = X[train_val_indices]
    Y_train_val = Y[train_val_indices]
    X_test = X[test_indices]
    Y_test = Y[test_indices]
    
    # Split train+val into train and val
    num_train_samples = X_train_val.size(0)
    num_val_samples = int(num_train_samples * validation_size)
    indices_train_val = torch.randperm(num_train_samples)
    val_indices = indices_train_val[:num_val_samples]
    train_indices = indices_train_val[num_val_samples:]
    
    X_train = X_train_val[train_indices]
    Y_train = Y_train_val[train_indices]
    X_val = X_train_val[val_indices]
    Y_val = Y_train_val[val_indices]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

# Perform the split
X_l_train, X_l_val, X_l_test, y_l_train, y_l_val, y_l_test = train_test_validation_split_torch(X_l, y_l, test_size=.1, validation_size=0.1)


def get_batches(X, batch_size):
    """Yield successive n-sized chunks from X."""
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i + batch_size]

# Configuration
input_dim = X_u.shape[1]


# Model, loss, and optimizer
model = SemiSupervisedModel(input_dim, latent_dim, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
reconstruction_criterion = nn.MSELoss()
regression_criterion = nn.MSELoss()
#scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer,step_size=3000,gamma=.5)


# Convert data to PyTorch tensors
X_l = torch.tensor(X_l, dtype=torch.float).to(device)
y_l = torch.tensor(y_l, dtype=torch.float).to(device)
X_u = torch.tensor(X_u, dtype=torch.float).to(device)
y_l_test = torch.tensor(y_l_test,dtype=torch.float).to(device)
X_l_test = torch.tensor(X_l_test,dtype=torch.float).to(device)
X_l_val = torch.tensor(X_l_val,dtype=torch.float).to(device)    
y_l_val = torch.tensor(y_l_val,dtype=torch.float).to(device)
# Training loop
Loss_LS=[]
Loss_Val=[]
for epoch in range(num_epochs):

    if epoch > epoch_to_start_regressor:
        lambda_l_start = 1

    model.train()
    total_loss = 0
    
    optimizer.zero_grad()
    
    # Handle labeled data
    _, regression_output = model(X_l)
    loss_l = regression_criterion(regression_output, y_l)
    
    # Handle unlabeled data
    decoded_u, _ = model(X_u)
    loss_u = reconstruction_criterion(decoded_u, X_u)
    

    # Combine losses and update
    loss = (loss_l * lambda_l_start) + (loss_u * lambda_u)
    loss.backward()
    optimizer.step()
    
    total_loss = loss.item()
    Loss_LS.append(total_loss)

    #scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    _, regression_output = model(X_l_val)
    loss_l_val = regression_criterion(regression_output, y_l_val[:len(X_l_val)])

    loss_val = (loss_l_val * lambda_l_start) + (loss_u * lambda_u)
    Loss_Val.append(loss_val.item())

    r2=r2_score(y_l_val[:len(X_l_val)].detach().cpu().numpy(),regression_output.detach().cpu().numpy())


    print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}, Regressor Loss {loss_l * lambda_l_start:.9f}, AutoEncoder Loss {loss_u * lambda_u :.9f} LR:{current_lr:.7f}, R2={r2 :.4f}, ValMAE={mean_absolute_error(y_l_val[:len(X_l_val)].detach().cpu().numpy(),regression_output.detach().cpu().numpy())}')

"""
    plt.plot(Loss_LS)
    plt.plot(Loss_Val)
    plt.show()
    plt.close()
"""
# Ensure the model is in evaluation mode
model.eval()

torch.save(model,'Model.pt')

# Generate predictions
with torch.no_grad():  
    _,predictions = model(X_l_test) 

# Assuming predictions need to be on CPU to plot or process further
predictions = predictions.cpu()

predictions_np_test = predictions.cpu().detach().numpy() if predictions.is_cuda else predictions.detach().numpy()
Y_test_np = y_l_test.cpu().numpy() if y_l_test.is_cuda else y_l_test.numpy()


#make predictions for training data aswell
with torch.no_grad():
    _,predictions = model(X_l)
predictions = predictions.cpu()
predictions_np_train = predictions.cpu().detach().numpy() if predictions.is_cuda else predictions.detach().numpy()
Y_train_np = y_l.cpu().numpy() if y_l.is_cuda else y_l.numpy()

#test set R2 and MAE
r2=r2_score(Y_test_np,predictions_np_test)
mae=mean_absolute_error(Y_test_np,predictions_np_test)






# Plotting for target A
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
plt.scatter(Y_train_np[:, 0], predictions_np_train[:, 0], alpha=0.5, color='orange')
plt.scatter(Y_test_np[:, 0], predictions_np_test[:, 0], alpha=1,color='blue')
plt.plot([0,1], [0,1], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')


plt.legend(frameon=False, fontsize=16)
plt.xlabel('Actual', fontsize=28,weight='bold')
plt.ylabel('Predicted', fontsize=28,weight='bold')
for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', direction='in')
plt.tick_params(axis='both', which='major', direction='in')
plt.tick_params(axis='both', which='major', length=4, width=2)
plt.tick_params(axis='both', which='minor', length=4, width=2)


# Plotting for target B
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.scatter(Y_train_np[:, 1], predictions_np_train[:, 1], alpha=0.5, color='orange',label='Train Data')
plt.scatter(Y_test_np[:, 1], predictions_np_test[:, 1], alpha=1,color='blue',label='Test Data')
plt.plot([0,1], [0,1], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')


plt.legend(frameon=False, fontsize=16)
plt.xlabel('Actual', fontsize=28,weight='bold')
plt.ylabel('Predicted', fontsize=28,weight='bold')
for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', direction='in')
plt.tick_params(axis='both', which='major', direction='in')
plt.tick_params(axis='both', which='major', length=4, width=2)
plt.tick_params(axis='both', which='minor', length=4, width=2)

plt.tight_layout()
plt.show()
plt.close()


