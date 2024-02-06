import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Util import *
from Models import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as MinMaxScalerSK
import json
################ Hyperparameters ################
Epochs = 10000
batch_size = 64
learning_rate = 1e-3
num_epochs = 100
#################################################
scaler=MinMaxScalerSK()

with open('ViscosityTrainingData.json') as f:
    data = json.load(f)

#grab all 'AutoEncodedData' values
X_l=[]
A=[]
B=[]
for i in range(len(data)):
    X_l.append(data[i]['AutoEncodedData'])
    A.append(data[i]['A'])
    B.append(data[i]['B'])




X_l=np.array(X_l)
#Norm A,B
AScaler=MinMaxScaler()
BScaler=MinMaxScaler()
A=AScaler.fit_transform(A)
B=BScaler.fit_transform(B)

X_l=scaler.fit_transform(X_l)

#train test split
X_l, X_test, A, A_test, B, B_test = train_test_split(X_l, A, B, test_size=0.1)

#convert to torch tensor
X_l=torch.tensor(X_l).float()
A=torch.tensor(A).float()
B=torch.tensor(B).float()

targets = torch.stack((A, B), dim=1)








#Shape check
print(X_l.shape,A.shape,B.shape)
def manual_batching(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]



# Model, optimizer, and loss functions
model = SemiSupervisedModel(input_dim=X_l.shape[1], latent_dim=50).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
reconstruction_loss_fn = nn.MSELoss()
regression_loss_fn = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    # Labeled data training
    for batch, target_batch in zip(manual_batching(X_l, batch_size), manual_batching(targets, batch_size)):
        batch, target_batch = batch.to(device), target_batch.to(device)
        optimizer.zero_grad()
        reconstructed, regression_output = model(batch)
        reconstruction_loss = reconstruction_loss_fn(reconstructed, batch)
        regression_loss = regression_loss_fn(regression_output, target_batch)
        loss = reconstruction_loss + regression_loss  # Combine losses as needed
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Unlabeled data training (for reconstruction loss only)
    for batch_u in manual_batching(X_u, batch_size):
        batch_u = batch_u.to(device)
        optimizer.zero_grad()
        reconstructed_u, _ = model(batch_u)
        loss_u = reconstruction_loss_fn(reconstructed_u, batch_u)
        loss_u.backward()
        optimizer.step()
        total_loss += loss_u.item()

    print(f"Epoch {epoch+1}, Total Loss: {total_loss}")







#evaluate
modelA.eval()
modelB.eval()

A_pred=modelA(X_l)
B_pred=modelB(X_l)

A_pred=A_pred.cpu().detach().numpy()
B_pred=B_pred.cpu().detach().numpy()
A=A.cpu().detach().numpy()
B=B.cpu().detach().numpy()

#predict on test set
X_test=torch.tensor(X_test).float()
X_test=X_test.to(device)

A_pred_test=modelA(X_test)
B_pred_test=modelB(X_test)

A_pred_test=A_pred_test.cpu().detach().numpy()
B_pred_test=B_pred_test.cpu().detach().numpy()


#side by side comparison plots for A and B
plt.figure()
plt.subplot(1,2,1)
plt.plot(A,A_pred,'o')
plt.plot(A_test,A_pred_test,'ro')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('True A')
plt.ylabel('Predicted A')
plt.subplot(1,2,2)
plt.plot(B,B_pred,'o')
plt.plot(B_test,B_pred_test,'ro')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('True B')
plt.ylabel('Predicted B')
plt.show()