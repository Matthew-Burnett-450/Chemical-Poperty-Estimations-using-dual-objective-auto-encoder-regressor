############################################ Hyperparameters ############################################
Epochs = 100
latentDim = 50
##########################################################################################################


import torch
from Models import *
import torch.optim as optim
import torch.nn as nn
import json
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence,PackedSequence,pad_sequence
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#load data
with open('UnlabledNMRData.json') as f:
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_tensor = torch.tensor(X).float().to(device)

input_dim = X_tensor.shape[1]

# Model, optimizer, and loss function initialization
model = VariationalAutoencoder(input_dim, 15).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=.0001)

# Hyperparameters
batch_size = 100


# Manual batching function (as defined previously)
def get_batches(X, batch_size):
    num_samples = X.shape[0]
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        yield X[start_idx:end_idx]

# Training loop
lossList = []

# Assume Epochs and batch_size are defined
for epoch in range(Epochs):
    model.train()
    total_loss = 0.0
    
    for inputs in get_batches(X_tensor, batch_size):
        inputs = inputs.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass through the model
        # Adjusted for VAE to get reconstruction, mu, and log_var
        reconstruction, mu, log_var = model(inputs)
        
        # Compute VAE loss
        loss = model.vae_loss(reconstruction, inputs, mu, log_var)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    # Store the total loss in the list for later analysis if needed
    lossList.append(total_loss / len(X_tensor))  # Normalize total loss by dataset size for consistency

    # Print loss statistics, normalized by the dataset size
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(X_tensor):.6f}')




plt.plot(lossList)
plt.show()
plt.close()

#pca

pca = PCA(n_components=2)

#predict
model.eval()
latent = model.encoder(X_tensor.to(device)).detach().cpu().numpy()

#kmeans
kmeans = KMeans(n_clusters=3, random_state=0,n_init='auto').fit(latent)
print(kmeans.labels_)

latent_reduced = pca.fit_transform(latent)
plt.scatter(latent_reduced[:, 0], latent_reduced[:, 1], c=kmeans.labels_)
plt.show()


with open('ViscosityTrainingData.json') as f:
    data = json.load(f)


#load labled data
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



X_tensor = torch.tensor(X).float().to(device)

latent = model.encoder(X_tensor.to(device)).detach().cpu().numpy()

#save latent to data
for i,Mol in enumerate(data):
    Mol['AutoEncodedData'] = latent[i].tolist()


with open('ViscosityTrainingData.json', 'w') as f:
    json.dump(data, f)

