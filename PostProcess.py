import numpy as np
import matplotlib.pyplot as plt
import os
import wandb
# load each file starting with name 'predictions_np_ and ending with .npz
# and append the data to a list     data = {'Y_test_np':Y_test_np,'predictions_np_test':predictions_np_test,'Y_train_np':Y_train_np,'predictions_np_train':predictions_np_train,'test r2':r2,'test mae':mae}

#inizialize the list
Y_test_np = []
predictions_np_test = []
Y_train_np = []
predictions_np_train = []
r2 = []

#load the data
for file in os.listdir():
    if file.startswith('predictions_np_') and file.endswith('.npy'):
        data = np.load(file,allow_pickle=True)
        #unpack array
        data = data.item()
        Y_test_np.append(data['Y_test_np'])
        predictions_np_test.append(data['predictions_np_test'])
        Y_train_np.append(data['Y_train_np'])
        predictions_np_train.append(data['predictions_np_train'])
        r2.append(data['test r2'])

plt.figure()
colors= ['b','g','r','c','m']*50
colors=['k']*200
#plot the best five 
r2 = np.array(r2)
idx = np.argsort(r2)[-100:]
for num,i in enumerate(idx):
    plt.scatter(Y_test_np[i],predictions_np_test[i],label='R2: {}'.format(r2[i]),c=colors[num],alpha=0.1)
    plt.title('Best 5 models')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.plot([0,1],[0,1],'k--')
    plt.legend()
plt.show()

#dowbload every model

import wandb
api = wandb.Api()

sweep = api.sweep("mb_uofsc/NMR_Prediction_of_Chem_Props/wc34oq4u")

params=[]
loss=[]
#donwnload parameters of every model and test mae
for run in sweep.runs:
    #skip if the run is not finished
    if run.state != 'finished':
        continue
    print(run.name)
    print(run.config)
    print(run.summary['Test MAE'])
    #grap config params and test mae and append to list as a list of list
    params.append(list(run.config.values()))
    loss.append(run.summary['Test MAE'])

#run pca on the parameters
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(params)
params_pca = pca.transform(params)

#plot 3d scatter plot of the parameters and test mae
#log scale the test mae
loss = np.log(loss)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(params_pca[:,0],params_pca[:,1],loss)
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('Test MAE')
plt.show()




                      