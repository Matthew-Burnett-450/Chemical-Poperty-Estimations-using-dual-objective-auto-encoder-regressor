import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from Models import *    
import json
from Util import *
import pandas as pd

model = torch.load('model.pt', map_location=torch.device('cpu'))

model.eval()

#load viscosities
with open('ViscosityTrainingData.json') as f:
    data = json.load(f)

#remove data whose names end in 'ene' or 'yne' upper or lower case lower case all before checking
data=[i for i in data if i['MolName'].lower().endswith('ane')]
                                                                                                        
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




A=torch.tensor(A).float()
B=torch.tensor(B).float()


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
X = torch.tensor(X).float()

#predict viscosities
y_pred = model(X)[1]
A_pred,B_pred=y_pred[:,0].detach().numpy(),y_pred[:,1].detach().numpy()


#untransform
A_pred=AScaler.inverse_transform(A_pred)
B_pred=BScaler.inverse_transform(B_pred)
A=AScaler.inverse_transform(A)
B=BScaler.inverse_transform(B)



#orrick erbar
def orrick_erbar(A,B,T_range):
    T_range = np.array(T_range)
    return np.exp(A+ (B/T_range))

T_range = np.linspace(200,450,25)
T_range_scatter = np.linspace(273,450,200)

#print name of the data ploted
plt.title('Viscosity Data')
for i in range(5):
    name=data[i]['MolName']
    plt.scatter(T_range_scatter,orrick_erbar(A_pred[i],B_pred[i],T_range_scatter),s=5,marker="^",label=name)
    plt.plot(T_range,orrick_erbar(A[i],B[i],T_range),label=name,linewidth=1.5,linestyle='dashed')
plt.xlabel('Temperature (K)')
plt.ylabel('Viscosity (Pa s)')
plt.legend(frameon=False)
plt.show()
plt.close()

plt.figure(figsize=(8*1.5,6*1.5))
for i in range(len(data)):
    name=data[i]['MolName']
    if i==3:
        plt.scatter(orrick_erbar(A[i],B[i],T_range_scatter)*1000,orrick_erbar(A_pred[i],B_pred[i],T_range_scatter)*1000,s=5,marker="^",label='Predictions',alpha=.1,color='k')

    plt.scatter(orrick_erbar(A[i],B[i],T_range_scatter)*1000,orrick_erbar(A_pred[i],B_pred[i],T_range_scatter)*1000,s=5,marker="^",alpha=.1,color='k')
plt.plot([0,2],[0,2],label='Ideal',linewidth=1.5,linestyle='dashed',color='k')
plt.legend(frameon=False, fontsize=16)

for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Actual Viscosity (mPa*s)',fontsize=28,fontweight='bold')
plt.ylabel('Predicted Viscosity (mPa*s)',fontsize=28,fontweight='bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', direction='in')
plt.tick_params(axis='both', which='major', direction='in')
plt.tick_params(axis='both', which='major', length=4, width=2)
plt.tick_params(axis='both', which='minor', length=4, width=2)
#figure size 
plt.xlim(0.001,2)
plt.ylim(0,2)
plt.show()



#load csv
HRJ_data=np.loadtxt('7720 HRJ.csv',delimiter=',',skiprows=1,dtype=np.float64)
#remove rows where intensity is zero
HRJ_data=HRJ_data[HRJ_data[:,1]>0]

hist, bin_edges = np.histogram(HRJ_data[:,0], bins=4000, range=(0,250),weights=HRJ_data[:,1])


#pass this histogram through the model
hist_normalized = hist / np.sum(hist)

plt.plot(hist_normalized)
plt.show()

hist_normalized = torch.tensor(hist_normalized).float()
#turn into tensor make 2D
hist_normalized = hist_normalized.unsqueeze(0)
y_pred = model(hist_normalized)
A_pred,B_pred=y_pred[1][0].detach().numpy()
A_pred_HRJ=AScaler.inverse_transform(A_pred)
B_pred_HRJ=BScaler.inverse_transform(B_pred)



print(A_pred_HRJ)

#load csv
jetA_data=np.loadtxt('10325 Jet A.csv',delimiter=',',skiprows=1)
#remove rows where intensity is zero

jetA_data=jetA_data[jetA_data[:,1]>0]

hist, bin_edges = np.histogram(jetA_data[:,0], bins=4000, range=(0,250),weights=jetA_data[:,1])


#pass this histogram through the model
hist_normalized = hist / np.sum(hist)
hist_normalized = torch.tensor(hist_normalized).float()
#turn into tensor make 2D
hist_normalized = hist_normalized.unsqueeze(0)
y_pred = model(hist_normalized)
A_pred,B_pred=y_pred[1][0].detach().numpy()
A_pred_jetA=AScaler.inverse_transform(A_pred)
B_pred_jetA=BScaler.inverse_transform(B_pred)


print(A_pred_jetA)


plt.figure(figsize=(8,6))
#plot the predicted spectrum
plt.plot(T_range,orrick_erbar(A_pred_jetA,B_pred_jetA,T_range)*1000,label='Jet-A POSF 10325 Predicted',color='k',marker='.')
plt.plot(T_range,orrick_erbar(A_pred_HRJ,B_pred_HRJ,T_range)*1000,label='HRJ POSF 7720 Predicted',color='k',linestyle='--')


plt.scatter([273.15-40,273.15-20,40+273.15],[9.55,4.70,1.80],label='Jet-A POSF 10325',color='k')
plt.scatter([233.15,253.15,313.15],[14.00,6.10,1.50],label='HRJ POSF 7720',color='k',marker='^')


plt.legend(frameon=False, fontsize=16)

for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Temperature (K)',fontsize=28,fontweight='bold')
plt.ylabel('Viscosity (mPa*s)',fontsize=28,fontweight='bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', direction='in')
plt.tick_params(axis='both', which='major', direction='in')
plt.tick_params(axis='both', which='major', length=4, width=2)
plt.tick_params(axis='both', which='minor', length=4, width=2)
#figure size 

plt.show()
