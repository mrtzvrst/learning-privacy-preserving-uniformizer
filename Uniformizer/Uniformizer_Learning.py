import numpy as np
import torch
from torch import nn, optim
# Logging metadata
import matplotlib.pyplot as plt
from FUNCs import Py_given_x, Entropy, Extended_Prob

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

##############################################################################
class NN_model(nn.Module):
    def __init__(self, Mat_size):
        super().__init__()
        self.l1 = nn.Linear(Mat_size, Mat_size)
    def forward(self, x):
        return torch.softmax(self.l1.weight, dim=0)
  
    
    
def loss_function(Qy_given_x, x, lm, M):
    t2 = torch.matmul(Qy_given_x, torch.transpose(x,0,1))
    return -torch.sum(t2*torch.log2(t2))/x.shape[0] + lm*torch.sum((torch.sum(t2, dim=1)/x.shape[0]-1/M)**2)
    

# [0.6 0.4]
Prob = np.array([0.36, 0.24, 0.24, 0.16])
lambda_ = 500
batch_size = 10000
epochs = 500
M = Prob.shape[0]
model = NN_model(M)
optimizer = optim.Adam(model.parameters(), lr=1e-2)



def train(M, Prob, epoch, batch_size):
    model.train()
    train_loss = 0
    
    
    for i in range(batch_size+1):
    
        Data = torch.LongTensor(np.random.choice(np.arange(M), size=batch_size, p=Prob))
        x = torch.nn.functional.one_hot(Data, num_classes=M)
        x = torch.tensor(x, dtype=torch.float)
        
        optimizer.zero_grad()
        Qy_given_x = model.forward(x)
        loss = loss_function(Qy_given_x, x, lambda_, M)
        loss.backward()
        train_loss += loss.item() 
        optimizer.step()
        
    print(epoch, train_loss/batch_size, optimizer.param_groups[0]['lr'])
    return Qy_given_x, train_loss/batch_size


temp = 500
for epoch in range(1, epochs + 1):
    y, loss1 = train(M, Prob, epoch, batch_size)
    if loss1>temp:
        optimizer.param_groups[0]['lr']/=8
    temp = loss1 
    if optimizer.param_groups[0]['lr'] < 0.001:
        break

Pyx = y*torch.tensor(Prob).repeat(M,1)
PXY_Learning = torch.transpose(Pyx,0,1)
PXgY_le = PXY_Learning/ (torch.sum(PXY_Learning, dim=0).repeat(M,1)) #this devision is not correct. It should be divided to the correct prob of y
MI_Learning = (Entropy(Prob) - 1/(M)*Entropy(PXgY_le.detach().numpy()))/2

   
_, PXY_algorithm= Py_given_x(Prob)
PXgY_al = PXY_algorithm / (1/M)
MI_algorithm = (Entropy(Prob) - 1/(M)*Entropy(PXgY_al))/2


print(f"for probability vector: {Prob} \n Algorithm MI is {MI_algorithm} \n Learning MI is {MI_Learning}")
print(torch.round(PXY_Learning, decimals = 3))
print(np.round(PXY_algorithm, 3))
    
    

    
  
    
