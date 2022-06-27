import numpy as np
from FUNCs import Entropy


N = 10
Prob = np.array([0.8,0.2]) # probability of 1 is 0.2
i_idx = 0 # should be less than N


X_dic = {}
for i in range(2**N):
    X_dic[i] = np.prod(Prob[np.array(list(np.binary_repr(i, width=N)), dtype=int)])  
        

Y_dic = X_dic.copy()
Y_dic = dict(zip(Y_dic,np.zeros(len(Y_dic))))
t0, t1 = Y_dic.copy(), Y_dic.copy()

for i in range(2**N):
    for j in range(2**N):
        Y_dic[(i+j)%(2**N)] += X_dic[i]*X_dic[j]
        
        if (np.binary_repr(i, width=N)[i_idx] == '1'):
            t1[(i+j)%(2**N)] += X_dic[i]*X_dic[j]
        else:
            t0[(i+j)%(2**N)] += X_dic[i]*X_dic[j]

            
Y_new = np.array(list(Y_dic.values()))

t0 = np.array(list(t0.values()))/Prob[0]
t1 = np.array(list(t1.values()))/Prob[1]
MI = Entropy(Y_new) - Prob[0]*Entropy(t0) - Prob[1]*Entropy(t1)     


print(f"\n\nFor Length N={N} and Bern~{Prob[1]} with H(Prob)={Entropy(Prob)}:\n\nEntropy rate of Y is {Entropy(Y_new)/N}\n\nMutual Information is {MI}")


"""

N =10
Prob = np.array([0.8,0.2]) # probability of 1 is 0.2
i_idx = 0 # should be less than N


X_dic = {}
for i in range(2**N):
    X_dic[i] = np.prod(Prob[np.array(list(np.binary_repr(i, width=N)), dtype=int)])  
        

Y_dic = {}
for i in range(2**(N-1)):
    Y_dic[i] = 0
t0, t1 = Y_dic.copy(), Y_dic.copy()

for i in range(2**N):
    for j in range(2**N):
        Y_dic[(i+j)%(2**(N-1))] += X_dic[i]*X_dic[j]
        
        if (np.binary_repr(i, width=N)[i_idx] == '1'):
            t1[(i+j)%(2**(N-1))] += X_dic[i]*X_dic[j]
        else:
            t0[(i+j)%(2**(N-1))] += X_dic[i]*X_dic[j]

            
Y_new = np.array(list(Y_dic.values()))

t0 = np.array(list(t0.values()))/Prob[0]
t1 = np.array(list(t1.values()))/Prob[1]
MI = Entropy(Y_new) - Prob[0]*Entropy(t0) - Prob[1]*Entropy(t1)     


print(f"\n\nFor Length N={N} and Bern~{Prob[1]} with H(Prob)={Entropy(Prob)}:\n\nEntropy rate of Y is {Entropy(Y_new)/N}\n\nMutual Information is {MI}")


"""
