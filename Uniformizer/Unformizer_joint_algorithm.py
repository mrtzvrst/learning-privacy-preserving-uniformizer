import numpy as np
from FUNCs import Entropy

RES = float('inf')
M = 8

def assign(mat, n, m, X, Y, LEN):
    global RES, M
    
    if np.sum(X)!=0 and np.max(X) <= np.max(Y):
        for i in range(n):
            for j in range(m):
                x, y = X.copy(), Y.copy()
                temp1 = mat.copy()
                """
                if Entropy(mat)>RES or LEN>2*M-1: 
                    return 
                """
                if x[i]!=0 and y[j]!=0 and x[i] <= y[j]:
                    temp1[i][j] = x[i]
                    y[j] -= x[i]
                    x[i] = 0
                    LEN+=1
                    assign(temp1,n ,m, x, y, LEN)
                            
    elif np.sum(Y)!=0 and np.max(X) > np.max(Y):
        for i in range(m):
            for j in range(n):
                x, y = X.copy(), Y.copy()
                temp2 = mat.copy()
                """
                if Entropy(mat)>RES or LEN>2*M-1: 
                    return 
                """
                if x[j]!=0 and y[i]!=0 and y[i] <= x[j]:
                    temp2[j][i] = y[i]
                    x[j] -= y[i]
                    y[i] = 0
                    LEN+=1
                    assign(temp2, n, m, x, y, LEN)
    else:
        t1 = Entropy(mat)
        if t1 < RES:
            
            print(mat,'\n', RES, '\n', np.sum(mat, axis=0))
            RES = t1
    
        
    
LEN = 4
x = np.array([0.075, 0.025, 0.1, 0.09, 0.08, 0.07, 0.06])
y = np.ones(4)*1/M   
"""
M = 4
x = np.array([0.11, 0.24, 0.24, 0.16])
y = np.ones(3)*1/M  
"""

mat = np.zeros([len(x),len(y)])
n, m = mat.shape[0], mat.shape[1]     
Out = assign(mat, n, m, x, y, LEN)

"""
mat = np.array([[1/8, 1/8, 0, 0, 0, 0, 0, 0],
 [0, 0, 1/8, 0, 0, 0.03, 0.045, 0],
 [0, 0, 0, 1/8, 0.025, 0, 0, 0],
 [0, 0, 0, 0, 0.1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0.09, 0, 0],
 [0, 0, 0, 0, 0, 0, 0.08, 0],
 [0, 0, 0, 0, 0, 0, 0, 0.07],
 [0, 0, 0, 0, 0, 0.005, 0, 0.055]])
IR = (Entropy(np.array([0.25, 0.2, 0.15, 0.1, 0.09, 0.08, 0.07, 0.06]))+3-Entropy(mat))/3
print(mat,'\n',IR)
"""


