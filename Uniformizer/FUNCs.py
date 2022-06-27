import numpy as np
import matplotlib.pyplot as plt

def Py_given_x(P):
    
    P=-np.sort(-P)
    x = P.copy()
    n = len(x)
    Mat = np.zeros([n,n])
    
    x_ind = np.array([i[0] for i in sorted(enumerate(x), key=lambda x:x[1], reverse=True)])

    for i in range(n-1):
        temp = 1/n
        while temp>0:
            t = min(temp, x[x_ind[0]])
            Mat[x_ind[0], i] = t
            x[x_ind[0]] = x[x_ind[0]]-t
            temp = temp-t
            x_ind = [i[0] for i in sorted(enumerate(x), key=lambda x:x[1], reverse=True)]
    
    Mat[:,n-1] = -np.sort(-P)-Mat.sum(axis=1) 
    
    P_joint = Mat.copy()
    for i in range(n):
        Mat[i,:] /= P[i]
    # P_joint is Pxy and Mat.transpose() is PY|X
    return Mat.transpose(), P_joint

#_____________________________________________________________________________

def Entropy(Mat):
    Mat = Mat.flatten()
    return sum([-i*np.log2(i) for i in Mat if i!=0])

#_____________________________________________________________________________

def Extended_Prob(Prob, N):
    m = len(Prob)
    m_new = m**N
    New_Prob = np.zeros(m_new)
    for i in range(m_new):
        New_Prob[i] = np.prod(Prob[np.array(list(np.binary_repr(i, width=N)), dtype=int)])  

    return New_Prob


