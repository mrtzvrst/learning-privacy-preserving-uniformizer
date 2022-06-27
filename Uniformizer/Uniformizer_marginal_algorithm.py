import numpy as np
import matplotlib.pyplot as plt
from FUNCs import Py_given_x, Entropy, Extended_Prob

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


Prob = np.array([0.1,0.9])
N = 10
Mutual_inf = np.zeros(N)
for i in range(1,N+1):
    Ext_prob = Extended_Prob(Prob, i)
    _, PXY = Py_given_x(Ext_prob)
    Px_given_y = PXY / (1/(len(Prob)**i))
    
    Mutual_inf[i-1] = (Entropy(Ext_prob) - 1/(len(Prob)**i)*Entropy(Px_given_y))/i

plt.plot(np.arange(1,N+1),Mutual_inf/Entropy(Prob),"-b", label="[0.1]")



"""
X = np.array([0.36,0.24,0.24,0.16])
y1 = np.array([24/25, 1/25])
y2 = np.array([23/25, 2/25])
y3 = np.array([14/25,11/25])

a = (Entropy(X) - 1/4*(Entropy(y1)+Entropy(y2)+Entropy(y3)))/2
print(a)
"""



"""
Prob = np.array([0.2,0.8])

Mutual_inf = np.zeros(N)
for i in range(1,N+1):
    Ext_prob = Extended_Prob(Prob, i)
    _, Px_given_y = Py_given_x(Ext_prob)
    Px_given_y /= 1/(len(Prob)**i)
    
    Mutual_inf[i-1] = (Entropy(Ext_prob) - 1/(len(Prob)**i)*Entropy(Px_given_y))/i

plt.plot(np.arange(1,N+1),Mutual_inf/Entropy(Prob),"-r", label="[0.2]")

Prob = np.array([0.3,0.7])


Mutual_inf = np.zeros(N)
for i in range(1,N+1):
    Ext_prob = Extended_Prob(Prob, i)
    _, Px_given_y = Py_given_x(Ext_prob)
    Px_given_y /= 1/(len(Prob)**i)
    
    Mutual_inf[i-1] = (Entropy(Ext_prob) - 1/(len(Prob)**i)*Entropy(Px_given_y))/i

plt.plot(np.arange(1,N+1),Mutual_inf/Entropy(Prob),"-k", label="[0.3]")

Prob = np.array([0.4,0.6])


Mutual_inf = np.zeros(N)
for i in range(1,N+1):
    Ext_prob = Extended_Prob(Prob, i)
    _, Px_given_y = Py_given_x(Ext_prob)
    Px_given_y /= 1/(len(Prob)**i)
    
    Mutual_inf[i-1] = (Entropy(Ext_prob) - 1/(len(Prob)**i)*Entropy(Px_given_y))/i

plt.plot(np.arange(1,N+1),Mutual_inf/Entropy(Prob),"-y", label="[0.4]")

plt.legend(loc="upper left")
plt.show()
"""





