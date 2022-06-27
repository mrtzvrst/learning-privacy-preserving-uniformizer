import numpy as np
from FUNCs import Entropy
from itertools import combinations, product
import random

#np.array([0.31, 0.24, 0.23, 0.22])
#np.array([0.26, 0.26, 0.26, 0.22])
#[0.36, 0.24, 0.24, 0.16]
#[0.45, 0.45, 0.05, 0.05]

PX_vector = np.array([0.36, 0.24, 0.24, 0.16])
alpha = 0.05 # Learning rate
gm = 1 # Discount factor
epsil = 0.5 # eps-greedy policy
Num_of_Episodes = 100

M = len(PX_vector)
PXY = np.zeros([M,M])
PX = PX_vector.copy() #Probability_vector copy
bucket = np.ones(M)/M #Uniform dist on the right

"""At this stage we update PXY, PX, bucket according to the masspoints in PX more than 1/M"""
ind = 0
for i in range(M):
    while PX[i]>=1/M:
        PXY[i,ind] = 1/M
        PX[i]-=1/M
        bucket[ind] = 0.0
        ind+=1

Non_empty_bucket_ind = np.where(bucket>0)[0]
if len(Non_empty_bucket_ind)==1:
    PXY [:,M-1] = PX 
    Out_matrix = PXY

else:
    """In temp_1 we obtain different combinations of availabel buckets"""
    temp_1 = []
    for i in range(1,len(Non_empty_bucket_ind)+1):
        temp_1 += list(combinations(Non_empty_bucket_ind, i))
    for i in range(len(temp_1)):
        temp_1[i] = set(temp_1[i])
    """Obtain the whole possible state as list of tuples"""
    STATEs = list(product(np.where(PX>0)[0] , temp_1))
    for i in range(len(STATEs)):
        STATEs[i] = list(STATEs[i])
    """Form Q table and initializw the table values to zero"""
    Q_matrix = {}
    for i in range(len(STATEs)):    
        Q_matrix[str(STATEs[i])] = dict(zip(STATEs[i][1],np.zeros(len(STATEs[i][1]))))
    del temp_1, STATEs # delete the unnecessary variables
    
    
    
    MI = 0
    counter = 0
    while counter < Num_of_Episodes:
        """we initialize every time we start a new episode"""
        rem_bucket = set(Non_empty_bucket_ind) # these are indices
        rem_PX = np.where(PX>0)[0].copy() # these are indices
        Len = len(rem_bucket)
              
        PXY_rep = PXY.copy()
        PX_rep = PX.copy()
        bucket_rep = bucket.copy()
        
        """We pick S_old randomly and remove the corresponding mass point from rem_PX (remaining PX)"""
        temp_1 = np.random.choice(rem_PX,1)[0]
        S_old = str([temp_1, rem_bucket])
        # rem_PX = np.delete(rem_PX, np.where(rem_PX==temp_1))
        
                
        #minus one is due to the remaining maximum state that we dealt with in the above
        for i in range(Len):
                                   
            if i < Len-1:
                available_actions = np.array(list(dict.keys(Q_matrix[S_old])))
                """epsil-greedy policy"""
                if len(available_actions) == 1:
                    action_taken = available_actions[0]
                else:
                    Indic = np.random.binomial(1,1-epsil) #greedy_action_indicator
                    greedy_action = max(Q_matrix[S_old], key=Q_matrix[S_old].get)
                    action_taken = Indic*greedy_action + (1-Indic)*np.random.choice(np.delete(available_actions, np.where(greedy_action)), size=1)
                               
                PXY_rep[eval(S_old)[0], action_taken] = PX_rep[eval(S_old)[0]] # bucket[Action]
                bucket_rep[action_taken] = 1/M - PX_rep[eval(S_old)[0]]
                PX_rep[eval(S_old)[0]] = 0
                
                
                """Update the new state space with respect to a target policy 
                    (we consider a random state selctor)"""
                rem_PX = np.delete(rem_PX, np.where(rem_PX==temp_1))
                temp_1 = np.random.choice(rem_PX,1)[0]
                S_new = str([temp_1, set(np.where(bucket_rep>PX_rep[temp_1])[0])])
                #rem_PX = np.delete(rem_PX, np.where(rem_PX==temp_1))
                
                R=0
                
                """Update Q values"""
                Q_matrix[S_old][action_taken[0]] += alpha*(R+gm*max(Q_matrix[S_new].values())-Q_matrix[S_old][action_taken[0]])
                S_old = S_new
            
            
            else:
                action_taken = np.array(list(dict.keys(Q_matrix[S_old])))[0]
                
                PXY_rep[eval(S_old)[0], action_taken] = PX_rep[eval(S_old)[0]] # bucket[Action]
                bucket_rep[action_taken] = 1/M - PX_rep[eval(S_old)[0]]
                PX_rep[eval(S_old)[0]] = 0
                
                last_idx = np.argmax(PX_rep)
                for i in range(M):
                    PXY_rep[last_idx, i] += 1/M - np.sum(PXY_rep[:,i]) 
                    
                if np.abs(np.sum(PXY_rep[last_idx,:]) - PX_rep[last_idx])<10^-10:
                    print("error happened")
                R = Entropy(PX)+np.log2(M)-Entropy(PXY_rep)
                MI = max(R,MI)
                print(MI)
                Q_matrix[S_old][action_taken] += alpha*(R-Q_matrix[S_old][action_taken])
                
            
                
            #state_set = np.delete(state_set, np.where(state_set==S_old)) 
        
        counter+=1
        """
        if counter%100 == 0:
            epsil /= 1.2
            print(epsil)
            """

LEN_buc = len(rem_bucket)

optimal = dict(zip(rem_bucket,np.zeros(len(rem_bucket))))
temp_3 = []
for j in reversed(range(LEN_buc)):
    temp_1 = 0
    temp_2 = []
    for k in Q_matrix.keys():        
        if len(Q_matrix[k])==j+1:
            if max(Q_matrix[k].values())>temp_1:
                temp_1 = max(Q_matrix[k].values())
                """k[1] is the mass point being connected to the bucket (max(...))"""
                temp_2 = [eval(k[1]), max(Q_matrix[k], key=Q_matrix[k].get)]
    temp_3 += [temp_2]
                  
print(temp_3)




   
    

    
    
    
    
    
    
    
    