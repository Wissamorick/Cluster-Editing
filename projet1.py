import numpy as np
from math import *
from scipy import linalg
import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
 


def invsqrt(D): #D^-(1/2) if D is diagonal
    n=len(D)
    for i in range(n):
        D[i][i]=1/sqrt(D[i][i])
    return D
    
def find_max(L): #return the index of the biggest element of L
    ind=0
    for i in range(1,len(L)):
        if L[i]>L[ind]:
            ind=i
    return ind
    
def find_maxs(L,k): #return the indices of the k biggest elements (in absolute value) of L

    indices=[]
    Labs=[] #L in absolute values
    
    for elt in L:
        Labs.append(abs(elt))
        
    for i in range(k):
        ind=find_max(Labs)
        indices.append(ind)
        Labs[ind]=-1
    return indices
    
def norm(v):
    return np.linalg.norm(v)
    

#STEP 1
K=2
Adj=np.array([[0, 1, 0, 0, 1, 1, 1],
              [1, 0, 1, 1, 0, 0, 0],
              [0, 1, 0, 1, 0, 0, 0],
              [0, 1, 1, 0, 1, 0, 0],
              [1, 0, 0, 1, 0, 1, 1],
              [1, 0, 0, 0, 1, 0, 0],
              [1, 0, 0, 0, 1, 0, 0]])
n=Adj[0].size
'''print("Matrice d adjacence:")
print(Adj)
print('\n')'''

#STEP 2
'''print("D=")
print(n)'''
D=np.zeros(n*n).reshape(7,7)
for i in range(n):
    D[i][i]=sum(Adj[i])
'''print(D)
print('\n')

print("L=")'''
Dbis=invsqrt(D)
L=Dbis@Adj@Dbis
'''print(L)'''

#STEP 3
values,V=linalg.eig(L) #V contains the eigenvectors stacked in columns
absvalues=abs(values)
indices=[]
VT=np.transpose(V) #transposed of V : each row is an eigenvector
XT=[] #XT will contain the k biggest eigenvectors stacked in columns
indices=find_maxs(values,K)
for i in indices:
    XT.append(VT[i])
X=np.transpose(XT)

#STEP 4
Y=[]
i=-1
for row in X:
    norme=norm(row)
    Y.append([])
    i+=1
    for elt in row:
        elt=elt/norme
        Y[i].append(elt)
Y=np.array(Y)

#STEP 5
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)



