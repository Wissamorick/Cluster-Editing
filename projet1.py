import numpy as np
from math import *
from scipy import linalg

def invsqrt(D): #D^-(1/2) if D is diagonal
    n=len(D)
    for i in range(n):
        D[i][i]=1/sqrt(D[i][i])
    return D

#STEP 1*
K=2
Adj=np.array([[0, 1, 0, 0, 1, 1, 1],
              [1, 0, 1, 1, 0, 0, 0],
              [0, 1, 0, 1, 0, 0, 0],
              [0, 1, 1, 0, 1, 0, 0],
              [1, 0, 0, 1, 0, 1, 1],
              [1, 0, 0, 0, 1, 0, 0],
              [1, 0, 0, 0, 1, 0, 0]])
n=Adj[0].size
print("Matrice d adjacence:")
print(Adj)
print('\n')

#STEP 2
print("D=")
print(n)
D=np.zeros(n*n).reshape(7,7)
for i in range(n):
    D[i][i]=sum(Adj[i])
print(D)
print('\n')

print("L=")
Dbis=invsqrt(D)
L=Dbis@Adj@Dbis
print(L)

#STEP 3
values,V=linalg.eig(L)
absvalues=abs(values)
indices=[]
