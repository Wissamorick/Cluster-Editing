import numpy as np
from math import *
from scipy import linalg
import pandas as pd
import numpy as np
import sklearn.metrics as sm
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
    print(indices)
    indices.sort()
    return indices

def find_max_indices(L): #returns the indices of the biggest element of L
    indices=[0]
    max=L[0]
    for i in range(len(L)):
        elt=L[i]
        if elt>max:
            indices=[i]
            max=elt
        elif elt==max:
            indices.append(i)
    return indices

def norm(v):
    return np.linalg.norm(v)


#STEP 1

a=input()
#print('taille de la premiere ligne = ',len(a))
compt=6
b=a.split()
n=int(b[2])
#print('n=',n)
e=int(b[3])
#print('e=',e)
Adj=np.zeros([n,n])
for i in range(e):
    edge=input()
    a,b=edge.split()
    a=int(a)
    b=int(b)
    Adj[a-1][b-1]=1
    Adj[b-1][a-1]=1
    #print(a,b,i)

'''
K=2
Adj=np.array([[0, 1, 0, 0, 1, 1, 1],
              [1, 0, 1, 1, 0, 0, 0],
              [0, 1, 0, 1, 0, 0, 0],
              [0, 1, 1, 0, 1, 0, 0],
              [1, 0, 0, 1, 0, 1, 1],
              [1, 0, 0, 0, 1, 0, 0],
              [1, 0, 0, 0, 1, 0, 0]])
              '''
'''
K=3
Adj=np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
              [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
              [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]])
'''
n=Adj[0].size

'''print("Matrice d adjacence:")
print(Adj)
print('\n')'''

#STEP 2
'''print("D=")
print(n)'''
D=np.zeros(n*n).reshape(n,n)
eps=1e-3
for i in range(n):
    D[i][i]=max(sum(Adj[i]),eps)
'''print(D)
print('\n')

print("L=")'''
Dbis=invsqrt(D)
L=Dbis@Adj@Dbis
'''print(L)'''

#STEP 3
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from sklearn.utils.fixes import lobpcg
from sklearn.cluster import KMeans



def _deterministic_vector_sign_flip(u):
    """Modify the sign of vectors for reproducibility.
    Flips the sign of elements of all the vectors (rows of u) such that
    the absolute maximum element of each vector is positive.
    Parameters
    ----------
    u : ndarray
        Array with vectors as its rows.
    Returns
    -------
    u_flipped : ndarray with same shape as u
        Array with the sign flipped vectors as its rows.
    """
    max_abs_rows = np.argmax(np.abs(u), axis=1)
    signs = np.sign(u[range(u.shape[0]), max_abs_rows])
    u *= signs[:, np.newaxis]
    return u

def _set_diag(laplacian, value, norm_laplacian):
    """Set the diagonal of the laplacian matrix and convert it to a
    sparse format well suited for eigenvalue decomposition.
    Parameters
    ----------
    laplacian : {ndarray, sparse matrix}
        The graph laplacian.
    value : float
        The value of the diagonal.
    norm_laplacian : bool
        Whether the value of the diagonal should be changed or not.
    Returns
    -------
    laplacian : {array, sparse matrix}
        An array of matrix in a form that is well suited to fast
        eigenvalue decomposition, depending on the band width of the
        matrix.
    """
    n_nodes = laplacian.shape[0]
    # We need all entries in the diagonal to values
    if not sparse.isspmatrix(laplacian):
        if norm_laplacian:
            laplacian.flat[::n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = (laplacian.row == laplacian.col)
            laplacian.data[diag_idx] = value
        # If the matrix has a small number of diagonals (as in the
        # case of structured matrices coming from images), the
        # dia format might be best suited for matvec products:
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            # 3 or less outer diagonals on each side
            laplacian = laplacian.todia()
        else:
            # csr has the fastest matvec and is thus best suited to
            # arpack
            laplacian = laplacian.tocsr()
    return laplacian


adjacency=Adj

Nmax=min(n,30)
values_K=list(range(2,Nmax))
values_editions=[]

liste_editions=[]
for i in range(n):
    for j in range(i):
        if Adj[i][j]==1:
            liste_editions.append([i,j])

for K in range(2,Nmax):
    try:
        editions=[]
        #print('K = ',K)
        n_components=K
        norm_laplacian=True
        laplacian, dd = csgraph_laplacian(adjacency, normed=norm_laplacian,
                                            return_diag=True)

        n_nodes = adjacency.shape[0]
        if n_nodes < 5 * n_components + 1:
            # see note above under arpack why lobpcg has problems with small
            # number of nodes
            # lobpcg will fallback to eigh, so we short circuit it
            if sparse.isspmatrix(laplacian):
                laplacian = laplacian.toarray()
            _, diffusion_map = eigh(laplacian, check_finite=False)
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                embedding = embedding / dd


        else:
            laplacian = _set_diag(laplacian, 1, norm_laplacian)
            # We increase the number of eigenvectors requested, as lobpcg
            # doesn't behave well in low dimension
            X = np.random.rand(laplacian.shape[0], n_components + 1)
            X[:, 0] = dd.ravel()
            _, diffusion_map = lobpcg(laplacian, X, tol=1e-15,
                                        largest=False, maxiter=2000)
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                embedding = embedding / dd
            if embedding.shape[0] == 1:
                raise ValueError

        embedding = _deterministic_vector_sign_flip(embedding)
        X=embedding[:n_components].T


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
        kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
        #print(kmeans.labels_)
        #print(len(set(kmeans.labels_)))
        #FORM THE NEW ADJACENCY MATRIX AND FIND THE NUMBER OF EDITIONS
        NewAdj=np.zeros([n,n])
        clusters=kmeans.labels_
        while max(clusters)>-1:
            cluster=find_max_indices(clusters)
            for index in cluster:
                for indexbis in cluster:
                    NewAdj[index][indexbis]=1
            for index in cluster:
                clusters[index]=-1

        editions_matrix=abs(NewAdj-Adj)
        for i in range(n):
            for j in range(i):
                if editions_matrix[i][j]==1:
                    editions.append([i,j])
        if len(editions)<len(liste_editions):
            liste_editions=editions
    except:
        pass

for arete in liste_editions:
    print(arete[0]+1,arete[1]+1)
#print("Number of editions : ", len(liste_editions))
