import numpy as np
from math import *

import pandas as pd
import sklearn.metrics as sm
from sklearn.cluster import KMeans
from sklearn import datasets
from scipy import sparse
from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from sklearn.utils.fixes import lobpcg
from sklearn.utils.extmath import _deterministic_vector_sign_flip

import time

def find_max(L): #return the index of the biggest element of L
    ind=0
    for i in range(1,len(L)):
        if L[i]>L[ind]:
            ind=i
    return ind

def norm(v):
    return np.linalg.norm(v)

def _set_diag(laplacian, value, norm_laplacian): #From sklearn : https://github.com/scikit-learn/scikit-learn/blob/4b53fc3f67fa6d7966bd51db7c9d754cd187d48f/sklearn/manifold/_spectral_embedding.py#L145
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

#STEP 1
t=time.time()

a=input()
b=a.split()
n=int(b[2])
#print('n=',n)
e=int(b[3])
#print('e=',e)

eps=1e-3 #avoid division by 0 when inverting D and there are isolated vertices (see Remark after Definition 2.1)

row=[]
col=[]
data=[]
liste_editions=[] #By default, a solution is to remove all the edges, so the list of the editions can be the list of the edges
for i in range(e):
    edge=input()
    a,b=edge.split()
    a=int(a)
    b=int(b)
    if (a==b):
           continue
    row.append(a-1) #Input vertices are from 1 to n, but in Python indices are from 0 to n-1 so let's use Python's indices
    col.append(b-1)
    data.append(1)
    row.append(b-1)
    col.append(a-1)
    data.append(1)
    liste_editions.append([a-1,b-1])
    #print(a,b,i)
rowcopy=row #to use .count() at line 110
row=np.array(row)
col=np.array(col)
data=np.array(data)
Adj=csr_matrix((data,(row,col)), shape=(n,n)) #Adjacency matrix
#D=[max(eps,row.count(i)) for i in range(n)]
diagindices=np.arange(n)
#D=csr_matrix((D,(diagindices,diagindices)),shape=(n,n))
invsqrtD=[1/sqrt(max(eps,rowcopy.count(i))) for i in range(n)] #D^{-1/2}
invsqrtD=csr_matrix((invsqrtD,(diagindices,diagindices)),shape=(n,n))
L=invsqrtD*Adj*invsqrtD
e=len(liste_editions) #updating e after eliminating auto-edges
adjacency=Adj

t=time.time()

eigenval,eigenvect=eigsh(L,k=n-1) #Let's compute L's eigenvalues and sort them
evs=np.sort(abs(eigenval))
#plt.plot(np.arange(n-1),evs,'bo')
#plt.show()

gaps=np.array([evs[i+1]-evs[i] for i in range(n-2)]) #Computing the gaps between eigenvalues (Section 3.5 Rounding)
#print(gaps)
Ktries=np.argsort(gaps)[-min(n//3,500):] #Values of K corresponding to the gaps
#print(Ktries)
nbr_editions=e #Removing all edges = a solution with e editions

itermax=len(Ktries)-1
iter=itermax+1
Kmin=(n*n)//e #Lemma 3.1 in Section 3.3
tmax=300 #Half of 10 min
while ((iter>0) and (time.time()-t<tmax)) : #if computing the eigenvalues lasted more than half the time we stop, else we can go further
    tmax=500 
    iter-=1
    K=n-2-Ktries[iter]
    if (K<Kmin):
        continue
    #print(time.time()-t)
    #print('K = ',K,'iter = ',iter)
    n_components=min(K,20) #We may change this number to compute more or less eigenvectors (e.g. K//3, K//10, 20, 100)
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
        try:
            _, diffusion_map = lobpcg(laplacian, X, tol=1e-15,
                                    largest=False, maxiter=2000)
        except:
            continue
        embedding = diffusion_map.T[:n_components]
        if norm_laplacian:
            embedding = embedding / dd
        if embedding.shape[0] == 1:
            #raise ValueError
            continue

    embedding = _deterministic_vector_sign_flip(embedding)
    X=embedding[:n_components].T

    #STEP 5
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    #print(kmeans.labels_)
    #print(len(set(kmeans.labels_)))
    
    #FORM THE NEW ADJACENCY MATRIX AND FIND THE NUMBER OF EDITIONS
    #NewAdj=np.zeros([n,n])
    clusters=kmeans.labels_
    row=[]
    col=[]
    data=[]
    while max(clusters)>-1:
        cluster=(np.array(clusters)==max(clusters)).nonzero()[0]
        for index in cluster:
            for indexbis in cluster:
                #NewAdj[index][indexbis]=1
                row.append(index)
                col.append(indexbis)
                data.append(1)
        for index in cluster:
            clusters[index]=-1
    row=np.array(row)
    col=np.array(col)
    data=np.array(data)
    NewAdj=csr_matrix((data,(row,col)),shape=(n,n))

    editions_matrix=abs(NewAdj-Adj) #1 if edited edge, 0 else
    nb_editions_K=(editions_matrix.getnnz()-n)//2

    if (nbr_editions>nb_editions_K): #A.getnnz() = number of nonzeros elements, i.e. edited edges
        liste_editions=[]
        nbr_editions=nb_editions_K
        (r,c)=editions_matrix.nonzero()
        for it in range(len(r)):
            #print(it)
            if (r[it]>c[it]) : #we don't want edges in double "a-b / b-a" or "a-a"
                liste_editions.append([r[it],c[it]])
for arete in liste_editions:
    print(arete[0]+1,arete[1]+1)
#print("Number of editions : ", len(liste_editions))
#print(time.time()-t)
