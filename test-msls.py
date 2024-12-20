from sksparse.cholmod import cholesky
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import spsolve
import numpy as np
from joblib import Parallel, delayed

# Generate random sparse systems
batch_size = 3
n = 100

A = sparse_random(n, n, density=0.01, format='csr')
A = A + A.T

sparse_matrices = [sparse_random(n, n, density=0.01, format='csr') for _ in range(batch_size)]
b = np.random.rand(n)

spsolve(A, b =b)


# Solve in parallel
# solutions = Parallel(n_jobs=-1)(delayed(spsolve)(A, b) for A, b in zip(sparse_matrices, rhs_vectors))

# print("Solutions:", solutions)