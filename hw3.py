import numpy as np
from scipy.integrate import solve_bvp
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

epsilon = 0.1
x = np.linspace(-4, 4, 81)
x_int = np.linspace(-3.9, 3.9, 79)
dx = 0.1
n = 79

A = np.zeros((n,n))
A[0][0] = (-2-(x_int[0]**2)*(dx**2)) + (4/3)
A[0][1] = (2/3)
A[n-1][n-1] = (-2-(x_int[n-1]**2)*(dx**2)) + (4/3)
A[n-1][n-2] = (2/3)

for j in range(1, n-2):
    A[j][j] = (-2-(x_int[j]**2)*(dx**2))
    A[j][j-1] = 1
    A[j][j+1] = 1
    
A_sparse = csr_matrix(A)
        

print(A)