import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


N = 200
dx = 1/N

#First Derivative Matrix
A = np.zeros((N,N))
for j in range (N-1):
    A[j,j+1] = 1
    A[j+1,j] = -1
A1 = A/(2*dx)

#Neumann BC
A2 = np.copy(A)
A2[0,0] = -4/3
A2[0,1] = 4/3
A2[N-1,N-1] = 4/3
A2[N-1, N-2] = -4/3

#Fill in second derivative matrices and make copies of this file to reuse