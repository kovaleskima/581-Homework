import numpy as np
import matplotlib.pyplot as plt

# Assume N and dx given
N = 200
dx = 0.1

L = 4  # domain size
N = 200  # discretization of interior
x = np.linspace(-L, L, N + 2)  # add boundary points
dx = x[1] - x[0]  # compute dx

# NOTE: Assuming B1 is computed from previous code
# Compute P matrix
P = np.zeros((N, N))
for j in range(N):
    P[j, j] = x[j + 1] ** 2  # potential x^2

# Compute linear operator
linL = -B1 + P

# Compute eigenvalues and eigenvectors
D,V = eig(linL)

sorted_indices = np.argsort(np.abs(D))[::-1] 
Dsort = D[sorted_indices]
Vsort =V[:, sorted_indices]

D5 = Dsort[N-5:N]
V5 = Vsort[:,N-5:N]

print("Eigenvectors:")
print(V.shape)
print("Eigenvalues:")
print(D.shape)

plt.plot(V5)