import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 81, endpoint=True)
dx = x[1] - x[0]
n = len(x)

# Define the diagonals for the tridiagonal matrix
main_diag = (-2 - (x**2) * (dx**2))  # Main diagonal
off_diag = np.ones(n-1)  # Off diagonals (ones)

# Create the sparse matrix using the diagonals
diagonals = [off_diag, main_diag, off_diag]
A = diags(diagonals, [-1, 0, 1], format='csr')

# Find the smallest 5 eigenvalues and eigenvectors
eigenvalues, eigenvectors = eigsh(A, k=5, which='SM')
eigenvalues = abs(eigenvalues[::-1])*100 
eigenvectors = eigenvectors[:, ::-1]

# Normalize and plot the first few eigenvectors
for i in range(5):
    norm = np.trapz(eigenvectors[:, i] ** 2, x)
    plt.plot(x, eigenvectors[:, i] / np.sqrt(norm), label=f'Mode {i+1}')

plt.legend()
plt.show()

print("Eigenvalues:", eigenvalues)