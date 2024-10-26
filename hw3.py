import numpy as np
from scipy.integrate import odeint
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

############################
# PART A
############################

starting_epsilon = 0.1
tol = 1e-4 # define a tolerance level
col = ['r', 'b', 'g', 'c', 'm', 'k'] # eigenfunc colors
yp = [-4, 4]
x = np.linspace(yp[0], yp[1],81, endpoint=True)
A1 = np.zeros((81,5))
A2 = np.zeros(5)


def system(y, x, epsilon):
    dy1dx = y[1]
    dy2dx = (x**2-epsilon)*y[0]
    return dy1dx, dy2dx

for modes in range(1, 6): # begin mode loop
    epsilon = starting_epsilon # initial value of eigenvalue beta
    depsilon = starting_epsilon / 100 # default step size in beta
    for i in range(1000): # begin convergence loop for beta
        y0 = [1, np.sqrt(16-epsilon)]
        y = odeint(system, y0, x, args=(epsilon,))
        if abs((y[-1, 1] + np.sqrt(16-epsilon)*y[-1,0]) - 0) < tol: # check for convergence
            print(epsilon) # write out eigenvalue
            break # get out of convergence loop
        if (-1) ** (modes + 1) * (y[-1, 1] + np.sqrt(16-epsilon)*y[-1,0]) > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon / 2
            depsilon /= 2
    A2[modes-1] = epsilon
    starting_epsilon = epsilon + 0.1 # after finding eigenvalue, pick new start
    norm = np.trapz(y[:, 0] * y[:, 0], x) # calculate the normalization
    A1[:,modes-1] = np.abs(y[:, 0] / np.sqrt(norm))
    plt.plot(x, y[:, 0] / np.sqrt(norm), col[modes - 1]) # plot modes

plt.show()

############################
# PART B
############################

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
eigenvalues = abs(eigenvalues[::-1])*(dx**2) 
eigenvectors = eigenvectors[:, ::-1]

# Normalize and plot the first few eigenvectors
for i in range(5):
    norm = np.trapz(eigenvectors[:, i] ** 2, x)
    plt.plot(x, eigenvectors[:, i] / np.sqrt(norm), label=f'Mode {i+1}')

plt.legend()
plt.show()

A3 = np.abs(eigenvectors[:, 0] / np.sqrt(norm))
A4 = eigenvalues
print("Eigenvalues:", eigenvalues)

############################
# PART C
############################

starting_epsilon = 0.1
gamma_array = [-0.05, 0.05]
tol = 1e-4
colors = ['r', 'b', 'g', 'c', 'm', 'k']  # More colors for more modes if needed
yp = [-2, 2]
x = np.linspace(yp[0], yp[1], 41, endpoint=True)

# Define arrays for eigenvalues and eigenfunctions for each gamma
A5, A6 = None, None  # for gamma = 0.5
A7, A8 = None, None  # for gamma = -0.5

def system(y, x, epsilon, gamma):
    dy1dx = y[1]
    dy2dx = gamma * y[0] ** 2 + (x ** 2 - epsilon) * y[0]
    return [dy1dx, dy2dx]

# Main loop over gamma values
for g, gamma in enumerate(gamma_array):
    epsilon = starting_epsilon
    
    for modes in range(2):  # Assume one mode per gamma in this example
        depsilon = starting_epsilon / 100
        
        for i in range(1000):  # Convergence loop for epsilon
            y0 = [1, np.sqrt(16 - epsilon)]
            y = odeint(system, y0, x, args=(epsilon, gamma))

            # Check convergence on boundary condition
            residual = (y[-1, 1] + np.sqrt(16 - epsilon) * y[-1, 0])
            if abs(residual) < tol:
                print(f"Converged epsilon for gamma={gamma}: {epsilon}")
                if gamma == 0.05:
                    A5 = y[:, 0] / np.sqrt(np.trapz(y[:, 0] ** 2, x))
                    A6 = epsilon
                elif gamma == -0.05:
                    A7 = y[:, 0] / np.sqrt(np.trapz(y[:, 0] ** 2, x))
                    A8 = epsilon
                break

            # Adjust epsilon based on residual
            if (-1) ** (modes + 1) * residual > 0:
                epsilon += depsilon
            else:
                epsilon -= depsilon / 2
                depsilon /= 2

        starting_epsilon = epsilon + 0.1  # Update starting epsilon for next mode
        norm = np.trapz(y[:, 0] ** 2, x)  # Normalize the eigenfunction
        normalized_eigenfunction = y[:, 0] / np.sqrt(norm)
        
        plt.plot(x, normalized_eigenfunction, color=colors[g], label=f'Gamma {gamma}, Mode {modes+1}')

plt.legend()
plt.show()

############################
# PART D
############################