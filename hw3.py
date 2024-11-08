import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
from numpy.polynomial.hermite import Hermite
from scipy.special import factorial

############################
# PART A
############################

starting_epsilon = 0.1
tol = 1e-4  # define a tolerance level
col = ['r', 'b', 'g', 'c', 'm', 'k']  # eigenfunction colors
yp = [-4, 4]
x = np.linspace(yp[0], yp[1], 81, endpoint=True)
A1 = np.zeros((81, 5))
A2 = np.zeros(5)

def system(x, y, epsilon):
    dy1dx = y[1]
    dy2dx = (x**2 - epsilon) * y[0]
    return [dy1dx, dy2dx]

for modes in range(1, 6):  # begin mode loop
    epsilon = starting_epsilon  # initial value of eigenvalue beta
    depsilon = starting_epsilon / 100  # default step size in beta
    for i in range(1000):  # begin convergence loop for beta
        y0 = [1, np.sqrt(16 - epsilon)]
        
        # Solve the system using solve_ivp
        sol = solve_ivp(system, [x[0], x[-1]], y0, args=(epsilon,), t_eval=x, method='RK45')
        
        if abs((sol.y[1, -1] + np.sqrt(16 - epsilon) * sol.y[0, -1]) - 0) < tol:  # check for convergence
            #print(f"Eigenvalue found for mode {modes}: {epsilon}")
            break  # exit convergence loop
        
        # Adjust epsilon based on the sign of the boundary condition
        if (-1) ** (modes + 1) * (sol.y[1, -1] + np.sqrt(16 - epsilon) * sol.y[0, -1]) > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon / 2
            depsilon /= 2

    A2[modes - 1] = epsilon
    starting_epsilon = epsilon + 0.1  # after finding eigenvalue, pick new start
    norm = np.trapz(sol.y[0] ** 2, x)  # calculate the normalization
    A1[:, modes - 1] = np.abs(sol.y[0] / np.sqrt(norm))
    plt.plot(x, sol.y[0] / np.sqrt(norm), col[modes - 1], label=f'Mode {modes}')  # plot modes

print(A2)
plt.legend()
plt.show()

############################
# PART B
############################
dx = 0.1
x = np.arange(-4, 4+dx, dx)
n = len(x)-2

# 2nd derivative matrix, Kutz Book pg. 186
A = np.zeros(shape=(n, n))
A += -2 * np.eye(n)
A += 1 * np.eye(n, k=1)
A += 1 * np.eye(n, k=-1)

A[0, 0] = -2/3
A[0, 1] = 2/3
A[-1, -1] = -2/3
A[-1, -2] = 2/3

A /= dx**2

P = np.diag(x[1:-1]**2)

eigenvalues, eigenvectors = np.linalg.eig(-A+P)

# Sort eigenvalues to get the smallest ones
order = np.argsort(eigenvalues)[:5]
A4 = eigenvalues[order]
print(A4)

# Get the respective eigenfunctions
A3 = np.zeros(shape=(len(x), 5))
A3[1:-1, :] = eigenvectors[:, order][:, :5]

# Check FDF & BDF coefficients here
A3[0,:] = (4/3)*A3[1,:] - (1/3)*A3[2,:]
A3[-1,:] = (4/3)*A3[-2,:] - (1/3)*A3[-3,:]
A3 = np.abs(A3 / np.sqrt(np.trapz(A3**2, x, axis=0)))

# Normalize and plot the first few eigenvectors
for i in range(5):
    plt.plot(x, A3[:, i])

plt.show()

############################
# PART C
############################

gamma_array = [-0.05, 0.05]
tol = 1e-4
dx = 0.1
x = np.arange(-2, 2+dx, dx)
pEig = []
pFunc = []
nEig = []
nFunc = []

# Define arrays for eigenvalues and eigenfunctions for each gamma
A5, A7 = np.zeros((41,2)), np.zeros((41,2)) 
A6, A8 = np.zeros(5), np.zeros(5)

def system(x, y, epsilon, gamma):
    dy1dx = y[1]
    dy2dx = (gamma * np.abs(y[0]) ** 2 + x ** 2 - epsilon) * y[0]
    return [dy1dx, dy2dx]

# Main loop over gamma values
for g, gamma in enumerate(gamma_array):
    starting_epsilon = 0.1
    A = 1e-6
    for modes in range(1,3): 
        dA = 0.01
        for i in range(100):  # Convergence loop for A
            epsilon = starting_epsilon
            depsilon = 0.2
            for j in range(100): #E loop
                y0 = [A, A*np.sqrt(4 - epsilon)]
                    
                # Solve the differential equation with solve_ivp
                sol = solve_ivp(system, [x[0], x[-1]], y0, args=(epsilon, gamma), t_eval = x)

                # Check convergence on boundary condition
                residual = sol.y[1, -1] + np.sqrt(4 - epsilon) * sol.y[0, -1]
                if abs(residual) < tol:
                    break
                # Adjust epsilon based on residual
                if (-1)**(modes+1)*residual > 0:
                    epsilon += depsilon
                else:
                    epsilon -= depsilon
                    depsilon/= 2
                    
            area = np.trapz(sol.y[0, :]**2, sol.t)
            if np.abs(area-1) < tol:
                break
            if area < 1:
                A+=dA
            else:
                A-=dA
                dA/=2

        starting_epsilon = epsilon + 0.2  # Update starting epsilon for next mode
        if gamma > 0:
            pEig.append(epsilon)
            pFunc.append(np.abs(sol.y[0]))    
        else:
            nEig.append(epsilon)
            nFunc.append(np.abs(sol.y[0]))
            
A5 = np.array(pFunc).T
A6 = np.array(pEig)
print(A6)
A7 = np.array(nFunc).T
A8 = np.array(nEig)
print(A8)
    
plt.plot(x, A5[:,0], color='g', label=f'Gamma {0.05}, Mode {1}')
plt.plot(x, A5[:,1], color='r', label=f'Gamma {0.05}, Mode {2}')
plt.show()

plt.plot(x, A7[:,0], color='b', label=f'Gamma {-0.05}, Mode {1}')
plt.plot(x, A7[:,1], color='m', label=f'Gamma {-0.05}, Mode {2}')
plt.show()

##########
# PART D #
##########

# Define a sample ODE system, e.g., dy/dt = -epsilon * y
def system(x, y, epsilon):
    dy1dx = y[1]
    dy2dx = (x ** 2 - epsilon) * y[0]
    return [dy1dx, dy2dx]

epsilon = 1
TOL = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
methods = ['RK45', 'RK23', 'Radau', 'BDF']
yp = [-2, 2]
x_span = (yp[0], yp[1])
y0 = [1, np.sqrt(3)]

# Array to store slopes
A9 = np.zeros((4,))

# Plot settings
plt.figure()

# Loop over methods
for idx, method in enumerate(methods):
    avg_step_sizes = []
    
    # Loop over tolerances
    for tol in TOL:
        options = {'rtol': tol, 'atol': tol}
        sol = solve_ivp(system, x_span, y0, method=method, args=(epsilon,), **options)
        
        # Calculate the average step size
        step_sizes = np.diff(sol.t)  # Differences between consecutive time points
        average_step_size = np.mean(step_sizes)
        avg_step_sizes.append(average_step_size)

    # Convert lists to arrays for polyfit
    log_avg_step_sizes = np.log10(avg_step_sizes)
    log_TOL = np.log10(TOL)

    # Calculate slope using polyfit and save it
    slope, _ = np.polyfit(log_avg_step_sizes, log_TOL, 1)
    A9[idx,] = slope

    # Plot the tolerances against the average step sizes for each method
    plt.loglog(avg_step_sizes, TOL, label=method, linewidth=2)

# Add labels, title, and legend
plt.xlabel('Average Step Size')
plt.ylabel('Tolerance')
plt.title('Log-Log Plot of Average Step Size vs. Tolerance for Different Methods')
plt.legend()
plt.show()

# Print the slopes array
print("Slopes for each method (RK45, RK23, Radau, BDF):")
print(A9)

##########
# PART E #
##########

# Get hermite polynomials from numpy
def H(n, x):
    hermite = Hermite(np.eye(n+1)[n])
    return hermite(x)

def phi(n, x):
    #Exact solution for the nth eigenfunction (starting from 0).
    #Note that m omega / h_bar = 1
    return 1 / (np.sqrt(2**n * factorial(n))) * (1/np.pi)**0.25 * np.exp(-x**2 / 2) * H(n, x)

dx =0.1
x = np.arange(-4, 4.1, dx)
phi_exact = np.zeros(shape=(81, 5))
eigs_exact = np.array([2*n+1 for n in range(5)])

for n in range(5): # 5 modes
    phi_exact[:, n] = phi(n, x)

# Check part A with exact solution
A10 = np.trapz((np.abs(A1) - np.abs(phi_exact))**2, dx=0.1, axis=0)
A11 = 100 * np.abs(A2 - eigs_exact) / eigs_exact

# Check part B with exact solution
A12 = np.trapz((np.abs(A3) - np.abs(phi_exact))**2, dx=0.1, axis=0)
A13 = 100 * np.abs(A4 - eigs_exact) / eigs_exact


