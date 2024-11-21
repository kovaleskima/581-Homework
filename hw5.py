from scipy.sparse import spdiags, csr_matrix
from scipy.sparse.linalg import bicgstab, gmres, spsolve, splu
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from scipy.linalg import lu, solve_triangular
import time

#############
# FUNCTIONS #
#############

def fft_for_psi(omega):
    omega_hat = np.fft.fft2(omega)

    kx = np.fft.fftfreq(N, dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, dx) * 2 * np.pi
    kx[0] = 1e-6
    ky[0] = kx[0]
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2

    psi_hat = -omega_hat / K2
    psi = np.real(np.fft.ifft2(psi_hat))

    return psi

def LU_for_psi(A, w):
    A[0,0] = 2
    lu_solver = splu(A)
    psi = lu_solver.solve(w)

    return psi

def stab_for_psi(A, w):
    A[0,0] = 2
    psi, info = bicgstab(A, w, tol=1e-6, maxiter=1000)
    
    # Check if the solver converged
    if info == 0:
        print("BiCGSTAB converged successfully.")
    elif info > 0:
        print(f"BiCGSTAB reached the maximum number of iterations. Current iteration: {info}")
    else:
        print("BiCGSTAB failed to converge.")
    
    return psi

def gmres_for_psi(A, w):
    A[0,0] = 2
    psi, info = gmres(A, w, tol=1e-6, restart=50, maxiter=1000)
    
    # Check if the solver converged
    if info == 0:
        print("GMRES converged successfully.")
    elif info > 0:
        print(f"GMRES reached the maximum number of iterations. Current iteration: {info}")
    else:
        print("GMRES failed to converge.")
    
    return psi

def rref_for_psi(A, w):
    # Solve the linear system A * psi = omega
    A[0,0] = 2
    psi = spsolve(A, w)  # Direct solve using sparse matrix
    
    return psi

def system(t, w, A, B, C, nu, N, method):
    w_flat = w
    w = w.reshape(N, N)
    if method == 'fft':
        psi = fft_for_psi(w)
    elif method == 'rref':
        psi = rref_for_psi(A, w_flat)
    elif method == 'LU':
        psi = LU_for_psi(A, w_flat)
    elif method == 'stab':
        psi = stab_for_psi(A, w_flat)
    else:
        psi = gmres_for_psi(A, w_flat)

    psi_flat = psi.flatten()

    # Compute the terms and reshape back to 2D before elementwise multiplication
    term1 = (C @ psi_flat).reshape(N, N) * (B @ w_flat).reshape(N, N)
    term2 = (B @ psi_flat).reshape(N, N) * (C @ w_flat).reshape(N, N)
    diffusion_term = (nu * A @ w_flat).reshape(N, N)

    # Calculate dwdt
    dwdt = term1 - term2 + diffusion_term

    # Flatten the result for solve_ivp
    return dwdt.flatten()

########################
# DEFINITIONS & PARAMS #
########################

# Set up spatial grid
N = 64
L = 20
nu = 0.001
dx = L/N
dy = L/N
x2 = np.linspace(-10, 10, N+1)
x = x2[:64]
y2 = np.linspace(-10, 10, N+1)
y = y2[:64]
t = np.linspace(0, 4, 9, endpoint=True)
X, Y = np.meshgrid(x, y)
method = ['fft', 'rref', 'LU']

#######################
# DERIVATIVE MATRICES #
#######################

# MATRIX A
m = N # N value in x and y directions
n = m * m # total size of matrix
e0 = np.zeros((n, 1)) # vector of zeros
e1 = np.ones((n, 1)) # vector of ones
e2 = np.copy(e1) # copy the one vector
e4 = np.copy(e0) # copy the zero vector
for j in range(1, m+1):
    e2[m*j-1] = 0 # overwrite every m^th value with zero
    e4[m*j-1] = 1 # overwirte every m^th value with one
# Shift to correct positions

e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]
e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]
# Place diagonal elements
diagonals = [e1.flatten(), e1.flatten(), e5.flatten(),
e2.flatten(), -4 * e1.flatten(), e3.flatten(),
e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]
A = (1/(dx**2))*spdiags(diagonals, offsets, n, n).toarray()
A = csr_matrix(A)

# MATRIX B
e1 = np.ones((n, 1))
e2 = np.ones((n, 1))
e3 = np.ones((n,1))
e4 = np.ones((n,1))

diagonals = [e1.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets = [(-n+m), -m, m, n-m]
B = spdiags(diagonals, offsets, n, n,).toarray()
B /= (2*dx)
B = csr_matrix(B)

# MATRIX C
e1 = np.zeros((n, 1))
e2 = np.ones((n, 1))
e3 = np.ones((n,1))
e4 = np.zeros((n,1))
for index in range(m):
    e1[m*index] = 1
    e2[m*index + m-1] = 0
    e3[m*index] = 0
    e4[m*index+ m-1] = 1

diagonals = [e1.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets = [(-m+1), -1, 1, m-1]
C = spdiags(diagonals, offsets, n, n,).toarray()
C /= (2*dy)
C = csr_matrix(C)

###############
# BEGIN SOLVE #
###############

# Define vorticity field (Gaussian bump)
#omega_0 = np.exp(-X**2 - (Y**2)/L)
#omega_0 = np.exp(-(X+0.5)**2 - (Y**2)/L) - np.exp(-(X-0.5)**2 - (Y**2)/L)
#omega_0 = np.exp(-(X+1)**2 - (Y**2)/L) + np.exp(-(X-1)**2 - (Y**2)/L)
omega_0 = np.exp(-(X+1)**2 - ((Y+1)**2)/L) + np.exp(-(X-1)**2 - ((Y+1)**2)/L) + \
    np.exp(-(X+1)**2 - ((Y-1)**2)/L) + np.exp(-(X-1)**2 - ((Y-1)**2)/L)


# Solve the system using solve_ivp
for m in range(len(method)):
    start_time = time.time() # Record the start time
    omega = solve_ivp(system, [t[0], t[-1]], omega_0.flatten(), args=(A, B, C, nu, N, method[m]), t_eval=t, method='RK45')

    # Store answers
    if method[m] == 'fft':
        A1 = omega.y
        print(A1.shape)
    elif method[m] == 'rref':
        A2 = omega.y
        print(A2.shape)
    else:
        A3 = omega.y
        print(A3.shape)

    end_time = time.time() # Record the end time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    omega_t_series = omega.y.reshape(N, N, len(t))

    # Set up the figure and initial plot
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, omega_t_series[:, :, 0], levels=50, cmap="viridis")
    colorbar = fig.colorbar(contour, ax=ax, label="Vorticity Omega")
    ax.set_title("Vorticity Evolution for ∇²ψ = ω (using FFT)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Function to update the plot for each frame
    def update(frame):
        ax.clear()
        contour = ax.contourf(X, Y, omega_t_series[:, :, frame], levels=50, cmap="viridis")
        ax.contour(X, Y, omega_t_series[:, :, frame], levels=15, colors="k", linewidths=0.5)
        ax.set_title(f"Vorticity at t = {t[frame]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return contour

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(t), repeat=False)

    # Show or save the animation
    plt.show()
np.save('maddy-answers-correct', np.concatenate((A1, A2, A3)))