import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy.linalg import solve_banded
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Initialize the spiral initial conditions
def initialize_spiral(Nx, Ny, Lx, Ly, m):
    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)

    A = np.sqrt(X**2 + Y**2)
    theta = np.angle(X + 1j * Y)

    U = np.tanh(A) * np.cos(m * theta - A)
    V = np.tanh(A) * np.sin(m * theta - A)

    return U, V, X, Y

def initialize_spiral_cheb(x_phys, y_phys, m):
    X, Y = np.meshgrid(x_phys, y_phys)

    A = np.sqrt(X**2 + Y**2)
    theta = np.angle(X + 1j * Y)

    U = np.tanh(A) * np.cos(m * theta - A)
    V = np.tanh(A) * np.sin(m * theta - A)

    return U, V, X, Y

# Define the lambda and omega functions
def lambda_func(A2):
    return 1 - A2

def omega_func(A2, beta):
    return -beta * A2

# Reaction-diffusion system with FFT (periodic boundaries)
def reaction_diffusion_fft(t, y, Nx, Ny, Lx, Ly, D1, D2, beta):
    U_hat = y[:Nx * Ny].reshape((Nx, Ny))
    V_hat = y[Nx * Ny:].reshape((Nx, Ny))
    
    U = np.fft.ifft2(U_hat).real
    V = np.fft.ifft2(V_hat).real

    A2 = U**2 + V**2
    Lambda = lambda_func(A2)
    Omega = omega_func(A2, beta)

    # Compute spatial derivatives using FFT
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky)
    Laplacian = -(KX**2 + KY**2)

    dUdt_hat = D1 * Laplacian * U_hat + np.fft.fft2(Lambda * U - Omega * V)
    dVdt_hat = D2 * Laplacian * V_hat + np.fft.fft2(Omega * U + Lambda * V)

    #dUdt = np.real(ifft2(dUdt_hat))
    #dVdt = np.real(ifft2(dVdt_hat))

    return np.concatenate([dUdt_hat.ravel(), dVdt_hat.ravel()])

def chebyshev_matrix(N):
	if N==0: 
		D = 0.; x = 1.
	else:
		n = np.arange(0,N+1)
		x = np.cos(np.pi*n/N).reshape(N+1,1) 
		c = (np.hstack(( [2.], np.ones(N-1), [2.]))*(-1)**n).reshape(N+1,1)
		X = np.tile(x,(1,N+1))
		dX = X - X.T
		D = np.dot(c,1./c.T)/(dX+np.eye(N+1))
		D -= np.diag(np.sum(D.T,axis=0))
	return D, x.reshape(N+1)

# Reaction-diffusion system with Chebyshev (no-flux boundaries)
def reaction_diffusion_chebyshev(t, y, L, D1, D2, beta, X, Y):
    U, V = y[:len(y)//2], y[len(y)//2:]

    A2 = U**2 + V**2
    Lambda = lambda_func(A2)
    Omega = omega_func(A2, beta)

    Laplacian_U = D1*L @ U
    Laplacian_V = D2*L @ V

    dUdt = Laplacian_U + Lambda * U - Omega * V
    dVdt = Laplacian_V + Omega * U + Lambda * V

    return np.concatenate([dUdt.ravel(), dVdt.ravel()])

# Solve the reaction-diffusion system (periodic boundaries)
def solve_periodic(Nx, Ny, Lx, Ly, T, dt, D1, D2, beta):
    U, V, X, Y = initialize_spiral(Nx, Ny, Lx, Ly, m=1)
    #transformed initial conditions
    U = np.fft.fft2(U)
    V = np.fft.fft2(V)
    # Flatten initial conditions for the solver
    y0 = np.concatenate([U.ravel(), V.ravel()])

    # Solve the system
    sol = solve_ivp(
        reaction_diffusion_fft,
        [0, T],
        y0,
        args=(Nx, Ny, Lx, Ly, D1, D2, beta),
        t_eval=np.arange(0, 4.5, dt),
        method='RK45',
        vectorized=False
    )

    # Reshape solution for U and V
    U_sol = sol.y[:Nx * Ny].reshape((Nx, Ny, -1))
    V_sol = sol.y[Nx * Ny:].reshape((Nx, Ny, -1))

    return X, Y, U_sol, V_sol, sol.y, sol.t

# Solve the reaction-diffusion system (no-flux boundaries)
def solve_no_flux(N, Lx, Ly, T, dt, D1, D2, beta):
    # Chebyshev differentiation matrices
    D, x = chebyshev_matrix(N)

    x_phys = 0.5 * (Lx) * (x+1) -10
    y_phys = x_phys

    #impose no flux boundaries
    D[N,:] = 0
    D[0,:] = 0

    # Laplacian in 2D
    D_2 = np.dot(D, D) / (100)
    I = np.eye(len(D_2))
    L = np.kron(I, D_2) + np.kron(D_2, I)
    
    U, V, X, Y = initialize_spiral_cheb(y_phys, x_phys, m=1)
    # Flatten initial conditions for the solver
    y0 = np.concatenate([U.ravel(), V.ravel()])

    # Solve the system
    sol = solve_ivp(
        reaction_diffusion_chebyshev,
        [0, T],
        y0,
        args=(L, D1, D2, beta, X, Y),
        t_eval=np.arange(0, 4.5, dt),
        method='RK45',
    )

    # Reshape solution for U and V
    U_sol = sol.y[: (N + 1) * (N + 1)].reshape((N + 1, N + 1, -1))
    V_sol = sol.y[(N + 1) * (N + 1) :].reshape((N + 1, N + 1, -1))

    return X, Y, U_sol, V_sol, sol.y, sol.t

# Parameters for FFT
Nx, Ny = 64, 64  # Number of grid points
Lx, Ly = 20, 20  # Domain size
T = 4            # Total simulation time
dt =  0.5        # Time step
D1, D2 = 0.1, 0.1  # Diffusion coefficients
beta = 1         # Parameter for omega

# Run the simulation for periodic boundaries
X, Y, U_sol, V_sol, A1, times = solve_periodic(Nx, Ny, Lx, Ly, T, dt, D1, D2, beta)
U_sol_initial = np.fft.ifft2(U_sol[:,:,0]).reshape(Nx,Ny).real
U_sol_final = np.fft.ifft2(U_sol[:,:,-1]).reshape(Nx, Ny).real

# Plot the initial and final states
plt.figure(figsize=(12, 6))
#print(A1)
#print(A1.shape)

plt.subplot(1, 2, 1)
plt.contourf(X, Y, U_sol_initial, cmap='viridis')
plt.colorbar()
plt.title('Initial U Fourier')

plt.subplot(1, 2, 2)
plt.contourf(X, Y, U_sol_final, cmap='viridis')
plt.colorbar()
plt.title('Final U Fourier')

plt.tight_layout()
plt.show()

# Parameters for Chebyshev
N = 30           # Number of Chebyshev points per dimension
Lx, Ly = 20, 20  # Domain size
T = 4            # Total simulation time
dt = 0.5         # Time step
D1, D2 = 0.1, 0.1  # Diffusion coefficients
beta = 1        # Parameter for omega

# Run the simulation for no-flux boundaries
X, Y, U_sol, V_sol, A2, times = solve_no_flux(N, Lx, Ly, T, dt, D1, D2, beta)

# Plot the initial and final states
plt.figure(figsize=(12, 6))
print(A2)
print(A2.shape)

plt.subplot(1, 2, 1)
plt.contourf(X, Y, U_sol[:, :, 0], cmap='viridis')
plt.colorbar()
plt.title('Initial U (No-Flux)')

plt.subplot(1, 2, 2)
plt.contourf(X, Y, U_sol[:, :, -1], cmap='viridis')
plt.colorbar()
plt.title('Final U (No-Flux)')

plt.tight_layout()
plt.show()
