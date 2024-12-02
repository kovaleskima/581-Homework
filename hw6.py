from scipy.sparse import spdiags, csr_matrix
from scipy.sparse.linalg import bicgstab, gmres, spsolve, splu
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from scipy.linalg import lu, solve_triangular
import time

def fast_fourier(omega):
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

def chebychev(omega):
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

########################
# DEFINITIONS & PARAMS #
########################

# define params
N = 64
L = 20
dx = L/N
dy = L/N
m = 1
beta = 1
D1 = 0.1
D2 = D1
# 1 index it because Nathan said so
x2 = np.linspace(-10, 10, N+1)
x = x2[:64]
y2 = np.linspace(-10, 10, N+1)
y = y2[:64]
# build grid
t = np.linspace(0, 4, 9, endpoint=True)
X, Y = np.meshgrid(x, y)
u = np.tanh(np.sqrt(X**2 + Y**2)) * np.cos(m * np.angle(X + 1j * Y) - np.sqrt(X**2 + Y**2))
v = np.tanh(np.sqrt(X**2 + Y**2)) * np.sin(m * np.angle(X + 1j * Y) - np.sqrt(X**2 + Y**2))
A2 = (u**2) + (v**2)
lambda_A = np.ones_like(A2)-A2
omega_A = -beta * A2