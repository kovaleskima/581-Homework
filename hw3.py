import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

def bvp_rhs2(x, y, epsilon):
    return np.vstack((y[1], (x**2 - epsilon) * y[0]))

def bvp_bc2(yl, yr, epsilon):
    print(epsilon, type(epsilon))
    return np.array([yl[0], yl[0]*np.sqrt(16-float(epsilon)), yr[0], -yr[0]*np.sqrt(16-float(epsilon))])

def mat4init(x):
    return np.array([np.cos((np.pi/2)*x),-(np.pi / 2)*np.sin((np.pi/2)*x)])

epsilon = 0.1
x_init = np.linspace(-4, 4, 80)
init2 = solve_bvp(bvp_rhs2, bvp_bc2, x_init, mat4init(x_init), p=[epsilon])
x2 = np.linspace(-1, 1, 100)
BS2 = init2.sol(x2)
plt.plot(x2, BS2[0])