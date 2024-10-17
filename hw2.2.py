import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

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
print(A1)
print(A2)