import numpy as np
import matplotlib.pyplot as plt

#SETUP:
#timestep = 0.1
#x ranges from -10 to 10.1
#y is the function we want to find the roots of
dx = 0.1
x = np.arange(-10,10+dx,dx)
y = x*np.sin(3*x)-np.exp(x)
plt.plot(x,y)
plt.axis([-10, 10, -10, 20])
plt.show() #verify that function looks correct, note important values of x

#OBSERVATIONS: we should be able to find ~8 roots for this function

#NEWTON-RAPHSON OVERHEAD:
# y_prime = dy/dx
# x_1 = initial guess
y_prime = np.sin(3*x) + 3*x*np.cos(3*x) - np.exp(x)
x_1 = -1.6

