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

#OBSERVATIONS: we should be able to find ~8 different roots for this function

#NEWTON-RAPHSON:
# f = y but as a python function
# f_prime = dy/dx
# x = array of possible zeroes seeded with initial guess of -1.6
# acceptable tolerance: 10^-6
def f(x):
    return x*np.sin(3*x)-np.exp(x)

def f_prime(x):
    return np.sin(3*x) + 3*x*np.cos(3*x) - np.exp(x)

x = np.array([-1.6])

for i in range(1000):
    if abs(f(x[i]))<1e-6: #if our last guess was good enough, break the loop
        break
    else: #otherwise, compute a new x using newton raphson method
        x = np.append(
            x, x[i] - f(x[i])/f_prime(x[i])
        )

print(i) #what iteration were we on?
print(x[i]) #see what we got for x
print(f(x[i])) #sanity check that f(x) is sufficiently close to 0

#BISECTION METHOD:
# We will use the same function f = y
# left endpoint: xl = -0.7
# right endpoint: xr = -0.4
# acceptable tolerance: 10^-6
xl = -0.7
xr = -0.4

for j in range(1000):
    xc = (xr+xl)/2
    if f(xc) > 0:
        xl = xc
    else:
        xr = xc
    
    if abs(f(xc)) < 1e-6:
        print(j) #iteration count
        print(xc) #see what we got for x
        print(f(xc)) #sanity check that f(xc) is sufficiently close to 0

        break