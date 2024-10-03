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
iterations_1 = 1

for i in range(1000):
    fc = f(x[i])
    x_new = x[i] - f(x[i])/f_prime(x[i])
    x = np.append(x, x_new)
    if abs(fc)<1e-6: #if our last guess was good enough, break the loop
        break


iterations_1 += i
A1 = x

print(iterations_1) #what iteration were we on?
print(x[i]) #see what we got for x
print(f(x[i])) #sanity check that f(x) is sufficiently close to 0

#BISECTION METHOD:
# We will use the same function f = y
# left endpoint: xl = -0.7
# right endpoint: xr = -0.4
# acceptable tolerance: 10^-6
xl = -0.7
xr = -0.4
x_result = []
iterations_2 = 1

for j in range(1000):
    xc = (xr+xl)/2
    x_result.append(xc)
    if f(xc) > 0:
        xl = xc
    else:
        xr = xc
    
    if abs(f(xc)) < 1e-6:
        iterations_2 += j #iteration count
        print(iterations_2)
        print(xc) #see what we got for x
        print(f(xc)) #sanity check that f(xc) is sufficiently close to 0

        break

A2 = x_result
A3 = [iterations_1, iterations_2]

A = np.array([[1,2],[-1,1]])
B = np.array([[2,0],[0,2]])
C = np.array([[2,0,-3],[0,0,-1]])
D = np.array([[1,2],[2,3],[-1,0]])
x = np.array([1,0])
y = np.array([0,1])
z = np.array([1,2,-1])

A4 = A+B
A5 = 3*x-4*y
A6 = A@x
A7 = B@(x-y)
A8 = D@x
A9 = (D@y) + z
A10 = A@B
A11 = B@C
A12 = C@D