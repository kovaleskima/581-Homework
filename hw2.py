import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

epsilon = 0.1;

# Define the ODE system as a first-order system
# y1 = y, y2 = y'
def ode_system(x, y, p, q, r):
    y1, y2 = y
    dy1dx = y2  # y' = y2
    dy2dx = q(x) * y1  # y'' = p(x)y' + q(x)y + r(x)
    return [dy1dx, dy2dx]

def solve_ivp_guess(guess):
    sol = solve_ivp(ode_system, [a, b], [alpha, guess], t_eval=np.linspace(a, b, 100), args=(p,q,r))
    print(sol.success)
    print(sol.message)
    print(sol.y.shape)
    beta = sol.y[0,-2]/(1+(sol.t[-1] - sol.t[-2])*np.sqrt(16-epsilon)) #impose rhs boundary
    return sol, beta  # return the value of y(b)

# Define the shooting method
def shooting_method(ode_system, p, q, r, a, b, alpha, guess, tol=1e-6):
   
    first_shot_sol, first_shot_beta = solve_ivp_guess(guess)
    solution = adjust(first_shot_sol, first_shot_beta, guess)

    return solution

# Define the parameters of the ODE
def p(x): return 0  # Coefficient for y'
def q(x): return (x**2-epsilon)  # Coefficient for y
def r(x): return 0  # Constant

def adjust(sol, beta, guess):
    dguess = 0.2
    iteration = 0
    while abs(sol.y[0,-1]-beta) > 1e-6 and iteration < 1000:
        if abs(sol.y[0,-1]-beta) == 0:
            return sol  # Exact solution found
        elif sol.y[0,-1]-beta > 0:
            guess -= dguess  # derivative is too small, walk up
        else:
            guess += dguess  # derivative is too large, walk down

        sol, beta = solve_ivp_guess(guess)
        dguess /= 2; #half the change in our guess
        iteration += 1

    return sol

# Boundary conditions
a = -4  # Left boundary x=a
b = 4  # Right boundary x=b
alpha = 0  # Boundary condition y(a) = alpha

# Initial guess for y'(a)
guess = 0.5

# Solve the ODE using the shooting method
solution = shooting_method(ode_system, p, q, r, a, b, alpha, guess)

# Plot the solution
x_vals = solution.t
y_vals = solution.y[0, :]
plt.plot(x_vals, y_vals, label="Shooting method solution")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Solution to the ODE using the Shooting Method")
plt.legend()
plt.grid(True)
plt.show()