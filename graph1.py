from math import *
import numpy as np
import matplotlib.pyplot as plt

def leftpoint_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b - h, n)
    return h * np.sum(f(x))

def midpoint_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a + h/2, b - h/2, n)
    return h * np.sum(f(x))


def f(x):
    return np.sin(x)

def true_integral(x):
    return -np.cos(x)

a = 0
b = 1
n = np.array([2**i for i in range(25)])
true_value = true_integral(b) - true_integral(a)

leftpoint_errors = []
midpoint_errors = []
for i in range(len(n)):
    leftpoint_errors.append(np.abs(leftpoint_rule(f, a, b, n[i]) - true_value))
    midpoint_errors.append(np.abs(midpoint_rule(f, a, b, n[i]) - true_value))

def rightpoint_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a + h, b, n)
    return h * np.sum(f(x))

rightpoint_errors = []
for i in range(len(n)):
    rightpoint_errors.append(np.abs(rightpoint_rule(f, a, b, n[i]) - true_value))

def trapezoid_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    return h * (f(a) + 2*np.sum(f(x[1:-1])) + f(b)) / 2

trapezoid_errors = []
for i in range(len(n)):
    trapezoid_errors.append(np.abs(trapezoid_rule(f, a, b, n[i]) - true_value))

def simpson_rule(f, a, b, n):
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n, 2):
        s += 4 * f(a + i * h)
    for i in range(2, n-1, 2):
        s += 2 * f(a + i * h)
    return h / 3 * s

simpson_errors = []
for i in range(len(n)):
    simpson_errors.append(np.abs(simpson_rule(f, a, b, n[i]) - true_value))



plt.loglog(n, leftpoint_errors, ':x', label='Leftpoint Rule')
plt.loglog(n, midpoint_errors, '-o', label='Midpoint Rule')
plt.loglog(n, rightpoint_errors, ':+', label='Rightpoint Rule', color = 'red')
plt.loglog(n, trapezoid_errors, '-s', label='Trapezoid Rule', color = 'purple')
plt.loglog(n, simpson_errors, '--', label='Simpson Rule', color = 'green')
plt.xlabel('Number of Subintervals')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid()
plt.title('Comparison of Integration Methods')
plt.show()
