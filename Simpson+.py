import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.sin(x)

def true_integral(x):
    return -np.cos(x)

a = 0
b = 1
n = np.array([2**i for i in range(25)])
true_value = true_integral(b) - true_integral(a)

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


plt.loglog(n, trapezoid_errors, '-s', label='Trapezoid Rule', color = 'purple')
plt.loglog(n, simpson_errors, '--', label='Simpson Rule', color = 'green')
plt.xlabel('Number of Subintervals')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid()
plt.title('Comparison of Integration Methods')
plt.show()


def chebyshev_integration(f, a, b, n):
    def chebyshev_nodes(n):
        return np.cos(np.pi * (2 * np.arange(1, n + 1) - 1) / (2 * n))

    x = chebyshev_nodes(n)
    x = 0.5 * (a + b) + 0.5 * (b - a) * x
    fx = f(x)
    c = np.zeros(n)
    c[-1] = np.sum(fx) / n
    for k in range(n - 1, 0, -1):
        c[k - 1] = 2 * np.sum(fx[::k]) / n
    return 0.5 * (b - a) * np.sum(c * np.cos(np.arange(0, n) * np.arccos(x)))