
import scipy.special
import numpy as np
from scipy.integrate import quad

def leftpoint_rule(func, a, b, eps):

    """
    Certain integration, left rectangle method
    :param func: integrated function
    :param a: lower border of integral
    :param b: upper border of integral
    :param eps: relative error
    :return: the value of integral and quantity of nodes
    """
    n = 1
    prev_integration = func((a + b) / 2) * (b - a)
    integral = 0
    while abs(integral - prev_integration) > eps:
        integral = 0
        n *= 2
        dx = (b - a) / n
        for i in range(1, n + 1):
            integral += func(a + (i - 1) * dx) * dx

    return integral, n



def rightpoint_rule(func, a, b, eps):

    """
    Certain integration, right rectangle method
    :param func: integrated function
    :param a: lower border of integral
    :param b: upper border of integral
    :param eps: relative error
    :return: the value of integral and quantity of nodes
    """


    n = 1
    prev_integration = func((a + b) / 2) * (b - a)
    integral = 0
    while abs(integral - prev_integration) > eps:
        integral = 0
        n *= 2
        dx = (b - a) / n
        for i in range(1, n + 1):
            integral += func(a + i * dx) * dx

    return integral, n



def midpoint_rule(func, a, b, eps):

    """
    Certain integration, midpoint rule
    :param func: integrated function
    :param a: lower border of integral
    :param b: upper border of integral
    :param eps: relative error
    :return: the value of integral and quantity of nodes
    """

    n = 1
    prev_integration = func((a + b) / 2) * (b - a)
    integral = 0
    while abs(integral - prev_integration) > eps:
        integral = 0
        n *= 2
        dx = (b - a) / n
        for i in range(1, n + 1):
            integral += func(a + (i - 0.5) * dx) * dx

    return integral, n


def trapezoid_rule(func, a, b, eps):
    """
        Certain integration, trapezoid method
        :param func: integrated function
        :param a: lower border of integral
        :param b: upper border of integral
        :param eps: relative error
        :return: the value of integral and quantity of nodes
        """
    n = 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    prev_integration = h * (func(a) + 2 * np.sum(func(x[1:-1])) + func(b)) / 2
    integration = 0

    while abs(integration - prev_integration) > eps:
        n *= 2
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        prev_integration = integration
        integration = h * (func(a) + 2 * np.sum(func(x[1:-1])) + func(b)) / 2

    return integration, n


def simpson_rule(func, a, b, eps):
    """
    Certain integration, Simpson's formula
    :param func: integrated function
    :param a: lower border of integral
    :param b: upper border of integral
    :param eps: relative error
    :return: the value of integral and quantity of nodes
    """

    n = 2  # start with two subintervals
    while True:
        h = (b-a)/n
        x = np.linspace(a, b, n+1)
        approx = h/3 * (func(a) + 2*np.sum(func(x[1:-1])) + func(b))
        error = np.abs(approx - quad(func, a, b, limit=n*10)[0])  # error estimate
        if error <= eps:
            return approx, n
        n *= 2  # double the number of subintervals

def romberg(f, a, b, eps):
    """
    Certain integration, Romberg's method
    :param func: integrated function
    :param a: lower border of integral
    :param b: upper border of integral
    :param eps: relative error
    :return: the value of integral and quantity of nodes
    """
    R = np.zeros((20, 20))
    h = b - a
    R[0, 0] = h * (f(a) + f(b)) / 2
    for i in range(1, 20):
        h /= 2
        sum = 0
        for j in range(1, 2**(i-1)+1):
            sum += f(a + (2*j-1) * h)
        R[i, 0] = R[i-1, 0] / 2 + h * sum
        for j in range(1, i+1):
            R[i, j] = (4**j * R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
        if np.abs(R[i, i-1] - R[i-1, i-1]) < eps:
            break
    return R[i, i]

def legendre(func, a, b, eps, true_integral):

    """
    Certain integration by Gauss'es
    :param func: integrated function
    :param a: lower border of integral
    :param b: upper border of integral
    :param eps: relative error
    :param true_integral: antiderivative
    :return: the value of integral and quantity of nodes
    """

    n = 0
    integral = 0
    k = lambda x: 0.5 * (b + a) + 0.5 * (b - a) * x

    while abs(integral - true_integral(b) - true_integral(a)) >= eps:
        n += 1
        nodes, weights = scipy.special.roots_legendre(n)
        integral = (0.5 * (b - a) * func(k(nodes)) * weights).sum()

    return integral, n


def chebyshev_integration(func, a, b, eps):
    """
        Certain integration using Chebyshev's method
        :param func: integrated function
        :param a: lower border of integral
        :param b: upper border of integral
        :param eps: relative error
        :return: the value of integral and quantity of nodes
        """
    n = 1
    while True:
        x = np.cos(np.pi * (2 * np.arange(1, n+1) - 1) / (2 * n))
        x = 0.5 * (b - a) * x + 0.5 * (b + a)
        integral = np.sum(func(x)) * 0.5 * (b - a) / n
        if abs(integral - scipy.integrate.fixed_quad(func, a, b, n=n)[0]) < eps:
            break
        n += 1
    return integral, n
