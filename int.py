
import scipy.special


def leftpoint_rule(func, a, b, eps, primitive):

    """
    Certain integration, left rectangle method
    :param func: integrated function
    :param a: lower border of integral
    :param b: upper border of integral
    :param eps: relative error
    :param primitive: antiderivative
    :return: the value of integral and quantity of nodes
    """
    n = 1
    integral = func((a + b) / 2) * (b - a)

    while abs(integral - (primitive(b)-primitive(a))) >= eps:
        integral = 0
        n *= 2
        dx = (b - a) / n
        for i in range(1, n + 1):
            integral += func(a + (i - 1) * dx) * dx

    return integral, n



def rightpoint_rule(func, a, b, eps, primitive):

    """
    Certain integration, right rectangle method
    :param func: integrated function
    :param a: lower border of integral
    :param b: upper border of integral
    :param eps: relative error
    :param primitive: antiderivative
    :return: the value of integral and quantity of nodes
    """


    n = 1
    integral = func((a + b) / 2) * (b - a)

    while abs(integral - (primitive(b)-primitive(a))) >= eps:
        integral = 0
        n *= 2
        dx = (b - a) / n
        for i in range(1, n + 1):
            integral += func(a + i * dx) * dx

    return integral, n



def midpoint_rule(func, a, b, eps, primitive):

    """
    Certain integration, midpoint rule
    :param func: integrated function
    :param a: lower border of integral
    :param b: upper border of integral
    :param eps: relative error
    :param primitive: antiderivative
    :return: the value of integral and quantity of nodes
    """

    n = 1
    integral = func((a + b) / 2) * (b - a)

    while abs(integral - (primitive(b)-primitive(a))) >= eps:
        integral = 0
        n *= 2
        dx = (b - a) / n
        for i in range(1, n + 1):
            integral += func(a + (i - 0.5) * dx) * dx

    return integral, n





def trapezoid_rule(func, a, b, eps):

    """
    Certain integration by trapezoid rule
    :param func: integrated function
    :param a: lower border of integral
    :param b: upper border of integral
    :param eps: relative error
    :return: the value of integral and quantity of nodes
    """

    n = 1
    dx = (b - a) / n
    integral = 0.5 * dx * (func(a) + func(b))
    err_est = 1

    while err_est > abs(eps * integral):
        old_integral = integral
        integral = 0.5 * (integral + midpoint_rule(func, a, b, n)[0])
        n *= 2
        err_est = abs(integral - old_integral)
    return integral, n


def legendre(func, a, b, eps, primitive):

    """
    Certain integration by Gauss'es
    :param func: integrated function
    :param a: lower border of integral
    :param b: upper border of integral
    :param eps: relative error
    :param primitive: antiderivative
    :return: the value of integral and quantity of nodes
    """

    n = 0
    integral = 0
    k = lambda x: 0.5 * (b + a) + 0.5 * (b - a) * x

    while abs(integral - (primitive(b) - primitive(a))) >= eps:
        n += 1
        nodes, weights = scipy.special.roots_legendre(n)
        integral = (0.5 * (b - a) * func(k(nodes)) * weights).sum()

    return integral, n