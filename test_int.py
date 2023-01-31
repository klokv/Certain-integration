import numpy as np
import pytest
from main import leftpoint_rule, rightpoint_rule, midpoint_rule, trapezoid_rule, simpson_rule


@pytest.mark.parametrize("method",
                         [
                            leftpoint_rule,
                            rightpoint_rule,
                            midpoint_rule,
                            trapezoid_rule,
                            simpson_rule
                         ]
                         )
@pytest.mark.parametrize("func, true_integral",
                         [
                             (lambda x: x, lambda x: x**2/2),
                             (lambda x: x**2, lambda x: x**3/3),
                             (lambda x: x**3, lambda x: x**4/4),
                             (lambda x: 3, lambda x: 3*x),
                             (lambda x: 3**x, lambda x: (3**x)/np.log(3, np.e)),
                             (lambda x: np.e**x, lambda x: np.e**x),
                             (lambda x: np.sin(x), lambda x: -np.cos(x)),
                             (lambda x: np.cos(x), lambda x: np.sin(x))
                         ]
                         )
@pytest.mark.parametrize("a, b, eps",
                         [
                             (-1, 1, 1e-6),
                             (-1, 3, 1e-6),
                             (2, 5, 1e-6),
                             (-3, 2, 1e-6),
                             (-3, 5, 1e-6)
                         ]
                         )
def test_integral(method, true_integral, func, a, b, eps):
    expected_value = true_integral(b) - true_integral(a)
    computed_value, computed_n = method(func, a, b, eps)
    assert abs(expected_value - computed_value) < 1e-6
    assert computed_n <= 10**6

