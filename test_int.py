import pytest

from numpy import e, sin, cos
from math import log
from int import legendre, leftpoint_rule, rightpoint_rule, midpoint_rule, trapezoid_rule


@pytest.mark.parametrize("name_func",
                         [
                            leftpoint_rule,
                            rightpoint_rule,
                            midpoint_rule,
                            trapezoid_rule,
                            legendre
                         ]
                         )
@pytest.mark.parametrize("func, primitive",
                         [
                             (lambda x: x, lambda x: x**2/2),
                             (lambda x: x**2, lambda x: x**3/3),
                             (lambda x: x**3, lambda x: x**4/4),
                             (lambda x: 3, lambda x: 3*x),
                             (lambda x: 3**x, lambda x: (3**x)/log(3, e)),
                             (lambda x: e**x, lambda x: e**x),
                             (lambda x: sin(x), lambda x: -cos(x)),
                             (lambda x: cos(x), lambda x: sin(x))
                         ]
                         )
@pytest.mark.parametrize("a, b, eps",
                         [
                             (-1, 1, 1e-3),
                             (-1, 3, 1e-3),
                             (2, 5, 1e-3),
                             (-3, 2, 1e-3),
                             (-3, 5, 1e-3)
                         ]
                         )
def test_integral(name_func, primitive, func, a, b, eps):
    res = name_func(func, a, b, eps, primitive)[0]
    res_analytic = primitive(b) - primitive(a)
    assert res_analytic == pytest.approx(res, abs=1e-2)


@pytest.mark.parametrize("name_func",
                         [
                             leftpoint_rule,
                             rightpoint_rule,
                             midpoint_rule,

                         ]
                         )
@pytest.mark.parametrize("func, primitive",
                         [
                             (lambda x: x**2, lambda x: x**3/3),

                             (lambda x: sin(x), lambda x: -cos(x)),
                             (lambda x: cos(x), lambda x: sin(x)),
                             (lambda x: 1/x, lambda x: log(abs(x), e))
                         ]
                         )
@pytest.mark.parametrize("a, b, eps",
                         [
                             (-1, 2, 1e-3),
                             (-7, 8, 1e-3),
                             (-12, 16, 1e-3),

                         ]
                         )
def test_wrong(name_func, primitive, func, a, b, eps):
    res = name_func(func, a, b, eps, primitive)[0]
    res_analytic = primitive(b) - primitive(a)
    assert res_analytic != pytest.approx(res, abs=1e-4)
