import numpy as np
from sympy import symbols, Add, Mul, S, diff


class PolynomialModel(object):
    """
    A class that represents a Polynomial Regression Model
    """
    def __init__(self, powers, coeffs, degree, num_covariates):
        """
        Creates a new Polynomial Regression Model

        :param powers: the powers of the model
        :param coeffs: the coefficients of the model
        :param degree: the degree of the model
        :param num_covariates: the number of covariates in the model
        """
        self._powers = powers
        self._coefficients = [round(coeff, 2) for coeff in coeffs]
        self._degree = degree
        self._num_covariates = num_covariates
        self._expression = self._sympy_func()
        self._symbols = self._expression.free_symbols
        self._fig = None
        self._gradient = self._generate_gradient()

    def predict(self, X):
        X = self._add_bias(X)
        return np.array([[np.sum([coeff * np.sum([x ** p
                                                  for (x, p) in
                                                  zip(small_x, power)])
                                  for power, coeff in
                                  zip(self._powers, self._coefficients)])]
                         for small_x in X])

    def minimize(self, pos, **kwargs):
        return self._optimize(pos, minimize=True, **kwargs)

    def maximize(self, pos, **kwargs):
        return self._optimize(pos, minimize=False, **kwargs)

    def _sympy_func(self):
        xs = (S.One,) + symbols('x0:%d' % self._num_covariates)
        return Add(*[coeff * Mul(*[x ** deg
                                   for x, deg in zip(xs, power)])
                     for power, coeff in zip(self._powers, self._coefficients)])

    def _generate_gradient(self):
        gradient = []
        for symbol in self._symbols:
            gradient.append(diff(self._expression, symbol))
        return gradient

    def _add_bias(self, X):
        bias = np.zeros((len(X), len(X[0]) + 1))
        bias[:, 1:] = X
        bias[:, 0] = 1
        return bias

    def _calculate_error(self, X, Y):
        return np.sum(np.abs(Y - self.predict(X)))

    def _get_gradient_at_point(self, point):
        gradient = []
        for partial_derivative in self._gradient:
            gradient.append(partial_derivative.subs(zip(self._symbols, point)))
        return np.array(gradient)

    def _optimize(self, pos, minimize=True, rang=None, constants=None,
                  learning_rate=.01, accuracy=6):
        if not constants:
            constants = set()
        pos = np.array(pos)
        last_pos = pos*2
        if rang:
            min_range, max_range = rang
        epsilon = 1*10**-(accuracy + 1)
        if minimize:
            minmax = -1
        else:
            minmax = 1
        while True:
            # if reach end of range, keep coord constant
            if rang:
                for i, coord, min_coord, max_coord in zip(range(len(pos)), pos, min_range, max_range):
                    if coord <= min_coord:
                        pos[i] = min_coord
                        constants.add(i)
                        print(constants)
                    elif coord >= max_coord:
                        pos[i] = max_coord
                        constants.add(i)
            if (self._get_gradient_at_point(pos) < epsilon).all() \
                    and (self._get_gradient_at_point(pos) > -epsilon).all():
                return [round(coord, accuracy) for coord in pos]
            elif not (pos - last_pos).any():
                return [round(coord, accuracy) for coord in pos]
            else:
                if constants:
                    pos_constants = [pos[i] for i in constants]
                last_pos = pos
                pos = pos + minmax * learning_rate * self._get_gradient_at_point(pos)
                if constants:
                    for i, constant in zip(constants, pos_constants):
                        pos[i] = constant
                print("pos", pos)
                print("operation", (pos - last_pos)**2)

    def __repr__(self):
        return str(self._expression)

    def __str__(self):
        return self.__repr__()