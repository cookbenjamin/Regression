from numpy import linalg, zeros, ones, hstack, asarray
import itertools
import polynomialModel as pm
import numpy as np


class polyPy(object):
    def __init__(self, split=.15, max_validation=5):
        self._lowest_test_error = None
        self._max_validation = max_validation
        self._validation_timer = None
        self._split = split
        self._best_degree = 0
        self._error = []
        pass

    def fit(self, dataset):
        # split data into training and test set
        train_X, train_Y, test_X, test_Y = self.split(dataset)
        self.trial(train_X, train_Y, test_X, test_Y)
        model = self.fit_model_with_degree(dataset[:,:-1], dataset[:,-1], self._best_degree)
        return model

    def split(self, dataset):
        np.random.shuffle(dataset)
        index = int(len(dataset)*self._split)
        train_X = dataset[:index][:,:-1]
        train_Y = dataset[:index][:,-1]
        test_X = dataset[index:][:, :-1]
        test_Y = dataset[index:][:, -1]
        return train_X, train_Y, test_X, test_Y

    def trial(self, train_X, train_Y, test_X, test_Y):
        self._validation_timer = 5
        degree = 1
        while self._validation_timer > 0:
            model = self.fit_model_with_degree(train_X, train_Y, degree)
            print(degree)
            print(model)
            error = model.error(test_X, test_Y)
            print("train error:", model.error(train_X, train_Y))
            print("test error:", error)
            print()
            self._error.append(error)
            self.check_error(error, degree)
            degree += 1

    def check_error(self, error, degree):
        if not self._lowest_test_error or error < self._lowest_test_error:
            self._lowest_test_error = error
            self._best_degree = degree
            self._validation_timer = self._max_validation
        else:
            self._validation_timer -= 1

    def fit_model_with_degree(self, X, Y, degree):
        Y = asarray(Y).squeeze()
        rows = Y.shape[0]
        X = asarray(X)
        num_covariates = X.shape[1]
        X = hstack((ones((X.shape[0], 1), dtype=X.dtype), X))

        generators = [self.basis_vector(num_covariates + 1, i)
                      for i in range(num_covariates + 1)]
        # All combinations of degrees
        powers = list(map(sum,
                          itertools.combinations_with_replacement(generators,
                                                                  degree)))
        # Raise data to specified degree pattern, stack in order
        A = hstack(asarray([self.as_tall((X ** p).prod(1)) for p in powers]))
        beta = linalg.lstsq(A, Y)
        print(beta[1])
        model = pm.PolyModel(powers, beta[0], degree, num_covariates)
        return model

    def basis_vector(self, n, i):
        x = zeros(n, dtype=int)
        x[i] = 1
        return x

    def as_tall(self, x):
        return x.reshape(x.shape + (1,))