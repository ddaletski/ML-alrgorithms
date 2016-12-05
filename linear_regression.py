import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class LinearRegression:
    def __init__(self, l2_penalty=0):
        self._coef = np.array([])
        self._l2_penalty = l2_penalty


    def _features(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack((ones, X))


    def _optimize(self, X, y, tolerance):
        self._coef = np.zeros(X.shape[1] + 1)
        gradient = [tolerance, tolerance]
        step = 1

        while(np.linalg.norm(gradient) > tolerance):
            gradient = -2 * np.dot(self._features(X).T,
                                   (y - self._features(X).dot(self._coef))) \
                       + 2 * self._l2_penalty * np.concatenate(([0], self._coef[1:]))
            step = np.minimum(1.0 / np.linalg.norm(gradient), step)
            self._coef -= step * gradient


    def fit(self, X, y, tolerance=1e-3):
        X = np.array(X)
        y = np.array(y)

        if len(X.shape) != 2:
            raise ValueError('X must be a matrix')
        elif len(y.shape) != 1:
            raise ValueError('y must be a 1-dimensional array')
        elif X.shape[0] != y.shape[0]:
            raise ValueError('first dimensions of X and y must be the same')

        self._optimize(X, y, tolerance)


    def predict(self, X):
        X = np.array(X)

        if len(X.shape) != 2:
            raise ValueError('X must be a matrix')
        elif X.shape[1] != self._coef.shape[0] - 1:
            raise ValueError('incorrect number of features in X: \
{:d}, must be {:d}'.format(X.shape[1], self._coef.shape[0] - 1))

        predictions = np.dot(self._features(X), self._coef)
        return predictions


    def r_squared(self, X, y):
        """ computes R-squared statistic based on (X => yhat) and y"""
        yhat = self.predict(X)
        residuals = yhat - y
        RSS = np.dot(residuals, residuals)
        TSS = np.var(y) * y.shape[0]
        return 1 - RSS / TSS


def generateData(samples, features, yfunc=lambda x: np.sum(x)):
    X = np.random.rand(samples, features)
    y = np.array([yfunc(row) for row in X])
    return X, y


X, y = generateData(300, 1, lambda x : np.sum(x) +
                    5 * np.sum(x)**2 +
                    0.5 * np.random.randn())

lsr = LinearRegression(l2_penalty=0.0)
lsr.fit(X, y)
predictions = lsr.predict(X)

lineX = np.array([X.min(), X.max()])
lineY = lsr._coef[1] * lineX + lsr._coef[0]
plt.plot(X, y, '.b')
plt.plot(lineX, lineY, '-r', label='regression line')
plt.legend(loc=1)
plt.title("R^2: {:f}".format(lsr.r_squared(X, y)))
plt.show()

