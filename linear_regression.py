import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')


class LinearRegression:
    def __init__(self, l2_penalty=0):
        self._coef = np.array([])
        self._l2_penalty = l2_penalty


    def _features(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack((ones, X))


    def _optimize(self, X, y, tolerance):
        features = self._features(X)
        self._coef = np.zeros(X.shape[1] + 1)

        gradient = [tolerance, tolerance]
        gradient_norm = np.linalg.norm(gradient)
        step = 1

        while(gradient_norm > tolerance):
            gradient = -2 * np.dot(features.T,
                                   (y - features.dot(self._coef))) \
                       + 2 * self._l2_penalty * np.concatenate(([0], self._coef[1:]))
            gradient_norm = np.linalg.norm(gradient)
            step = np.minimum(1.0 / gradient_norm, step)
            self._coef -= step * gradient


    def fit(self, X, y, tolerance=1e-6):
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


if __name__ == "__main__":
    samples = 100
    lsr = LinearRegression()
    fig = plt.figure(figsize=(10, 5))

    #########################################
    ## 2d, one feature
    #########################################

    ax1 = fig.add_subplot(121)
    X = np.random.rand(samples, 1)
    y = (3 - 2*X + 0.5*np.random.rand(samples, 1)).reshape(samples)

    lsr.fit(X, y)
    slope, intercept = lsr._coef
    ax1.scatter(X, y, c='g')
    ax1.plot([0, 1], [slope, slope + intercept])
    plt.title("R squared: {:f}".format(lsr.r_squared(X, y)))

    #########################################
    ## 3d, polynomial features
    #########################################
    func = lambda x, x2, y, y2: 5 + x + 5 * x2 + 2 * y2 + 2 * np.random.rand()
    x1 = np.random.rand(samples)
    y1 = np.random.rand(samples)
    x2 = x1**2
    y2 = y1**2
    X = np.array([x1, x2, y1, y2]).T
    y = np.array([func(*row) for row in X])

    lsr.fit(X, y)

    ax2 = fig.add_subplot(122, projection='3d')

    xx = np.linspace(0, 1, samples)
    yy = np.linspace(0, 1, samples)
    xx, yy = np.meshgrid(xx, yy)

    zz = np.ones((samples, samples)) * lsr._coef[0] + \
         xx * lsr._coef[1] + xx**2 * lsr._coef[2] + \
         yy * lsr._coef[3] + yy**2 * lsr._coef[4]

    ax2.scatter(x1, y1, y, c='g')

    ax2.plot_surface(xx, yy, zz, antialiased=False, cmap='gray')

    ########################################

    plt.title("R squared: {:f}".format(lsr.r_squared(X, y)))
    plt.show()
