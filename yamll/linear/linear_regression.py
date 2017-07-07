import numpy as np

class LinearRegression:
    def __init__(self, l2_penalty=0, l1_penalty=0):
        self._coef = np.array([])
        self._l2_penalty = l2_penalty
        self._l1_penalty = l1_penalty


    def _features(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack((ones, X))

    
    def _optimize_closed(self, X, y):
        features = self._features(X)
        self._coef = np.zeros((X.shape[1] + 1, y.shape[1]))
        self._coef = np.linalg.inv(features.T.dot(features)).dot(features.T).dot(y)
        

    def _optimize_gradient(self, X, y, tolerance):
        features = self._features(X)
        self._coef = np.zeros((X.shape[1] + 1, y.shape[1]))

        gradient = [tolerance, tolerance]
        gradient_norm = np.linalg.norm(gradient)
        step = 1.0

        while(gradient_norm > tolerance and step > tolerance):
            gradient = -2 * np.dot(features.T,
                                   (y - features.dot(self._coef))) \
                       + self._l2_penalty * np.vstack((np.zeros((1, y.shape[1])), self._coef[1:])) \
                       + self._l1_penalty * np.sign(np.vstack((np.zeros((1, y.shape[1])), self._coef[1:])))
            gradient_norm = np.linalg.norm(gradient)
            step = np.minimum(1.0 / gradient_norm, step * 0.9999)
            self._coef -= step * gradient

        if self._l1_penalty:
            self._coef[np.abs(self._coef) < tolerance*10] *= 0


    def fit(self, X, y, method='gd', tolerance=1e-6):
        X = np.array(X)
        y = np.array(y)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        elif len(y.shape) > 2:
            raise ValueError('y must be a 1- or 2-dimensional array')

        if len(X.shape) != 2:
            raise ValueError('X must be a 2-dimensional array')
        elif X.shape[0] != y.shape[0]:
            raise ValueError('first dimensions of X and y must be the same')
           
        if method == 'gd':
            self._optimize_gradient(X, y, tolerance)
        elif method == 'closed':
            self._optimize_closed(X, y)


    def predict(self, X):
        X = np.array(X)

        if len(X.shape) != 2:
            raise ValueError('X must be a 2-dimensional array')
        elif X.shape[1] != self._coef.shape[0] - 1:
            raise ValueError('incorrect number of features in X: \
{:d}, must be {:d}'.format(X.shape[1], self._coef.shape[0] - 1))

        predictions = np.dot(self._features(X), self._coef)
        return predictions


    def r_squared(self, X, y):
        """ computes R-squared statistic based on (X => yhat) and y"""
        y = np.array(y)
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        yhat = self.predict(X)
        residuals = yhat - y
        RSS = np.diag(np.dot(residuals.T, residuals))
        TSS = np.array([np.var(y[:, i]) * y.shape[0] for i in range(y.shape[1])])
        return 1 - RSS / TSS


    def set_params(self, params):
        self._coef = np.array(params)
