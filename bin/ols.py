# ==========================================================
# ordinal linear regression
# approach here TODO
# author:
# ==========================================================

import linalg

class ols(object):
    def __init__(self, fit_intercept=True):
        self._param = None
        self._intercept = None
        self.fit_intercept = fit_intercept
        self._n_feature = None

    def _normlizeInput(self, X, y):
        """ normlize X and y, remove mean and devide by
        this function is used in fit
        """
    def fit(self, X, y, sample_weight=None):
        """ fitting linear regression between X and y
        args:
            X: matrix [n_sample, n_feature],
                Training data
            y: vector [n_sample],
                Training data
            sample_weight, vector [n_shape], optional,
                individual weight for each sample
        """
        # check X, y and sample_weight shape
        # normalize X, y
        # compute X*X
        # compute X*y
        # compute inv(X*X)*(X*y)
        # set value into self._param and self._intercept

    def get_param(self):
        """ get param
        return:
            param: vector [n_feature]
        """
        # check if self._param is none
        # (the function is get called before fit)
        return self._param

    def get_intercept(self):
        """ get intercept
        return:
            intercept: float
        """
        # check if self._intercept is none
        # (the function is get called before fit)
        return self._intercept

    def predict(self, X):
        """ fit using this model
        y = X * param + intercept

        args:
            X: matrix [n_sample, n_features]
                test samples
        return
            y: vector [n_sample], prediction values
        """
        # check matrix shape
        # compute prediction
        # return prediction
    def score(self, X, y, sampe_weight=None):
        """ compute the coefficient of determination R^s of the prediciton
        args:
            X: matrix [n_sample, n_feature], test samples
            y: vector [n_sample] test samples
            sampe_weight: vector [n_sample] test sample weights
        return:
            score: float
        """
        # check shape of X and y
        # predict yhat
        # compute R^2 score
        # return R^2 score
