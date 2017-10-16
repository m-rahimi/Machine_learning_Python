import lightgbm as lgb
from sklearn.metrics import r2_score

class LgbReg(object):
    def __init__(self, num_iterations=10, **kwargs):
        self.clf = None
        self.num_iterations = num_iterations
        self.params = kwargs
        self.params.update({'objective': 'regression_l1'})

    def fit(self, X, y, num_iterations=None):
        num_iterations = num_iterations or self.num_iterations
        dtrain = lgb.Dataset(X, label= y)
        self.clf = lgb.train(self.params, dtrain, num_iterations)

    def predict(self, X):
#        dtest = lgb.Dataset(X)
        return self.clf.predict(X)

    def score(self, X, y):
        Y = self.predict(X)
        return r2_score(y, Y)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'num_iterations' in params:
            self.num_iterations = params.pop('num_iterations')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self
