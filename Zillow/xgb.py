import xgboost as xgb
from sklearn.metrics import r2_score

class XGBoostReg():
    def __init__(self, num_boost_round=10, **kwargs):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = kwargs
        self.params.update({'objective': 'reg:linear'})

    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        dtrain = xgb.DMatrix(X, label=y)
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)

    def score(self, X, y):
        Y = self.predict(X)
        return r2_score(y, Y)

    def get_params(self, deep=True):
        return self.params

    def get_score(self):
        return self.clf.get_fscore()

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self
