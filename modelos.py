from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn import svm

def modelos(model=None, classificador=None, solver=None, max_iter=4000, n_estimators=50):
    if model == 'lr':
        base_lr = LogisticRegression(solver=solver, max_iter=max_iter)
        if classificador == 'ovr':
            ovr = OneVsRestClassifier(base_lr)
            return ovr
        else:
            return base_lr
    elif model == 'rf':
        model = RandomForestClassifier(n_estimators=n_estimators)
        return model
    elif model == xgb:
        model = xgb.XGBClassifier()
        return model
    elif model == 'svm':
        model = svm.SVC(gamma='scale')
        return model

