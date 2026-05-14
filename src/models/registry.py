from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegressionCV

MODELS={"rf": RandomForestClassifier(random_state=73),
        "xgb": XGBClassifier(random_state=73),
        "lr_cv": LogisticRegressionCV(random_state=73)}