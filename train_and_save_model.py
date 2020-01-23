import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier    
from sklearn.externals import joblib 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline 

    
def create_and_save_model(X, y):

    clf1 = XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,
    colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,scale_pos_weight=0.2,seed=27,reg_alpha=0.4,
    reg_lambda=1,early_stopping_rounds=50, show_progress=True)

    clf2 = AdaBoostClassifier(n_estimators=150)

    # initialize the base classifier 
    base_cls = DecisionTreeClassifier() 
    
    # no. of base classifier 
    num_trees = 200
    
    # bagging classifier 
    clf3 = BaggingClassifier(base_estimator = base_cls, 
                            n_estimators = num_trees, 
                            random_state = 8, n_jobs=-1) 

    clf4 = RandomForestClassifier(bootstrap=True, class_weight={0:2.5,1:1}, criterion='entropy',
                max_depth=60, max_features="auto", max_leaf_nodes=50,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=5, min_samples_split=6,
                min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=-1,
                oob_score=True, random_state=10, verbose=1, warm_start=False)

    params = {'n_estimators': 200, 'max_depth': 20, 'subsample': 0.6,
            'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3, 'loss': 'exponential', 'max_features':'auto', 'verbose':1 } #'ccp_alpha': 0.04
    clf5 = GradientBoostingClassifier(**params)

    estimators=[('xgb', clf1), ('abc', clf2), ('bc', clf3), ('rf', clf4), ('gbc', clf5)]

    stack_estimator = XGBClassifier(learning_rate =0.1,n_estimators=300,max_depth=5,min_child_weight=1,
    gamma=0,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',nthread=40,scale_pos_weight=1,
    seed=27,reg_alpha=0,reg_lambda=1,
    early_stopping_rounds=50, show_progress=True)

    model = StackingClassifier(estimators=estimators, final_estimator = stack_estimator, n_jobs=-1, cv = 5, verbose = 1)

    model.fit(X, y)

    file_name = 'model_final.pkl'
    joblib.dump(model, file_name) 

    return file_name
    
    