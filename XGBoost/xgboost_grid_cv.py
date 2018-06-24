import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
import matplotlib.pylab as plt
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV


# 加载数据
def load_data(train_data_path,
              test_data_path,
              predict_cols,
              target_col,
              dtypes,
              index_col=False):

    train_data = pd.read_csv(train_data_path, usecols=predict_cols + target_col, dtype=dtypes, index_col=index_col)
    train_label = train_data[target_col]

    test_data = pd.read_csv(test_data_path, usecols=predict_cols, dtype=dtypes, index_col=index_col)

    return train_data, train_label, test_data


# 生成xgboost数据
def xgb_data(train_data, train_label, test_data):

    xgb_train_data = xgb.DMatrix(train_data, label=train_label)
    xgb_test_data = xgb.DMatrix(test_data)

    del train_data, train_label, test_data
    gc.collect()
    return xgb_train_data, xgb_test_data


# 初始化xgboost模型
def init_model():

    xgb = XGBClassifier(booster='gbtree',
                        nthread=4,
                        learning_rate=0.1,
                        min_child_weight=1,
                        gamma=0,
                        n_estimators=1000,
                        max_depth=5,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        objective='binary:logistic',
                        seed=27)

    return xgb


# 交叉验证
def xgb_cv(xgb_model, train_data, train_label, cv_folds=5, early_stopping_rounds=50):

    xgb_param = xgb_model.get_xgb_params()
    xgb_train_data = xgb.DMatrix(train_data, label=train_label)

    cv_result = xgb.cv(xgb_param,
                       xgb_train_data,
                       num_boost_round=xgb_model.get_params()['n_estimators'],
                       nfold=cv_folds,
                       metrics='auc',
                       early_stopping_rounds=early_stopping_rounds)
    print(cv_result)

    return cv_result


# 训练
def xgb_train(xgb_model, train_data, train_label):

    # 训练模型
    xgb_model.fit(train_data, train_label, eval_metric='auc')

    # 预测
    pre_label = xgb_model.predict(train_data)
    pre_prob = xgb_model.predict_proba(train_data)[:, 1]

    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(train_label, pre_label))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train_label, pre_prob))

    return



# 确定n_estimators
def tun_nestimators(xgb_model, train_data, train_label, cv_folds=5, early_stopping_rounds=50):

    cv_result = xgb_cv(xgb_model, train_data, train_label, cv_folds=cv_folds, early_stopping_rounds=early_stopping_rounds)

    print("the n_estimators is %s" % (cv_result.shape[0]))

    xgb_model.set_params(n_estimators=cv_result.shape[0])

    return xgb_model


# 确定max_depth 和 min_child_weight 参数
def tun_depth_child_weight(xgb_model, train_data, train_label, param_test):

    if param_test is None:
        param_test = {
            'max_depth': range(3, 10, 2),
            'min_child_weight': range(1, 6, 2)
        }

    gsearch = GridSearchCV(estimator=xgb_model, param_grid=param_test, scoring='roc_auc', iid=False, cv=5)
    gsearch.fit(train_data, train_label)

    print(gsearch.grid_scores_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)

    return


# 确定 gamma 参数
def tun_gamma(xgb_model, train_data, train_label, param_test):

    if param_test is None:
        param_test = {
            'gamma': [i / 10.0 for i in range(0, 5)]
        }

    gsearch = GridSearchCV(estimator=xgb_model, param_grid=param_test, scoring='roc_auc', iid=False, cv=5)
    gsearch.fit(train_data, train_label)

    print(gsearch.grid_scores_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)

    return

# 确定 subsample 和 colsample_bytree 参数
def tun_subs_cols(xgb_model, train_data, train_label, param_test):

    if param_test is None:
        param_test = {
            'subsample': [i / 10.0 for i in range(6, 10)],
            'colsample_bytree': [i / 10.0 for i in range(6, 10)]
        }

    gsearch = GridSearchCV(estimator=xgb_model, param_grid=param_test, scoring='roc_auc', iid=False, cv=5)
    gsearch.fit(train_data, train_label)

    print(gsearch.grid_scores_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)

    return

# 确定 reg_alpha 参数
def tun_reg_alpha(xgb_model, train_data, train_label, param_test):

    if param_test is None:
        param_test = {
            'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
        }

    gsearch = GridSearchCV(estimator=xgb_model, param_grid=param_test, scoring='roc_auc', iid=False, cv=5)
    gsearch.fit(train_data, train_label)

    print(gsearch.grid_scores_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)

    return

# 确定 reg_lambda 参数
def tun_reg_lambda(xgb_model, train_data, train_label, param_test):

    if param_test is None:
        param_test = {
            'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100]
        }

    gsearch = GridSearchCV(estimator=xgb_model, param_grid=param_test, scoring='roc_auc', iid=False, cv=5)
    gsearch.fit(train_data, train_label)

    print(gsearch.grid_scores_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)

    return

# 特征选择
def feature_select(xgb_model, train_data, train_label):

    # 训练模型
    xgb_model.fit(train_data, train_label, eval_metric='auc')

    feat_imp = pd.Series(xgb_model.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar',title='Feature importance')
    plt.ylabel('Feature Importance Score')
    print(feat_imp)
    return












