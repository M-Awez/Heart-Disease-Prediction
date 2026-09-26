'''
In this file we are going to write model training
'''
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report , accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve
from log_code import setup_logging
logger = setup_logging('train_all_models')

def knn(X_train , X_test , y_train , y_test):
    try:
        global knn_obj
        knn_obj = KNeighborsClassifier(n_neighbors=5)
        knn_obj.fit(X_train , y_train)
        logger.info(f"Confusion Matrix : ")
        logger.info(f"{confusion_matrix(y_test , knn_obj.predict(X_test))}")
        logger.info(f"Accuracy score: {accuracy_score(y_test , knn_obj.predict(X_test))}")
        logger.info(f"Classification Report: {classification_report(y_test , knn_obj.predict(X_test))}")
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")


def naive_bayes(X_train, X_test, y_train, y_test):
    try:
        global nb_obj
        nb_obj = GaussianNB()
        nb_obj.fit(X_train, y_train)
        logger.info(f"Confusion Matrix : ")
        logger.info(f"{confusion_matrix(y_test, nb_obj.predict(X_test))}")
        logger.info(f"Accuracy score: {accuracy_score(y_test, nb_obj.predict(X_test))}")
        logger.info(f"Classification Report: {classification_report(y_test, nb_obj.predict(X_test))}")
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

def logistic_regression(X_train, X_test, y_train, y_test):
    try:
        global lr_obj
        lr_obj = LogisticRegression()
        lr_obj.fit(X_train, y_train)
        logger.info(f"Confusion Matrix : ")
        logger.info(f"{confusion_matrix(y_test, lr_obj.predict(X_test))}")
        logger.info(f"Accuracy score: {accuracy_score(y_test, lr_obj.predict(X_test))}")
        logger.info(f"Classification Report: {classification_report(y_test, lr_obj.predict(X_test))}")
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

def decision_tree(X_train, X_test, y_train, y_test):
    try:
        global dt_obj
        dt_obj = DecisionTreeClassifier(criterion='entropy')
        dt_obj.fit(X_train, y_train)
        logger.info(f"Confusion Matrix : ")
        logger.info(f"{confusion_matrix(y_test, dt_obj.predict(X_test))}")
        logger.info(f"Accuracy score: {accuracy_score(y_test, dt_obj.predict(X_test))}")
        logger.info(f"Classification Report: {classification_report(y_test, dt_obj.predict(X_test))}")
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

def random_forest(X_train, X_test, y_train, y_test):
    try:
        global rf_obj
        rf_obj = RandomForestClassifier(criterion='entropy', n_estimators=10)
        rf_obj.fit(X_train, y_train)
        logger.info(f"Confusion Matrix : ")
        logger.info(f"{confusion_matrix(y_test, rf_obj.predict(X_test))}")
        logger.info(f"Accuracy score: {accuracy_score(y_test, rf_obj.predict(X_test))}")
        logger.info(f"Classification Report: {classification_report(y_test, rf_obj.predict(X_test))}")
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

def adaboost(X_train, X_test, y_train, y_test):
    try:
        global ab_obj
        lr = LogisticRegression()
        ab_obj = AdaBoostClassifier(estimator = lr, n_estimators = 10)
        ab_obj.fit(X_train, y_train)
        logger.info(f"Confusion Matrix : ")
        logger.info(f"{confusion_matrix(y_test, ab_obj.predict(X_test))}")
        logger.info(f"Accuracy score: {accuracy_score(y_test, ab_obj.predict(X_test))}")
        logger.info(f"Classification Report: {classification_report(y_test, ab_obj.predict(X_test))}")
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

def gradient_boosting(X_train, X_test, y_train, y_test):
    try:
        global gb_obj
        gb_obj = GradientBoostingClassifier(n_estimators=10)
        gb_obj.fit(X_train, y_train)
        logger.info(f"Confusion Matrix : ")
        logger.info(f"{confusion_matrix(y_test, gb_obj.predict(X_test))}")
        logger.info(f"Accuracy score: {accuracy_score(y_test, gb_obj.predict(X_test))}")
        logger.info(f"Classification Report: {classification_report(y_test, gb_obj.predict(X_test))}")
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

def xtr_grad_boost(X_train, X_test, y_train, y_test):
    try:
        global xgb_obj
        xgb_obj = XGBClassifier()
        xgb_obj.fit(X_train, y_train)
        logger.info(f"Confusion Matrix : ")
        logger.info(f"{confusion_matrix(y_test, xgb_obj.predict(X_test))}")
        logger.info(f"Accuracy score: {accuracy_score(y_test, xgb_obj.predict(X_test))}")
        logger.info(f"Classification Report: {classification_report(y_test, xgb_obj.predict(X_test))}")
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

def auc_roc_curve_str(X_train , X_test , y_train , y_test):
    try:
        knn_fpr , knn_tpr , knn_thr = roc_curve(y_test , knn_obj.predict(X_test))
        nb_fpr, nb_tpr, nb_thr = roc_curve(y_test, nb_obj.predict(X_test))
        lr_fpr, lr_tpr, lr_thr = roc_curve(y_test, lr_obj.predict(X_test))
        dt_fpr, dt_tpr, dt_thr = roc_curve(y_test, dt_obj.predict(X_test))
        rf_fpr, rf_tpr, rf_thr = roc_curve(y_test, rf_obj.predict(X_test))
        ab_fpr, ab_tpr, ab_thr = roc_curve(y_test, ab_obj.predict(X_test))
        gb_fpr, gb_tpr, gb_thr = roc_curve(y_test, gb_obj.predict(X_test))
        xb_fpr, xb_tpr, xb_thr = roc_curve(y_test, xgb_obj.predict(X_test))

        plt.figure(figsize=(8,3))
        plt.title('AUC & ROC CURVES')
        plt.xlabel('FPR')
        plt.ylabel('TPR')

        plt.plot(knn_fpr , knn_tpr , label = 'KNN')
        plt.plot(nb_fpr , nb_tpr , label = 'NB')
        plt.plot(lr_fpr , lr_tpr , label = 'LR')
        plt.plot(dt_fpr , dt_tpr , label = 'DT')
        plt.plot(rf_fpr , rf_tpr , label = 'RF')
        plt.plot(ab_fpr , ab_tpr , label = 'AB')
        plt.plot(gb_fpr , gb_tpr , label = 'GB')
        plt.plot(xb_fpr , xb_tpr , label = 'XGB')
        plt.legend(loc = 0)
        plt.show()

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

def all_models(X_train , X_test , y_train , y_test):
    try:
        logger.info('=====================================KNN========================================')
        knn(X_train, X_test, y_train, y_test)
        logger.info('==================================NAIVE BAYES===================================')
        naive_bayes(X_train, X_test, y_train, y_test)
        logger.info('==============================LOGISTIC REGRESSION===============================')
        logistic_regression(X_train, X_test, y_train, y_test)
        logger.info('================================DECISION TREE===================================')
        decision_tree(X_train, X_test, y_train, y_test)
        logger.info('================================RANDOM FOREST===================================')
        random_forest(X_train, X_test, y_train, y_test)
        logger.info('==================================ADABOOST======================================')
        adaboost(X_train, X_test, y_train, y_test)
        logger.info('============================GRADIENT BOOSTING===================================')
        gradient_boosting(X_train, X_test, y_train, y_test)
        logger.info('=========================XTREME GRADIENT BOOSTING===============================')
        xtr_grad_boost(X_train, X_test, y_train, y_test)
        auc_roc_curve_str(X_train,X_test , y_train , y_test)

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")
