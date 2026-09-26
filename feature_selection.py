import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
import sys
from log_code import setup_logging
logger = setup_logging('feature_selection')

def cons_quasi(X_train , X_test, y_train, y_test):
    try:
        #   Constant Technique
        cons_obj = VarianceThreshold(threshold=0.0)
        cons_obj.fit(X_train)
        logger.info(f"cols to remove {X_train.columns[~cons_obj.get_support()]}")
        X_train = X_train.drop(['fbs_tr'] , axis = 1)
        X_test = X_test.drop(['fbs_tr'] , axis = 1)
        logger.info(X_train.columns)
        logger.info(X_train.shape)
        logger.info(X_test.columns)
        logger.info(X_test.shape)

        #   Quasi Constant technique

        qus_obj = VarianceThreshold(threshold=0.1)
        qus_obj.fit(X_train)
        logger.info(f"Columns to remove:{X_train.columns[~qus_obj.get_support()]}")
        X_train = X_train.drop(['trestbps_tr', 'chol_tr', 'exang_tr', 'ca_tr'], axis = 1)
        X_test = X_test.drop(['trestbps_tr', 'chol_tr', 'exang_tr', 'ca_tr'], axis = 1)
        logger.info(X_train.columns)
        logger.info(X_train.shape)
        logger.info(X_test.columns)
        logger.info(X_test.shape)

        # Correlation and Hypothesis Testing

        # v = []
        # for i in X_train.columns:
        #     a, p_val = pearsonr(X_train[i] , y_train)
        #     v.append(p_val)
        # v = np.array(v)
        # logger.info(v)
        # ab = pd.Series(v , index=X_train.columns)
        # c = 0
        # for i in ab:
        #     if i>0.05:
        #         s = X_train.columns[c]
        #     c+=1
        # logger.info(s)
        # plt.figure(figsize=(5, 3))
        # plt.title("Hypothesis Testing")
        #
        # plt.xlabel("column Names")
        # plt.ylabel("P_values for Each Independent column")
        #
        # plt.bar(X_train.columns, v)
        #
        # plt.show()
        X_train = X_train.drop('restecg_tr' , axis = 1)
        X_test = X_test.drop('restecg_tr' , axis = 1)
        logger.info(X_train.columns)
        logger.info(X_train.shape)
        logger.info(X_test.columns)
        logger.info(X_test.shape)
        return X_train, X_test

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")
