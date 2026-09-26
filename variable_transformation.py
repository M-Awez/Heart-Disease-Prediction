'''
In this file we are going to perform Variable transformation and Trimming methods
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import boxplot
import sys
from scipy.stats import yeojohnson
from log_code import setup_logging
logger = setup_logging('variable_transformation')

def normal_trimming(X_train , X_test):
    try:
        #Yeojohnson for Variable Transformation
        for i in X_train.columns:
            X_train[i+'_yeo'], lam = yeojohnson(X_train[i])
            X_test[i+'_yeo'], lam = yeojohnson(X_test[i])
            X_train = X_train.drop([i], axis = 1)
            X_test = X_test.drop([i] , axis = 1)

            # Trimming for Outlier handling
            iqr = X_train[i+'_yeo'].quantile(0.75) - X_train[i+'_yeo'].quantile(0.25)
            ul = X_train[i+'_yeo'].quantile(0.75) + (1.5 * iqr)
            ll = X_train[i+'_yeo'].quantile(0.25) - (1.5 * iqr)
            X_train[i+'_tr'] = np.where(X_train[i+'_yeo']<ll , ll,
                                        np.where(X_train[i+'_yeo']>ul , ul , X_train[i+'_yeo']))
            X_test[i + '_tr'] = np.where(X_test[i + '_yeo'] < ll, ll,
                                          np.where(X_test[i + '_yeo'] > ul, ul, X_test[i + '_yeo']))
            X_train = X_train.drop([i+'_yeo'], axis = 1)
            X_test = X_test.drop([i+'_yeo'],axis = 1)

        logger.info(X_train.shape)
        logger.info(X_test.shape)
        logger.info(X_train.columns)
        logger.info(X_test.columns)
        logger.info(X_train.head())
        return X_train, X_test

    except Exception as e:
        er_type,er_msg,er_line=sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")