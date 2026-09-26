'''
It is the main file of the project where we are going to call all the functions
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as p
import sys
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from log_code import setup_logging
logger = setup_logging('main')
from variable_transformation import normal_trimming
from feature_selection import cons_quasi
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from train_all_models import all_models
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

class HEART_DISEASE_PREDICTION:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(path)
            # logger.info(self.df.head())
            self.X = self.df.iloc[: , :-1]
            self.y = self.df.iloc[: , -1]
            # logger.info(self.y.shape)
            # logger.info(self.X.shape)
            # logger.info(self.df.isnull().sum())
            self.X_train, self.X_test, self.y_train , self.y_test = train_test_split(self.X , self.y, test_size=0.2 , random_state=42)
            logger.info(self.X_train.shape)
            logger.info(self.X_test.shape)
            logger.info(self.y_train.shape)
            logger.info(self.y_test.shape)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

    def outliers_check(self):
        try:
            self.X_train, self.X_test = normal_trimming(self.X_train , self.X_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

    def feature_Select(self):
        try:
            self.X_train, self.X_test = cons_quasi(self.X_train, self.X_test , self.y_train , self.y_test)
            logger.info(self.X_train.shape)
            logger.info(self.y_train.shape)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

    def data_balance(self):
        try:

            # Data balancing

            logger.info(f"num of 1s : {sum(self.y_train == 1)}")
            logger.info(f"num of 0s : {sum(self.y_train == 0)}")
            bal_obj = SMOTE(random_state=42)
            self.X_train , self.y_train = bal_obj.fit_resample(self.X_train , self.y_train)
            logger.info(f"num of 1s : {sum(self.y_train == 1)}")
            logger.info(f"num of 0s : {sum(self.y_train == 0)}")

            #Scaling Down

            self.sc_obj = StandardScaler()
            self.sc_obj.fit(self.X_train)
            self.X_train_scaled = self.sc_obj.transform(self.X_train)
            self.X_test_scaled = self.sc_obj.transform(self.X_test)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

    def algorithms(self):
        try:
            all_models(self.X_train_scaled , self.X_test_scaled , self.y_train , self.y_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

    def best_model_train(self):
        try:
            nb_obj = GaussianNB(var_smoothing=np.float64(1e-10))
            nb_obj.fit(self.X_train_scaled , self.y_train)
            logger.info(f"accuracy score :{accuracy_score(self.y_test , nb_obj.predict(self.X_test_scaled))}")
            logger.info(f"Confusion Matrix: {confusion_matrix(self.y_test , nb_obj.predict(self.X_test_scaled))}")
            logger.info(f"Classification Report: {classification_report(self.y_test , nb_obj.predict(self.X_test_scaled))}")

            # Hyperparameter Tuning

            # params = {
            #     "var_smoothing" : np.logspace(-10,-1,10)
            # }
            # grid_search = GridSearchCV(
            #     estimator=nb_obj,
            #     param_grid=params,
            #     cv=5,
            #     scoring='accuracy',
            #     n_jobs=-1
            # )
            # grid_search.fit(self.X_train_scaled, self.y_train)
            # logger.info(f"Best Parameters: {grid_search.best_params_}")
            # logger.info(f"Best CV Accuracy: {grid_search.best_score_}")

            # For testing new data

            abc = np.array([[60, 1,3,172,0.6 , 0 , 2]])
            self.sc_obj.transform(abc)
            logger.info(nb_obj.predict(abc)[0])

            # Saving the Model

            with open ('model.pkl' , 'wb') as f:
                pickle.dump(nb_obj , f)
            with open ('scaled.pkl' , 'wb') as p:
                pickle.dump(self.sc_obj , p)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")


if __name__ == "__main__":
    try:
        obj = HEART_DISEASE_PREDICTION("Z:\ML Viharatech Projects\Mini Project 2\heart.csv")
        obj.outliers_check()
        obj.feature_Select()
        obj.data_balance()
        # obj.algorithms()
        obj.best_model_train()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")