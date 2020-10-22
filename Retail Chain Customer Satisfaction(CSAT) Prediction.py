import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

os.getcwd()
os.chdir("...\\locationfolder")

FullRaw = pd.read_csv("...\\RetailChain.csv")

#Checking for Missing Value
FullRaw.isnull().sum()
FullRaw = FullRaw.fillna(FullRaw.mean())

#Divide FullRaw into Train and Test by random sampling:
from sklearn.model_selection import train_test_split
Train, Test = train_test_split(FullRaw, test_size=0.3, random_state = 123) # Split 70-30%

# Step 3: Divide into Xs (Indepenedents) and Y (Dependent)
Train_X = Train.drop(['Customer_Satisfaction'], axis = 1).copy()
Train_Y = Train['Customer_Satisfaction'].copy()
Test_X = Test.drop(['Customer_Satisfaction'], axis = 1).copy()
Test_Y = Test['Customer_Satisfaction'].copy()

Train_X.shape
Train_Y.shape
Test_X.shape
Test_Y.shape

import xgboost
reg=xgboost.XGBRegressor()

# =============================================================================
# booster=['gbtree','gblinear']
# base_score=[0.25,0.5,0.75,1]
# 
# ## Hyper Parameter Optimization
# n_estimators = [100, 500, 1000, 1250, 1500]
# max_depth = [2, 3, 5, 7, 10]
# booster=['gbtree','gblinear']
# learning_rate=[0.05,0.1,0.15,0.20]
# min_child_weight=[1,2,3,4]
# 
# # Define the grid of hyperparameters to search
# hyperparameter_grid = {
#     'n_estimators': n_estimators,
#     'max_depth':max_depth,
#     'learning_rate':learning_rate,
#     'min_child_weight':min_child_weight,
#     'booster':booster,
#     'base_score':base_score
#     }
# 
# # Set up the random search with 4-fold cross validation
# from sklearn.model_selection import RandomizedSearchCV
# random_cv = RandomizedSearchCV(estimator=reg,
#             param_distributions=hyperparameter_grid,
#             cv=5, n_iter=50,
#             scoring = 'neg_mean_absolute_error',n_jobs = 4,
#             verbose = 5, 
#             return_train_score = True,
#             random_state=42)
# 
# random_cv.fit(Train_X,Train_Y)
# 
# random_cv.best_estimator_
# 
# reg = xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints=None,
#              learning_rate=0.1, max_delta_step=0, max_depth=2,
#              min_child_weight=1,monotone_constraints=None,
#              n_estimators=1000, n_jobs=0, num_parallel_tree=1,
#              objective='reg:squarederror', random_state=0, reg_alpha=0,
#              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
#              validate_parameters=False, verbosity=None)
#
# Model1=reg.fit(Train_X,Train_Y)
# Model1.score(Train_X,Train_Y)
# #0.999960890451533
# =============================================================================

Model=reg.fit(Train_X,Train_Y)
Model.score(Train_X,Train_Y)
#0.9999994973280882

import pickle
filename = 'Final_Model.pkl'
pickle.dump(reg, open(filename, 'wb'))

# =============================================================================
# y_pred=reg.predict(Test_X)
# Model.score(Test_X, y_pred)
# 
# # Model Evaluation
# RMSE = np.sqrt(np.mean((Test_Y - y_pred)**2))
# # 0.37373652724563433
# 
# MAPE = (np.mean(np.abs(((Test_Y - y_pred)/Test_Y))))*100
# # 3.789928021026265
# =============================================================================
