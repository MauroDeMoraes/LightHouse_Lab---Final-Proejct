# -*- coding: utf-8 -*-
"""FP_Modelling.ipynb"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import pickle
from sklearn.linear_model import LogisticRegression


### Loading data ###
data = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/final_project/Final_dataset_top8leagues_LV-22-03.csv')

# Delete columns that shows the result to the model. with this columns the model will overfit
data = data.drop(columns=['Pointswon_HT','Pointswon_AT','HAttack_effic', 'AAttack_effic','FTHG', 'FTAG',
                          'HDefense_effic',	'ADefense_effic','HTHG','HTAG', 'HTR'])


# Alter columns to categorical 
data['Country'] = data['Country'].astype('category')
data['League'] = data['League'].astype('category')
data['Season'] = data['Season'].astype('category')
data['Season'] = data['Season'].astype('category')
data['HomeTeam'] = data['HomeTeam'].astype('category')
data['AwayTeam'] = data['AwayTeam'].astype('category')
data['FTR'] = data['FTR'].astype('category')
data['Year'] = data['Year'].astype('category')
data['Month'] = data['Month'].astype('category')
data['Day'] = data['Day'].astype('category')
data['Week-Day'] = data['Week-Day'].astype('category')


### Scalling data ###

scaler = MinMaxScaler()
# Save the variable you don't want to scale
name_var = data[['Country', 'League', 'Season', 'HomeTeam', 'AwayTeam',  'Year', 'Month', 'Day', 'Week-Day','Tied', 'Away_won','Home_won','FTR']]

# Fit scaler to your data
scaler.fit(data.drop(['Country', 'League', 'Season', 'HomeTeam', 'AwayTeam',  'Year', 'Month', 'Day', 'Week-Day','Tied', 'Away_won','Home_won','FTR'], axis = 1))

# Calculate scaled values and store them in a separate object
scaled_values = scaler.transform(data.drop(['Country', 'League', 'Season', 'HomeTeam', 'AwayTeam',  'Year', 'Month', 'Day', 'Week-Day','Tied', 'Away_won','Home_won','FTR'], axis = 1))

data_scl = pd.DataFrame(scaled_values, index = data.index, columns = data.drop(['Country', 'League', 'Season', 'HomeTeam', 'AwayTeam',  'Year', 'Month', 'Day', 'Week-Day','Tied', 'Away_won','Home_won','FTR'], axis = 1).columns)
data_scl[['Country', 'League', 'Season', 'HomeTeam', 'AwayTeam',  'Year', 'Month', 'Day', 'Week-Day','Tied', 'Away_won','Home_won', 'FTR']] = name_var


### Spliting Data ###

X = X.drop(columns=(['FTR']))
y = X['FTR'] # H home A away D draw
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


## Checking target ballancing ##
counts = np.bincount(y.iloc[:,])
print(counts[0], 100 * float(counts[0]) / len(y))
print(counts[1], 100 * float(counts[1]) / len(y))
print(counts[2], 100 * float(counts[2]) / len(y))


## Data Balacing"""
oversample = SMOTE()
X_smt, y_smt = oversample.fit_resample(X, y)
# summarize distribution
counter = Counter(y_smt)
for k,v in counter.items():
	per = v / len(y) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.show()

## Splitting after balanced ##
X_train,X_test,y_train,y_test=train_test_split(X_smt,y_smt,test_size=0.2,random_state=0)

#### Modeling ####

## Radom Forest ##
random_f = RandomForestClassifier(n_estimators=300)
random_f.fit(X_train, y_train)
y_pred_Ran = random_f.predict(X_test)

# compute accuracy
accuracy_RF = accuracy_score(y_test,y_pred_Ran)


## XGBosst ##
GBosst = GradientBoostingClassifier(n_estimators=800, learning_rate=0.25,  
                                 max_depth=10, random_state=0).fit(X_train, y_train)
y_pred_G = GBosst.predict(X_test)

# compute accuracy
accuracy_XGB = accuracy_score(y_test,y_pred_G)



### Logist Regression ## 
# instantiate the model (using the default parameters)
logreg_nc = LogisticRegression()
# fit the model with data
logreg_nc.fit(X_train,y_train)
y_pred =logreg_nc.predict(X_test)

# compute accuracy
accuracy_LR = accuracy_score(y_test,y_pred)

# Pritting the accuracy for all models. 
print(f'Accuracy Random Forest: {accuracy_RF}, \tAccuracy XGBosst: {accuracy_XGB},\tAccuracy Logistic Regression: {accuracy_LR}')

## import pickle ##
filename = 'Model_xgboosting_03-25.p'
pickle.dump(GBosst,open(filename, 'wb'))

# Grid Seach for best hyperparametrs

param_grid = {
  "learning_rate"    : [0.05, 0.15, 0.2, 0.25, 0.3] ,
  "max_depth"       : [1, 3, 5, 10],
  "n_estimators": [100, 330, 600],
  "ccp_alpha"        : [ 0.0, 0.1]
}
k_folds = 5

grid_bosst = GridSearchCV(GBosst, param_grid=param_grid, verbose=1, n_jobs=-1) # verbose=1 -> print results, n_jobs=-1 -> use all processors in parallel
grid_result_naive = grid_bosst.fit(X_train, y_train)

best_result = grid_result_naive.best_score_
best_learning_rate = grid_result_naive.best_params_['learning_rate']
best_max_depth = grid_result_naive.best_params_['max_depth']
best_n_estimators = grid_result_naive.best_params_['n_estimators']
best_ccp_alpha = grid_result_naive.best_params_['ccp_alpha']
print(f'Best score:\t{best_result}\nbest_learning_rate\t{best_learning_rate}\nBest max_depth:\t{best_max_depth}\nbest_n_estimators:\t{best_n_estimators}\nbest_ccp_alpha:{best_ccp_alpha}')



