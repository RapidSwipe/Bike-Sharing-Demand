import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

dataset=pd.read_csv('train.csv')
train_data=dataset



#Check null variables in data
train_data.isnull().values.any()


train_data["hour"] = [t.hour for t in pd.DatetimeIndex(train_data.datetime)]
train_data["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_data.datetime)]
train_data["month"] = [t.month for t in pd.DatetimeIndex(train_data.datetime)]
train_data['year'] = [t.year for t in pd.DatetimeIndex(train_data.datetime)]




train_data.drop('datetime',axis=1,inplace=True) 


train_data.to_csv('saved_data.csv')
 
corr = train_data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(train_data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(train_data.columns)
ax.set_yticklabels(train_data.columns)
plt.show()

#Dropping column 'workingday' because it's highly correlated to 'day' column
train_data=train_data.drop('workingday',axis=1)


train_data=train_data.drop('casual',axis=1)
train_data=train_data.drop('registered',axis=1)

y=train_data['count'].values
X=train_data.drop('count',axis=1).values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=350)
regressor.fit(X_train,y_train)


y_pred=regressor.predict(X_test)


#Calculating RMSLE
from sklearn.metrics import mean_squared_log_error
np.sqrt(mean_squared_log_error( y_test, y_pred ))


#Random forest RMSLE
#0.33862937908947516
from sklearn.model_selection import GridSearchCV

parameters=[{'n_estimators':[200,250,300,350,400,450], 'max_features': ['auto', 'sqrt', 'log2']}
    ]

grid_search=GridSearchCV(estimator=regressor,
                         param_grid=parameters,
                         scoring='neg_mean_squared_error',
                         cv=10,
                         n_jobs=-1)
grid_search=grid_search.fit(X_train,y_train)
best_acuraccy=grid_search.best_score_
best_parameters=grid_search.best_params_


#Importing vales to predict (Test set)


X_test=pd.read_csv('test.csv')


X_test["hour"] = [t.hour for t in pd.DatetimeIndex(X_test.datetime)]
X_test["day"] = [t.dayofweek for t in pd.DatetimeIndex(X_test.datetime)]
X_test["month"] = [t.month for t in pd.DatetimeIndex(X_test.datetime)]
X_test['year'] = [t.year for t in pd.DatetimeIndex(X_test.datetime)]

X_test_datetime=X_test['datetime'].values
X_test.drop('datetime',axis=1,inplace=True) 

X_test=X_test.drop('workingday',axis=1)
X_test=X_test.values

X_test=sc.fit_transform(X_test)

final_predictions=regressor.predict(X_test)
final_predictions=np.round(final_predictions)


output={'datetime':X_test_datetime,'count':final_predictions}
final_output=pd.DataFrame(output,columns=['datetime','count'])






