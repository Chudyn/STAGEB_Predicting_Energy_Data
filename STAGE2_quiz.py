import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#The goal is to predict energy consumption by appliances.

Data = pd.read_csv('Energy_data.csv')
#print(Data.head()) 
#print(Data.isnull().sum().sort_values(ascending = True)) #no null values
 
df = Data.drop(columns=['date', 'lights'])
print(df)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df)
scaled_feat = scaler.transform(df)
df_MinMaxSc = pd.DataFrame(data = scaled_feat, columns = df.columns)
features_df = df_MinMaxSc.drop(columns=['Appliances'])
target_variable= df_MinMaxSc['Appliances']


x = df.iloc[:, 3].values
y = df.iloc[:, 11].values

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#convert the data into 2 Dimensional array.

#regressor.fit(X_train,Y_train) #training our machine learning model using these data.

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=42)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#convert the data into 2 Dimensional array.
X_train =X_train.reshape(-1, 1)
Y_train =Y_train.reshape(-1, 1)
regressor.fit(X_train,Y_train)
X_test =X_test.reshape(-1, 1)
y_pred = regressor.predict(X_test)
print(y_pred)


#ridge rgression
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.4)
ridge_reg.fit(X_train, Y_train)
print(ridge_reg)

#Feature selcetion and lasso Regression
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(X_train, Y_train)


from sklearn.metrics import r2_score
r2_score= r2_score(Y_test, y_pred)
round(r2_score, 2) #0.64

#Residual sum of squares(RSS)
import numpy as np
rss = np.sum(np.square(Y_test - y_pred))
round(rss,2 )

#Root mean square error(RMSE)
from sklearn.metrics import mean_squared_error
rmse =np.sqrt(mean_squared_error(Y_test, y_pred))
round(rmse,2 ) #3.63

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(Y_test, y_pred)
round(mae, 2) #2.82

