from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle

# Loads Boston House Price dataset
df_train = pd.read_csv("../train.csv")
df_test = pd.read_csv("../test.csv")

X, Y = df_train.drop(['medv', 'ID'], axis=1), df_train['medv']
# X_test, Y_test = df_test.drop(['medv', 'ID'], axis=1), df_test['medv']

# X, Y = pd.concat([X_train, X_test]), pd.concat([Y_train, Y_test])
print(X.info())
model = RandomForestRegressor()
model.fit(X, Y)

# Apply Model to Make Prediction
file_name = 'model.pkl'

pickle.dump(model, open(file_name, 'wb'))


