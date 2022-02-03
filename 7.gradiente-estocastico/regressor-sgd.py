import math
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = pd.read_excel('../0.dados/boston.xlsx')

x = data.iloc[:, 1:-1].to_numpy()
y = data.iloc[:, -1].to_numpy()

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=200, shuffle=True, random_state=0)

escala = StandardScaler()
escala.fit(x_train)
x_train = escala.transform(x_train)
x_test = escala.transform(x_test)

SGDR = SGDRegressor(loss='squared_loss', alpha=0.1, penalty='l2')
SGDR = SGDR.fit(x_train, y_train)

y_predict_test = SGDR.predict(x_test)

mse = mean_squared_error(y_test, y_predict_test)
rmse = math.sqrt(mse)

print('%4s = %.4f' % ('rmse', rmse))
