import math
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_excel('../0.dados/boston.xlsx')

x = data.iloc[:, 1:-1].to_numpy()
y = data.iloc[:, -1].to_numpy()

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=200, shuffle=True, random_state=0)

PF = PolynomialFeatures(degree=2).fit(x_train)
x_train_p = PF.transform(x_train)
x_test_p = PF.transform(x_test)

# L1
L = Lasso(alpha=0.1).fit(x_train_p, y_train)

# L2
R = Ridge(alpha=50.0).fit(x_train_p, y_train)

y_predict_test = R.predict(x_test_p)

mse = mean_squared_error(y_test, y_predict_test)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_predict_test)

print('grau =', 2)
print('%4s = %.4f' % ('rmse', rmse))
