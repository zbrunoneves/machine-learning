import math
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_excel('../0.dados/boston.xlsx')

x = data.iloc[:, 1:-1].to_numpy()
y = data.iloc[:, -1].to_numpy()

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=200, shuffle=True, random_state=0)

# graus polinomiais de 1 a 5
for d in range(1, 6):

    PF = PolynomialFeatures(degree=d).fit(x_train)
    x_train_p = PF.transform(x_train)
    x_test_p = PF.transform(x_test)

    LR = LinearRegression().fit(x_train_p, y_train)

    y_predict_test = LR.predict(x_test_p)

    mse = mean_squared_error(y_test, y_predict_test)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_predict_test)

    print('grau =', d)
    print('%4s = %.4f' % ('mse', mse))
    print('%4s = %.4f' % ('rmse', rmse))
    print('%4s = %.4f' % ('r2', r2), '\n')
