import pandas as pd

from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

from sklearn.linear_model import LogisticRegression

# =============================================================================
# REGRESSÃO LOGÍSTICA PARA O CONJUNTO ORANGE
# =============================================================================

data = pd.read_csv('../0.dados/orange-telecom-churn-data.csv')

non_numeric_variables = [
    i for i in data.columns if data[i].dtype == 'object' or i == 'area_code'
]

data = data.drop(['state', 'phone_number'], axis=1)

data = pd.get_dummies(data, columns=['area_code'])

binarizer = LabelBinarizer()
for v in ['intl_plan', 'voice_mail_plan']:
    data[v] = binarizer.fit_transform(data[v])

selected_variables = [
    'account_length',
    'intl_plan',
    'voice_mail_plan',
    'number_vmail_messages',
    'total_day_minutes',
    'total_day_calls',
    'total_day_charge',
    'total_eve_minutes',
    'total_eve_calls',
    'total_eve_charge',
    'total_night_minutes',
    'total_night_calls',
    'total_night_charge',
    'total_intl_minutes',
    'total_intl_calls',
    'total_intl_charge',
    'number_customer_service_calls',
    #'area_code_408',
    #'area_code_415',
    #'area_code_510'
    'churned'
    ]

data = data[selected_variables]

shuffled_data = data.sample(frac=1)

x = shuffled_data.loc[:, shuffled_data.columns != 'churned'].values
y = shuffled_data.loc[:, shuffled_data.columns == 'churned'].values

q = 4000  # qtde de amostras selecionadas para treinamento

x_train = x[:q, :]
y_train = y[:q].ravel()

x_test = x[q:, :]
y_test = y[q:].ravel()

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

classifier = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs')

classifier.fit(x_train, y_train)

y_prediction = classifier.predict(x_test)

acc = sum(y_prediction == y_test)/len(y_test)

print("Acurácia = %.1f %%" % (100*acc))
