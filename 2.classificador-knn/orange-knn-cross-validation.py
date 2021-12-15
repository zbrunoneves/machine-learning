import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer

# =============================================================================
# CLASSIFICADOR KNN - K NEARST NEIGHBORS
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
    # 'area_code_408',
    # 'area_code_415',
    # 'area_code_510'
    'churned'
]

data = data[selected_variables]

shuffled_data = data.sample(frac=1)

x = shuffled_data.loc[:, shuffled_data.columns != 'churned'].values
y = shuffled_data.loc[:, shuffled_data.columns == 'churned'].values

x_train, y_train, x_test, y_test = \
    train_test_split(x, y, train_size=0.8, shuffle=True)

scores = cross_val_score(
    KNeighborsClassifier(n_neighbors=5, p=2),
    x,
    y.ravel(),
    cv=5
)

print("Scores:", scores*100)
print("Acurácia média = %.1f %%" % (100 * sum(scores)/5))
