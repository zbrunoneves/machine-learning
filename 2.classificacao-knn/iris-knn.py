import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# =============================================================================
# CLASSIFICADOR KNN - K NEARST NEIGHBORS
# =============================================================================

data = pd.read_csv('../0.dados/iris-data.csv')
data['species'] = data['species'].apply(lambda r: r.replace('Iris-', ''))

data_shuffled = data.sample(frac=1)

x_train = data_shuffled.iloc[:100, :-1].values
y_train = data_shuffled.iloc[:100, -1].values

x_test = data_shuffled.iloc[100:, :-1].values
y_test = data_shuffled.iloc[100:, -1].values

classifier = KNeighborsClassifier(n_neighbors=5).fit(x_train, y_train)

y_prediction = classifier.predict(x_test)

print("Acurácia:", sum(y_prediction == y_test)/len(y_test))
