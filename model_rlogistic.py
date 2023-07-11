import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import plot_tree

# Charger les données à partir du fichier CSV
data = pd.read_csv('donnees.csv')
print(data.columns)

# Diviser les données en variables indépendantes (X) et variable dépendante (y)
#X = data[['feature1', 'feature2']]
#X = data.drop(columns=['feature2'])
X = data.drop('ContainsBug', axis=1)  # Supprime la variable dependante et prend tout le reste
y = data['ContainsBug'] #prend la variable dependante

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instancier le modèle de régression logistique
model = LogisticRegression()

# Entraîner le modèle sur les données d'entraînement
model.fit(X_train, y_train)

# Faire des prédictions sur les données de test
y_pred = model.predict(X_test)

##############################
# Faire des prédictions sur les données de test
y_pred = model.predict(X_test)

# Calculer la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle : {:.2f}%".format(accuracy * 100))
##############################

# Calculer les métriques de performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Afficher les résultats
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Afficher le monogramme du modèle
plt.figure(figsize=(10, 8))
tree = model.coef_.T  # Coefficients du modèle (monogramme pour la régression logistique)
feature_names = X.columns
print(feature_names)
#_ = plot_tree(tree, feature_names=feature_names, filled=True)


# Get the coefficients of the logistic regression model
coefficients = model.coef_[0]

# Plot the coefficients
plt.figure(figsize=(10, 6))
plt.bar(feature_names, coefficients)
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Coefficients')
plt.title('Logistic Regression Coefficients')
plt.tight_layout()
plt.show()
