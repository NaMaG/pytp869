import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

# Charger les données à partir du fichier CSV
data = pd.read_csv('averagesWithBugs.csv')

# Séparer les caractéristiques (X) et la variable cible (y)
X = data[['feature1', 'feature2', 'feature3']]
y = data['variable_cible']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer une instance du modèle de forêt aléatoire
model = RandomForestClassifier()

# Entraîner le modèle sur les données d'entraînement
model.fit(X_train, y_train)

# Visualiser un arbre de décision
tree = model.estimators_[0]  # Sélectionner le premier arbre
feature_names = X.columns.tolist()  # Convertir les noms des colonnes en une liste
print(feature_names)
class_names = [str(label) for label in model.classes_]  # Convertir les noms des classes en une liste
print(class_names)
_ = plot_tree(tree, feature_names=feature_names, filled=True)
plt.show()
plt.savefig('monogram2.png')