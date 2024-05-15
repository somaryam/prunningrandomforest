import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import tree
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#
data = '/content/heart.csv'
df = pd.read_csv(data)
df.head()

#
X = df.drop(columns=['target'])
y = df['target']
print(X.shape)
print(y.shape)

#
x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y)
print(x_train.shape)
print(x_test.shape)

#
clf = tree.DecisionTreeClassifier(random_state=0)
clf.fit(x_train,y_train)
y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)

#
plt.figure(figsize=(20,20))
features = df.columns
classes = ['Not heart disease','heart disease']
tree.plot_tree(clf,feature_names=features,class_names=classes,filled=True)
plt.show()



## apres visionnange 

from sklearn.ensemble import RandomForestClassifier

# Initialisation et entraînement du modèle Random Forest Classifier
clf = RandomForestClassifier(random_state=0)
clf.fit(x_train, y_train)
y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)

# Affichage des scores de précision
print(f'Train score {accuracy_score(y_train_pred, y_train)}')
print(f'Test score {accuracy_score(y_test_pred, y_test)}')

# Affichage des matrices de confusion
plot_confusionmatrix(y_train_pred, y_train, dom='Train')
plot_confusionmatrix(y_test_pred, y_test, dom='Test')



#pre prunning
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Définition des hyperparamètres à tester
params = {'max_depth': [2, 4, 6, 8, 10, 12],
          'min_samples_split': [2, 3, 4],
          'min_samples_leaf': [1, 2]}

# Initialisation du modèle Random Forest Classifier
clf = RandomForestClassifier(random_state=0)

# Recherche des meilleurs hyperparamètres avec GridSearchCV
gcv = GridSearchCV(estimator=clf, param_grid=params)
gcv.fit(x_train, y_train)

# Entraînement du modèle avec les meilleurs hyperparamètres trouvés
model = gcv.best_estimator_
model.fit(x_train, y_train)

# Prédiction sur les ensembles d'entraînement et de test
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Affichage des scores de précision
print(f'Train score {accuracy_score(y_train_pred, y_train)}')
print(f'Test score {accuracy_score(y_test_pred, y_test)}')

# Affichage des matrices de confusion
plot_confusionmatrix(y_train_pred, y_train, dom='Train')
plot_confusionmatrix(y_test_pred, y_test, dom='Test')

# Visualisation de l'arbre
plt.figure(figsize=(20, 20))
tree.plot_tree(model.estimators_[0], feature_names=features, class_names=classes, filled=True)
plt.show()







#post prunning
# Importation des bibliothèques nécessaires
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Initialisation d'une liste pour stocker les modèles pour chaque alpha
clfs = []
for ccp_alpha in ccp_alphas:
    # Initialisation du modèle Random Forest Classifier avec ccp_alpha spécifié
    clf = RandomForestClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(x_train, y_train)
    clfs.append(clf)

# Suppression du dernier modèle et de la dernière valeur de ccp_alpha
clfs = clfs[:-1]

# Calcul des scores de précision pour chaque modèle
train_acc = []
test_acc = []
for c in clfs:
    y_train_pred = c.predict(x_train)
    y_test_pred = c.predict(x_test)
    train_acc.append(accuracy_score(y_train_pred, y_train))
    test_acc.append(accuracy_score(y_test_pred, y_test))

# Affichage des scores de précision
print('Train scores:', train_acc)
print('Test scores:', test_acc)

# Entraînement d'un modèle avec ccp_alpha = 0.020
clf_ = RandomForestClassifier(random_state=0, ccp_alpha=0.020)
clf_.fit(x_train, y_train)
y_train_pred = clf_.predict(x_train)
y_test_pred = clf_.predict(x_test)

# Affichage des scores de précision et des matrices de confusion pour le modèle avec ccp_alpha = 0.020
print(f'Train score {accuracy_score(y_train_pred, y_train)}')
print(f'Test score {accuracy_score(y_test_pred, y_test)}')

# Affichage des matrices de confusion
plot_confusionmatrix(y_train_pred, y_train, dom='Train')
plot_confusionmatrix(y_test_pred, y_test, dom='Test')

# Sélection d'un arbre aléatoire dans la forêt
tree_in_forest = np.random.randint(len(clf_.estimators_))

# Visualisation de l'arbre pour le modèle avec ccp_alpha = 0.020
plt.figure(figsize=(20, 20))
tree.plot_tree(clf_, feature_names=features, class_names=classes, filled=True)
plt.show()
