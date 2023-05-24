from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

np.random.seed(42)

iris = load_iris()
X = iris["data"][:, :]
y = iris["target"]

rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(X, y)
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)
