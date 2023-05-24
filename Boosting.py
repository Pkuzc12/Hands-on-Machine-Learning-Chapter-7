from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
import numpy as np

np.random.seed(42)

X, y = make_moons(n_samples=200, noise=0.15)

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5
)
ada_clf.fit(X, y)

m = 100
X = 6*np.random.rand(m, 1)-3
y = 0.5*X**2+X+2+np.random.randn(m, 1)

tree_reg1 = DecisionTreeClassifier(max_depth=2)
tree_reg1.fit(X, y)

y2 = y-tree_reg1.predict(X)
tree_reg2 = DecisionTreeClassifier(max_depth=2)
tree_reg2.fit(X, y2)

y3 = y-tree_reg2.predict(X)
tree_reg3 = DecisionTreeClassifier(max_depth=2)
tree_reg3.fit(X, y3)

X_new = 6*np.random.rand(m, 1)-3

y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X, y)

# The following content is omitted.
