import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# a)
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=plt.cm.colors.ListedColormap(['red', 'blue']))
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, marker='x', cmap=plt.cm.colors.ListedColormap(['red', 'blue']))

# b)
# inicijalizacija i ucenje modela logisticke regresije
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train , y_train)

# c)
theta0 = LogRegression_model.intercept_
theta1 = LogRegression_model.coef_[0,0]
theta2 = LogRegression_model.coef_[0,1]

# d)
# predikcija na skupu podataka za testiranje
y_test_p = LogRegression_model.predict(X_test)
cm = confusion_matrix(y_test, y_test_p)

# tocnost
print (" Tocnost : ", accuracy_score(y_test, y_test_p))
# report
print(classification_report(y_test, y_test_p))

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
disp.plot()

# e)
plt.figure()
y_correct = y_test_p == y_test
plt.scatter(X_test[:,0], X_test[:,1], c=y_correct, cmap=plt.cm.colors.ListedColormap(['red', 'blue']))


plt.show()