import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()


# 1.1
# k = 5
KNN_model = KNeighborsClassifier(n_neighbors = 5)
KNN_model.fit(X_train_n, y_train)
y_test_p = KNN_model.predict(X_test)
y_train_p = KNN_model.predict(X_train_n)
print("KNN klasifikacija (k=5): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (k=5)\nTocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()


# 1.2
# k = 1
KNN_model = KNeighborsClassifier(n_neighbors = 1)
KNN_model.fit(X_train_n, y_train)
y_test_p = KNN_model.predict(X_test)
y_train_p = KNN_model.predict(X_train_n)
print("KNN klasifikacija (k=1): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (k=1)\nTocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()

# k = 100
KNN_model = KNeighborsClassifier(n_neighbors = 100)
KNN_model.fit(X_train_n, y_train)
y_test_p = KNN_model.predict(X_test)
y_train_p = KNN_model.predict(X_train_n)
print("KNN klasifikacija (k=100): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (k=100)\nTocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()


# 2
param_grid = {'n_neighbors': np.arange(1,100)}

svm_gscv = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5, scoring ='accuracy', n_jobs =-1)
svm_gscv.fit(X_train_n, y_train)
print('KNN grid search result: ' + str(svm_gscv.best_params_))
#print(svm_gscv.best_score_)
#print(svm_gscv.cv_results_)


# 3
SVM_model = svm.SVC(kernel='rbf', gamma = 0.1, C=0.1)
SVM_model.fit(X_train_n, y_train)
y_test_p = SVM_model.predict(X_test)
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("SVM(G=0.1,C=0.1)\nTocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()


SVM_model = svm.SVC(kernel='rbf', gamma = 0.1, C=1)
SVM_model.fit(X_train_n, y_train)
y_test_p = SVM_model.predict(X_test)
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("SVM(G=0.1,C=1)\nTocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()


SVM_model = svm.SVC(kernel='rbf', gamma = 0.1, C=10)
SVM_model.fit(X_train_n, y_train)
y_test_p = SVM_model.predict(X_test)
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("SVM(G=0.1,C=10)\nTocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()


SVM_model = svm.SVC(kernel='rbf', gamma = 1, C=0.1)
SVM_model.fit(X_train_n, y_train)
y_test_p = SVM_model.predict(X_test)
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("SVM(G=1,C=0.1)\nTocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()


SVM_model = svm.SVC(kernel='rbf', gamma = 1, C=1)
SVM_model.fit(X_train_n, y_train)
y_test_p = SVM_model.predict(X_test)
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("SVM(G=1,C=1)\nTocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()


SVM_model = svm.SVC(kernel='rbf', gamma = 5, C=1)
SVM_model.fit(X_train_n, y_train)
y_test_p = SVM_model.predict(X_test)
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("SVM(G=5,C=1)\nTocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()


# 4
param_grid = {'gamma': np.arange(0.1,10,0.1), 'C': np.arange(0.1,10,0.1)}

svm_gscv = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=5, scoring ='accuracy', n_jobs =-1)
svm_gscv.fit(X_train_n, y_train)
print('SVM grid search result: ' + str(svm_gscv.best_params_))

plt.show()