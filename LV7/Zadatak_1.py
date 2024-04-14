import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X


# 1.1
# creates a 2x3 grid (max 6 plots)
def plot_data(X):
    plt.figure()
    for x in range(len(X)):
        ax = plt.subplot(230 + x + 1)
        plt.scatter(X[x][:,0],X[x][:,1])
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('flagc = ' + str(x+1))

    plt.subplots_adjust(wspace=0.5, hspace=0.5)


# generiranje podatkovnih primjera
X1 = generate_data(500, 1)

X2 = generate_data(500, 2)

X3 = generate_data(500, 3)

X4 = generate_data(500, 4)

X5 = generate_data(500, 5)

plot_data([X1, X2, X3, X4, X5])


# 1.2
# creates a 2x3 grid (max 6 plots)
def fit_and_plot(X, k):
    plt.figure()
    km = KMeans(n_clusters = k, init ='random', n_init =5, random_state =0)
    for i in range(len(X)):
        km.fit(X[i])
        labels = km.predict(X[i])
        ax = plt.subplot(230 + i + 1)
        ax.scatter(X[i][:,0],X[i][:,1], c=labels)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title('flagx = ' + str(i+1))
    
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.suptitle('KMeans, k=' + str(k))


# k = 3
fit_and_plot([X1, X2, X3, X4, X5], 3)

# k = 4
fit_and_plot([X1, X2, X3, X4, X5], 4)

# k = 5
fit_and_plot([X1, X2, X3, X4, X5], 5)



# 1.3
# ?????



plt.show()