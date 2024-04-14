import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()


# 2.1
pixels = {}

for pixel in img_array_aprox:
    pixels[tuple(pixel)] = 1

print("Number of unique colors: ", len(pixels))


# 2.2, 2.3, 2.4

def predict(img, k):
    km = KMeans(n_clusters=k, init='random', n_init=5, random_state=0)
    km.fit(img)
    labels = km.predict(img)
    centers = km.cluster_centers_
    
    img_array_aprox_kmeans = img.copy()
    for i in range(len(img)):
        img_array_aprox_kmeans[i] = centers[labels[i]]
    
    j = km.inertia_

    return [labels, centers, img_array_aprox_kmeans, j]

def predict_and_plot(img, k, plot_scatter=False):
    labels, centers, img_array_aprox_kmeans, j = predict(img, k)
    
    if plot_scatter:
        plt.figure()
        plt.scatter(img[::10,0], img[::10,1], img[::10,2], c=labels[::10])
    
    plt.figure()
    plt.title("Image after KMeans (k=" + str(k) + ")")
    plt.imshow(np.reshape(img_array_aprox_kmeans, (w,h,d)))
    plt.tight_layout()
    
    return [labels, centers, img_array_aprox_kmeans, j]

# k = 5
predict_and_plot(img_array_aprox, 5, plot_scatter=True)


# k = 8
predict_and_plot(img_array_aprox, 8)


# k = 15
predict_and_plot(img_array_aprox, 15)


# 2.5
# test_4.jpg
img = Image.imread("imgs\\test_3.jpg")
img = img.astype(np.float64) / 255
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
img_array_aprox = img_array.copy()

# k = 8
predict_and_plot(img_array_aprox, 8)


# test_6.jpg
img = Image.imread("imgs\\test_6.jpg")
img = img.astype(np.float64) / 255
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
img_array_aprox = img_array.copy()

# k = 8
predict_and_plot(img_array_aprox, 8)


# 2.6
img = Image.imread("imgs\\test_3.jpg")
img = img.astype(np.float64) / 255
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
img_array_aprox = img_array.copy()

labels, centers, img_array_aprox_kmeans, j1 = predict(img_array_aprox, 3)
labels, centers, img_array_aprox_kmeans, j2 = predict(img_array_aprox, 4)
labels, centers, img_array_aprox_kmeans, j3 = predict(img_array_aprox, 5)
labels, centers, img_array_aprox_kmeans, j4 = predict(img_array_aprox, 6)
labels, centers, img_array_aprox_kmeans, j5 = predict(img_array_aprox, 7)
labels, centers, img_array_aprox_kmeans, j6 = predict(img_array_aprox, 8)
labels, centers, img_array_aprox_kmeans, j7 = predict(img_array_aprox, 9)
labels, centers, img_array_aprox_kmeans, j8 = predict(img_array_aprox, 10)
labels, centers, img_array_aprox_kmeans, j9 = predict(img_array_aprox, 11)
labels, centers, img_array_aprox_kmeans, j10 = predict(img_array_aprox, 12)

j = [j1, j2, j3, j4, j5, j6, j7, j8, j9, j10]

plt.figure()
plt.scatter(range(3, 13), j)


# 2.7
labels, centers, img_array_aprox_kmeans, j3 = predict(img_array_aprox, 5)

for i in range(len(img_array_aprox_kmeans)):
    if(img_array_aprox_kmeans[i] == centers[1]).all():
        img_array_aprox_kmeans[i] = [0, 0, 0]
    else:
        img_array_aprox_kmeans[i] = [1, 1, 1]
        

plt.figure()
plt.title("One color group after KMeans")
plt.imshow(np.reshape(img_array_aprox_kmeans, (w,h,d)))

plt.show()