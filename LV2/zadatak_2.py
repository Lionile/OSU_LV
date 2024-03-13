import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("data.csv", delimiter=",", usemask=True, skip_header=True)

print(data[0:5,])

print("Sadrzi: "  + str(data.shape[0]) + " mjerenja")
#plt.scatter(data[0:,1], data[0:,2]) #svaki
plt.scatter(data[0::50,1], data[0::50,2]) #svaki 50-i
plt.show()
print("Visina (muski):")
ind = (data[:,0] == 1)
print("min: " + str(np.min(data[ind,1])))
print("avg: " + str(np.average(data[ind,1])))
print("max: " + str(np.max(data[ind,1])))

print("\nVisina (zene):")
ind = (data[:,0] == 0)
print("min: " + str(np.min(data[ind,1])))
print("avg: " + str(np.average(data[ind,1])))
print("max: " + str(np.max(data[ind,1])))
