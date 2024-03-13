import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")
img = img[:,:,0].copy()

print(img.shape)
print(img.dtype)


plt.figure()
plt.imshow(img, cmap="gray")
plt.title("Normal")


brightness_increase = 75                                # by how much to increase the brightness
brighter_img = img + brightness_increase                # increase it
ind = ((img.astype(int) + brightness_increase) > 255)   # indexes of pixels that overflowed
brighter_img[ind] = 255                                 # set the that overflowed to the max value

plt.figure()
plt.imshow(brighter_img, cmap="gray")
plt.title("Brighter")


quarter_image = img[:,int(img.shape[1]/4):int(img.shape[1]/4)*2]

plt.figure()
plt.imshow(quarter_image, cmap="gray")
plt.title("Quarter")


rotated_img = np.rot90(img, axes=(1,0))

plt.figure()
plt.imshow(rotated_img, cmap="gray")
plt.title("Rotated")


flipped_img = np.fliplr(img)

plt.figure()
plt.imshow(flipped_img, cmap="gray")
plt.title("Flipped")

plt.show()