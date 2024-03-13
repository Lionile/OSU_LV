import numpy as np
import matplotlib.pyplot as plt

black_square = np.ones((50,50))
white_square = np.zeros((50,50))

left_half = np.vstack((white_square, black_square))
right_half = np.vstack((black_square, white_square))

img = np.hstack((left_half, right_half))

plt.figure()
plt.imshow(img, cmap="gray")

plt.show()