import numpy as np
import matplotlib.pyplot as plt

plt.plot([1, 3], [1, 1], 'b', linewidth=1, marker="o", markersize=5)
plt.plot([1, 2], [1, 2], 'b', linewidth=1, marker="o", markersize=5)
plt.plot([2, 3], [2, 2], 'b', linewidth=1, marker="o", markersize=5)
plt.plot([3, 3], [2, 1], 'b', linewidth=1, marker="o", markersize=5)

plt.axis([0,4,0,4])

plt.xlabel('x os')
plt.ylabel('y os')
plt.title('primjer')
plt.show()