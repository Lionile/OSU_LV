import numpy as np
from tensorflow import keras
from keras import layers
from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

model = keras.models.load_model("LV8_model.keras")

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

