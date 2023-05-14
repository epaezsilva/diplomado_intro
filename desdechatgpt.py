import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Cargar los datos de MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocesamiento de datos
X_train = np.array([cv2.resize(image, (32, 32)) for image in X_train])
X_test = np.array([cv2.resize(image, (32, 32)) for image in X_test])

X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

# Construcción del modelo
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluación del modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Predicción de caracteres
img = cv2.imread('caracter.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (32, 32))
img = img.reshape(1, 32, 32, 1)
img = img.astype('float32')
img /= 255

prediction = model.predict(img)
print(prediction)
