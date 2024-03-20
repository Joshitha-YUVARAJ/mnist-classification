# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

![Screenshot 2024-03-20 132839](https://github.com/Joshitha-YUVARAJ/mnist-classification/assets/145742770/a860b89e-8efd-4552-9480-785f22ff9cf0)


## Neural Network Model

![Screenshot 2024-03-18 100959](https://github.com/Joshitha-YUVARAJ/mnist-classification/assets/145742770/9d17fc08-9fd9-49ca-9bf6-c196631a7a90)


## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries

### STEP 2:
Build a CNN model
### STEP 3:
Compile and fit the model and then predict

## PROGRAM

### Name:YUVARAJ JOSHITHA
### Register Number:212223240189
## Library Importing:
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

```
## Shaping:
```
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
```
## One Hot Encoding:
```
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
```
## CNN Model:
```
model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Flatten())
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
```
## Metrics:
```
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
```
## Prediction:
```
img = image.load_img('/content/image four.jpg')
type(img)
img = image.load_img('/content/image four.jpg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2024-03-18 102636](https://github.com/Joshitha-YUVARAJ/mnist-classification/assets/145742770/a31a3bd0-706f-4b3b-a8ec-e6b8bc774686)
![Screenshot 2024-03-18 102653](https://github.com/Joshitha-YUVARAJ/mnist-classification/assets/145742770/fb15e353-a604-4814-804e-719bd99993ef)



### Classification Report

![Screenshot 2024-03-18 102717](https://github.com/Joshitha-YUVARAJ/mnist-classification/assets/145742770/abc17a7d-6b52-4523-9d22-b770696955a0)


### Confusion Matrix

![Screenshot 2024-03-18 102704](https://github.com/Joshitha-YUVARAJ/mnist-classification/assets/145742770/35087462-c1fc-4851-9239-0add15478791)


### New Sample Data Prediction

.![Screenshot 2024-03-18 102738](https://github.com/Joshitha-YUVARAJ/mnist-classification/assets/145742770/7cf6c518-0c5e-422f-8287-40f016ade082)


## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
