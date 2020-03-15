from __future__ import absolute_import, division, print_function, unicode_literals

# TF and tf.keras

import tensorflow as tf
from tensorflow import keras

# Helper Lib
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_img, train_labels), (test_img, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_img.shape     #Return shape of the image
len(train_labels)   #Number of labels

train_labels

test_img.shape
len(test_labels)

#PREPROCESS DATA

plt.figure()
plt.imshow(train_img[0])  #Show the first img in the collection
plt.colorbar()
plt.grid(False)
plt.show()

#GRAYSCALE BOI
train_img = train_img/255.0
test_img = test_img/255.0

plt.figure(figsize=(10,10))
for i in range(25):   #Displaying first 25 img
    plt.subplot(5,5,i+1)            #5x5 size img at i+1th location
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_img[i], cmap=plt.cm.binary) #cmap = wtf
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#BUILD THE MODEL
## SETUP LAYERS
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),        #Reformat ==> Transforms img format from 2D array to 1D array
    keras.layers.Dense(128, activation = 'relu'),       #Fully connected layers
    keras.layers.Dense(10)                              #Softmax layer ==> returns array of 10 prob scores (sum = 1)
])                                                      #Output a fookin logits
##COMPILE MODEL
model.compile(optimizer='adam',                         #Optim ==> how model is updated based on data shown and loss func
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])
##TRAIN MODEL
### Procedure
#### 1) Feed training data to the model (train_img and train_labels)
#### 2) Model learns to assoc img and labels
#### 3) Ask model to make predictions in test set (test_img)
#### 4) Verify pred with test_labels

### 1) FEED THE MODEL
model.fit(train_img, train_labels, epochs=10)
### 2) EVAL ACCU
test_loss, test_acc = model.evaluate(test_img, test_labels, verbose = 2) ###### WTF IS VERBOSE #####
print('\nTest Accuracy: ', test_acc)
### 3) PRED
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])    # Attach a softmax to convert the logits to prob
pred = probability_model.predict(test_img)
pred[0]             # Test pred ==> Output an array of prob
np.argmax(pred[0])  # Output the index of the one with highest prob (DON'T FORGET AN OUTPUT OF 9 ==> THE 10TH ONE)
test_labels[0]      # Check the answer

def plot_img(i, pred_array, true_label, img):               # A Graph to look at full set of 10 class pred
    pred_array, true_label, img = pred_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap = plt.cm.binary)

    pred_label = np.argmax(pred_array)
    if pred_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[pred_label],               # REVIEW DIS BOI
                                         100*np.max(pred_array),
                                         class_names[true_label]),
                                         color = color)
def plot_value_array(i, pred_array, true_label):
    pred_array, true_label = pred_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))                                                     # xticks has tick on x axis range from 0-9
    plt.yticks([])
    thisplot = plt.bar(range(10), pred_array, color = "#777777")              # x = 0-9, y = pred_array
    plt.ylim([0,1])
    pred_label = np.argmax(pred_array)

    thisplot[pred_label].set_color('red')                                     # The bar that got wrong = red
    thisplot[true_label].set_color('blue')

### 4) VERIFY PRED
#test
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_img(i, pred[i], test_labels, test_img)
plt.subplot(1,2,2)
plot_value_array(i, pred[i], test_labels)
plt.show()

#real stuff ==> Plot the first X test img pred_labels and true labels.
# Correct = Blue, Incorrect = Red
num_rows = 6
num_cols = 6
num_img = num_cols*num_rows
plt.figure(figsize=(2*2*num_cols, 2*num_rows))              #### REVIEW DIS BUNCH
for i in range(num_img):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_img(i, pred[i], test_labels, test_img)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, pred[i], test_labels)
plt.tight_layout()
plt.show()

# USE THE TRAINED MODEL
## Make pred about a single img
img = test_img[3]
print(img.shape)

## Keras ==> good at dealing with batch or collection of examples at once ==> need to add img to a list (batch)
img = (np.expand_dims(img,0))
print(img.shape)

pred_single = probability_model.predict(img)        # The learned model with output as a prob
print(pred_single)                                  # keras.Model.predict returns a list of lists ==> one list for
                                                    # each img

plot_value_array(3, pred_single[0], test_labels)    # Put in 3 bc it will correspond with the true_labels[3] ==>
                                                    # compare with test_img[3]
_ = plt.xticks(range(10), class_names, rotation = 45)       # What's the mechanism of this '_'

np.argmax(pred_single[0])

