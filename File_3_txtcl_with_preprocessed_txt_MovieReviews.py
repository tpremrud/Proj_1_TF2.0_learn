# Binary classification ==> Data (IMDB) split just like in File_2
# Uses tf.keras
# Setup #
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import numpy as np

print(tf.__version__)

# Download the IMDB Dataset ==> comes packaged in tfds (already preprocessed) ==> review (seq of words) converted to
# seq of integers already #
(train_data, test_data), info = tfds.load(
    # Use the version pre-encoded with an ~8k vocab
    'imdb_reviews/subwords8k',
    # Return the train/test datasets as a tuple
    split = (tfds.Split.TRAIN, tfds.Split.TEST),            # a = (1,2,3) ==> TUPLE ==> immutable (can't be changed)
    # Return (example, label) pairs from the dataset (instead of a dict). ==> REVIEW DIS PLS
    as_supervised=True,
    # Also return the 'info' structure
    with_info=True)             #    ^
                                #    ^
# Try the Encoder # ==> info dataset ^ includes a txt encoder (a tfds.features.text.SubwordTextEncoder)
encoder = info.features['text'].encoder         # REVIEW DIS PLS ==> This txt encoder = reversibly encode and str
print('Vocabulary size: {}'.format(encoder.vocab_size))
sample_str = 'Hello TensorFlow.'

encoded_str = encoder.encode(sample_str)
print('Encoded str is {}'.format(encoded_str))

original_str = encoder.decode(encoded_str)
print('The original str: "{}"'.format(original_str))

assert original_str == sample_str           # Check if both the encoded and the decoded are the same
# Encoder ==> encodes str by breaking it into subwords or characters ==> IF the word is not in dict
# ^ The more the str resembles what is already in the dataset ==> the shorter the encoded representation will be
for ts in encoded_str:                      # REVIEW DIS PLS ==> Show the encoded rep compared to the OG stuff
    print('{} ====> {}'.format(ts, encoder.decode([ts])))

# Explore the data #
# Data = preprocessed ==> already encoded ==> each integer shows specific word-piece in dict
for train_ex, train_label in train_data.take(1):          # What does .take do??
    print('Encoded txt:', train_ex[:10].numpy())
    print('Label: ', train_label.numpy())

encoder.decode(train_ex)        # info structure ==> contains encoder/decoder ==> encoder can be used to recover OG txt

# print('OG training ex: "{}"'.format(encoder.decode(train_ex)))
# for i in train_ex:
#     print('{} ===> {}'.format(i, encoder.decode([i])))

# Prepare the data for training #
# CREATE BATCHES of training data for the model
BUFFER_SZ = 1000

train_batches = (
    train_data
    .shuffle(BUFFER_SZ)
    .padded_batch(32))           # Each txt ==> different lengths ==> use this to zero pad the sequences while batching
                                 # ^
test_batches = (
    test_data
    .padded_batch(32))
# ^ Each batch has shape (batch_sz, sequence_length) ==> Each batch has different length (padding is dynamic)

for example_batch, label_batch in train_batches.take(2):            # what is .take
    print("Batch shape:", example_batch.shape)
    print("label shape:", label_batch.shape)

# Build the model #
# !!!! ==> does not use masking ==> zero-padding is used as part of the input ==> padding length may affect output
# NNs ==> build by stacking the layers ==> need 2 main architectural decisions
# 1) No. of layers
# 2) No. of hidden units for each layer
# In this Ex. ==> input = array of word-indices ==> label = predict 0 or 1 ==> build a continuous BoW style model
model = keras.Sequential([
    keras.layers.Embedding(encoder.vocab_size,16),      # Embedding layer ==> takes encoded vocab and
                                                        # look up the embedding vector for each word-index
                                                        # The vectors add a dim ==> (batch, seq, embedding)
    keras.layers.GlobalAveragePooling1D(),              # Returns fixed length output vector
                                                        # ==> avg over the sequence dim
                                                        # ==> making model to be able to handle various input length
    keras.layers.Dense(1)])                             # For nume stability ==> use 'linear' activation func = logits

model.summary()     # summary of this classifier model

# NOTE: no. of hidden units ==> dimension of the representational space for the layer ==> amount of freedom
# allowed when learning an internal representation ==> more hidden units = can learn more complex rep
# HOWEVER ==> too much = overfit

# Loss function and optim # ==> 'binary_crossentropy' loss func
model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model # ==> passing 'Dataset' obj to the model's fit func (set no. of epochs too)
history = model.fit(train_batches,
                    epochs=10,
                    validation_data=test_batches,
                    validation_steps=30)            # REVIEW DIS

# EVALUATE MODEL #
loss, accuracy = model.evaluate(test_batches)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# Create graph of accu and loss over time # ==> model.fit() returns a 'History' obj ==> a dict with everything recorded
history_dict = history.history
history_dict.keys()                 # Has four entries ==> train and val loss and accu

# Plot training and val loss and accu for comparison #
import  matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" ==> blue dot
plt.plot(epochs, loss, 'bo', label='Training loss')
# "b" ==> solid blue line
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()           # CLEAR FIGURE

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

# NOTE: You can see that the val loss and accu peak after abt 20 epochs ==> OVERFIT ==> see this = stop training after
# that amount of epochs ==> do this with 'callback'