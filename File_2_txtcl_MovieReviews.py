###############################################
# Binary Classification
# Tutorial showing basic application of transfer learning with TF Hub or Keras
# TF Hub is a lib and platform for transfer learning
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf

import pip
# pip install -q tensorflow-hub                          # ! ==> It means run it as a shell command rather than a notebook command
# Same thing happens if take out the ! and run it in terminal
# pip install -q tfds-nightly
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())  # Review what is dis
print("Hub ver: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# Download IMDB Dataset #
# Split training set ==> 60:40 ==> training 15000: CV 10000 and 25000 test
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)  # Figure out what does this do (the as_supervised thing)

# Explore data #
# Data format ==> each Ex is a sentence showing movie review and a corresponding label
# 0 = - review, 1 = + review
train_ex_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_ex_batch  # ==> prints out first 10 Ex
train_labels_batch

# Building the model #
# NNs ==> created by stacking layers ==> 3 main architectural decisions
# 1) How to represent the txt
# 2) How many layers to use in the model
# 3) How many HIDDEN UNITS for each layer
# Represent txt ==> convert sentences into embedding vectors
# Use pre-trained txt embedding (in this Ex, use model from TF Hub) as the first layer ==> 3 benefits
# 1) don't have to worry abt txt processing
# 2) can benefit from transfer learning
# 3) embedding has a fixed sz ==> simpler to process

# Create a Keras layer ==> use the model to embed sentences ==> NO MATTER THE LENGTH OF INPUT TXT
# ==> OUTPUT SHAPE OF EMBEDDINGS ==> (num_ex, embedding_dim)
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
hub_layer(train_ex_batch[:3])           # First 3 Ex

# BUILD FULL MODEL
model = tf.keras.Sequential()
model.add(hub_layer)                                        # Review what is dis
# ^ TF Hub layer ==> use pre-trained Save Model to map a sentence into its embedding vec
# Pre-trained model ==> split sentence into tokens ==> embeds each token ==> combines the embedding; dim:(#Ex,emb_dim)
model.add(tf.keras.layers.Dense(16, activation='relu'))     # fixed length output vector ==> piped through Dense layer
model.add(tf.keras.layers.Dense(1))                         # Single output node ==> Sigmoid ==> output prob Range 0-1
# Layers stacked sequentially to build the classifier

model.summary()

# Loss func and optim #
# Binary Classification and Sigmoid ==> Use binary_crossentropy loss func ==> BETTER FOR DEALING WITH PROB
# ^ Measures dist bet prob distributions (Ground Truth and pred)
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model #
history = model.fit(train_data.shuffle(10000).batch(512),       # Train for 20 epochs in mini-batches of 512 samples
                    epochs=20,                                  # 20 ite over all x_train and y_train
                    validation_data=validation_data.batch(512),
                    verbose=1)                                  # BUT WTF IS DIS ???
# With 50 epochs ==> loss : 0.482 - accu : 0.847
# 20 epochs ==> loss : 0.320 - accu : 0.855


# Eval the model #
results = model.evaluate(test_data.batch(512), verbose=2)

for name,value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name,value))                            # FIGURE OUT WHY IS SYNTAX LIKE DIS
