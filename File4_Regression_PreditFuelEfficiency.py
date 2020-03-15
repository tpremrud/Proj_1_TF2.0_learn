# Regression # ==> Predict output of continuous value (price or prob)
# Auto MPG dataset ==> build model to predict efficiency of 70s and 80s automobiles
# Has para ==> such as 'cylinders', 'displacement', 'horsepower' and 'weight'
# Use 'seaborn' for 'pairplot' and some func from tensorflow_docs

from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

import tensorflow_docs as tfdocs
