#%% ---------- Import Libraries ----------

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model
from tensorflow.keras.layers import Normalization, Rescaling

from pathlib import Path
import os.path

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

#%% ---------- Import Data ----------

dataset = "Drug Vision/Data Combined"
image_dir = Path(dataset)

filepaths = list(image_dir.glob(r"**/*.jpg")) + list(image_dir.glob(r"**/*.png"))

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name="filepath").astype(str)
labels = pd.Series(labels, name="label")

image_df = pd.concat([filepaths, labels], axis=1)

#%% ---------- Data Visualization ----------

random_index = np.random.randint(0, len(image_df), 25)
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(11, 11))

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(image_df.filepath[random_index[i]]))
    ax.set_title(image_df.label[random_index[i]])

plt.tight_layout()

#%% data preprocessing: train-test split, data augmentation, resize, rescaling

# train-test split
train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=42)

# data augmentation: veri artırımı
train_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)

test_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",  # independent -> görüntü
    y_col="label",     # dependent -> target variable -> etiket
    target_size=(224, 224),  # görüntülerin boyutu
    color_mode="rgb",
    class_mode="categorical",
    batch_size=16,
    shuffle=True,
    seed=42,
    subset="training"
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="label",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=16,
    shuffle=True,
    seed=42,
    subset="validation"
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col="filepath",
    y_col="label",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=16,
    shuffle=False
)

# resize, rescale
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(224, 224),
    layers.Rescaling(1./255)
])
























































