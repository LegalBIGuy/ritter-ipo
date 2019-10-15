from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print(tf.__version__)

np.random.seed(42)

df = pd.read_csv('IPO2609FeatureEngineering.csv')

tf.keras.backend.set_floatx('float64')
batch_size = 128

# Train / Val / Test Split
train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10
ipo_train, ipo_test = train_test_split(df, test_size=1 - train_ratio)
ipo_val, ipo_test = train_test_split(ipo_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

train_y = ipo_train.pop('underpriced')
train_x = normalize(ipo_train)
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_dataset = dataset.shuffle(len(train_x)).batch(batch_size).repeat()

val_y = ipo_val.pop('underpriced')
val_x = normalize(ipo_val)
dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_dataset = dataset.shuffle(len(train_x)).batch(batch_size).repeat()

def get_compiled_model():
  model = tf.keras.Sequential([
    keras.layers.Dense(train_x.shape[1], activation='relu',
      kernel_initializer=keras.initializers.he_normal()),
    #keras.layers.Dropout(0.2),
    keras.layers.Dense(15, activation='relu'), #, kernel_regularizer=keras.regularizers.l2(0.01)
    keras.layers.Dense(7, activation='relu'),
    keras.layers.Dense(5, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
  ])
  # Optimizers: sgd, adam, RMSprop
  # Loss Functions: mean_squared_error, mean_squared_logarithmic_error, binary_crossentropy
  model.compile(optimizer = 'adam', learning_rate=0.05,
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return model


model = get_compiled_model()
history = model.fit(train_dataset, 
  validation_data=val_dataset,
  epochs=750, 
  steps_per_epoch=train_x.shape[0]/batch_size,
  validation_steps = val_x.shape[0]/batch_size)

# list all data in history
print(history.history.keys())

# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

test_y = ipo_test.pop('underpriced')
test_x = normalize(ipo_test)
m = test_x.shape[0]
y = test_y.astype(float).values.reshape(m,1)

preds = model.predict_classes(test_x).reshape(m,1)
print("Accuracy: "  + str(np.sum((preds == normalize(y))/m)))   
