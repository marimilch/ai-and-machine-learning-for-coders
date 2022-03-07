import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Callback to stop on >95% accuracy
class AccuracyAtLeast95(tf.keras.callbacks.Callback): # inherit from tf.keras.callbacks.Callback
  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('accuracy')>.95):
      print('\n☺️ Accuracy above 95%. Training will be stopped now.')
      self.model.stop_training = True

callbacks = AccuracyAtLeast95()

# Get MNIST fashion data set
data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

# Normalize images
training_images = training_images / 255.0
test_images = test_images / 255.0

# 2D Array to 1D Array then 2-layer dense network
model = Sequential([
  Flatten(input_shape=(28, 28)),
  Dense(128, activation=tf.nn.relu), # activation: rectified linear unit -> y = max(0, x)
  Dense(10, activation=tf.nn.softmax), # activation: softmax, typical for last layers
])

model.compile(
  optimizer='adam', # adam is an evolution to stochastic gradient optimizer
  loss='sparse_categorical_crossentropy', # good for categorization
  metrics=['accuracy']
)

# Training
model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

# Testing
model.evaluate(test_images, test_labels)

# Prediction
# classifications = model.predict(test_images)

# print(classifications[0])
# print(test_labels[0])


