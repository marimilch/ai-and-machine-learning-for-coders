import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Callback to stop on >95% accuracy
class AccuracyAtLeast95(tf.keras.callbacks.Callback): # inherit from tf.keras.callbacks.Callback
  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('accuracy')>.95):
      print('\n☺️  Accuracy above 95%. Training will be stopped now.')
      self.model.stop_training = True

callbacks = AccuracyAtLeast95()

# Get MNIST fashion data set
data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

# Normalize images
training_images = training_images / 255.0
test_images = test_images / 255.0

# Add single dimension for convolution
training_images = training_images.reshape(len(training_images), 28, 28, 1)
test_images = test_images.reshape(len(test_images), 28, 28, 1)

# 2D Array to 1D Array then 2-layer dense network
model = Sequential([
  Conv2D(64, (3,3), input_shape=(28,28,1), activation="relu"), # 64 "Conv-Neurons" finding each a matching 3x3 filter for 28x28 pic (monochrome)
  MaxPooling2D(2, 2), # Reduce 2x2 groups in the pic to 1x1 with the value set to the highest in the 4x4-group
  Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation="relu"), # Additional hidden convolutional layer
  MaxPooling2D(2, 2), # additional hidden pooling layer
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


