import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(X_train, y_train), (_,_) = mnist.load_data()
print(X_train[1])
X_train= X_train/255.0