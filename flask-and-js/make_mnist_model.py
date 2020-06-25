import tensorflow as tf

def makeMnistModel():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (_,_) = mnist.load_data()
    X_train= X_train/255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train,epochs=5)
    model.save('./mnist_model.h5')

makeMnistModel()