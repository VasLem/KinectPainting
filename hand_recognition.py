"""
Hand Recogntion using neural networks
"""
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers

class HandRecogntionModel(object):

    def __init__(self):
        self.model = None

    def train_model(x_train, y_train):
        self.model = models.Sequential([
            layers.Flatten(),
            layers.Conv2D(5, (3, 3)),
            layers.Dropout(0.2),
            layers.GRUCell(1),
            layers.Dense(10, activation=tf.nn.softmax)
        ])
        self.is_trained = True
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        self.model.fit(x_train, y_train, epochs=5)

    def test_model(x_test, y_test):
        self.model.evaluate(x_test, y_test)

    def save_model():
        import pickle
        with open('saved_hand_model', 'w') as out:
            pickle.dump(out, self.model)
