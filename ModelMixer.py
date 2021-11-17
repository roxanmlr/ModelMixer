import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from interfaceModelMixer import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import metrics

from tensorflow import keras
from tensorflow.keras import layers


class MyModel:
    """"
    MyModel est un modele lineaire de regression
    """

    def __init__(self,*,dataset : pd.DataFrame,features : [str],label : str):
        self.features = features
        train_dataset = dataset.sample(frac=0.7, random_state=0)
        test_dataset = dataset.drop(train_dataset.index)

        train_dataset = train_dataset.copy()
        test_dataset = test_dataset.copy()

        train_labels = train_dataset.pop(label)
        test_labels = test_dataset.pop(label)

        train_features = train_dataset[features]
        test_features = test_dataset[features]

        self.model = tf.keras.models.Sequential()
        self.model.add(layers.Input(shape=(len(features),)))
        self.output_layer = layers.Dense(units=1)
        self.model.add(self.output_layer)

        self.model.compile(loss='mean_absolute_error', optimizer=tf.optimizers.Adam(learning_rate=0.1),
                     metrics=[metrics.mean_squared_error,
                              metrics.mean_absolute_error,
                              metrics.mean_absolute_percentage_error])
        self.model.fit(
            train_features,
            train_labels,
            epochs=100,
            verbose=0,
            validation_split=0.2)




    def compute(self, dataset: pd.DataFrame):
        return self.model.predict(dataset[self.features])


if __name__ == '__main__':
    n = 30
    x = np.random.randint(low=0, high=100, size=n)
    y = np.random.randint(low=0, high=75, size=n)
    res = 2 * x + y

    dataset = pd.DataFrame({'x': x, 'y': y, 'res': res})

    myModel = MyModel(dataset=dataset,features=['x','y'],label='res')
    print(np.array(myModel.output_layer.kernel))