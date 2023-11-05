import json

import tensorflow as tf
from tensorflow import keras
from keras.optimizers import SGD

from src.data import DataFacesImages


class CNNStruct:
    """
    Class to define CNN architecture in order to instantiate a CNN object.
    """
    def __init__(
            self,
            conv_pool_type: str,  # either "Conv", "ConvPool" or "ConvConvPool"
            activation_type: str,
            n_conv: int,
    ):
        # Input Layer -> Conv Layer -> MaxPool Layer -> ... -> Fully Connected Layer -> Output Layer
        self.conv_pool_type = conv_pool_type
        self.activation_type = activation_type
        self.n_conv = n_conv


class CNN:
    """
    Main class to define, build and train a CNN using TensorFlow lib through Keras API
    """
    def __init__(
            self,
            data_img: DataFacesImages = None,
            architecture: CNNStruct = None,
            network: keras.models.Sequential = None,
    ):
        self.network = network
        self.architecture = architecture
        self.data_img = data_img
        if data_img is not None:
            self.input_shape = data_img.train_generator.image_shape
            self.n_cat = data_img.train_generator.num_classes

    def build(self) -> None:
        """
        Build the CNN using Keras methods and according to given architecture.

        Returns:
            NoneType:
        """
        struc = self.architecture
        activation = struc.activation_type

        model = keras.models.Sequential()

        # We choose to double number of filters after each block of layers
        if self.architecture.conv_pool_type == "ConvConv":
            for k in range(struc.n_conv):
                n_filters = int(2 ** k) * 32
                input_shape = {}
                if k == 0:
                    input_shape = {"input_shape": self.input_shape}
                model.add(keras.layers.Conv2D(n_filters, (3, 3), **input_shape))
                model.add(keras.layers.Activation(activation))
                model.add(keras.layers.Dropout(0.25))

        if self.architecture.conv_pool_type == "ConvPool":
            for k in range(struc.n_conv):
                n_filters = int(2 ** k) * 32
                input_shape = {}
                if k == 0:
                    input_shape = {"input_shape": self.input_shape}
                model.add(keras.layers.Conv2D(n_filters, (3, 3),  **input_shape))
                model.add(keras.layers.Activation(activation))
                model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
                model.add(keras.layers.Dropout(0.25))

        if self.architecture.conv_pool_type == "ConvConvPool":
            for k in range(struc.n_conv):
                n_filters = int(2 ** k) * 32
                input_shape = {}
                if k == 0:
                    input_shape = {"input_shape": self.input_shape}
                model.add(keras.layers.Conv2D(n_filters, (3, 3),  **input_shape))
                model.add(keras.layers.Conv2D(n_filters, (3, 3)))
                model.add(keras.layers.Activation(activation))
                model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
                model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Flatten())  # to 1-D array

        model.add(keras.layers.Dense(64))
        model.add(keras.layers.Activation(activation))
        model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.Dense(self.n_cat))
        model.add(keras.layers.Activation("softmax"))

        opt = SGD(learning_rate=0.01)  # learning rate can be changed if not satisfactory results
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

        self.network = model

    def train(self, n_epoch: int = 15) -> None:
        """
        Train the CNN: find the weights to give to each neuron.

        Args:
            n_epoch (int): Number of epochs for the training phase.

        Returns:
            NoneType:
        """
        train_gen = self.data_img.train_generator
        validation_gen = self.data_img.validation_generator
        self.network.fit(train_gen, epochs=n_epoch, validation_data=validation_gen)

    def save_archi(self, save_path: str) -> None:
        """
        Save CNN architecture to `save_path`.

        Args:
            save_path (str): File path (.keras extension) where to save network's architecture.

        Returns:
            NoneType:
        """
        json_string = self.network.to_json()
        with open(save_path, "w") as json_file:
            json.dump(json_string, json_file)

    def save_weights(self, save_path: str) -> None:
        """
        Save CNN weights (use after training) to `save_path`.

        Args:
            save_path (str): File path (.h5 extension) where to save network's weights.

        Returns:
            NoneType:
        """
        self.network.save_weights(filepath=save_path)

    def load_archi(self, path: str) -> None:
        """
        Load CNN architecture from file.

        Args:
            path (str): File path (.keras extension) to network's architecture to load.

        Returns:
            NoneType:
        """
        with open(path, "r") as json_file:
            json_string = json.load(json_file)
            self.network = tf.keras.models.model_from_json(json_string)

    def load_weights(self, path: str) -> None:
        """
        Load CNN weights from file.

        Args:
            path (str): File path (.h5 extension) to network's weights to load.

        Returns:
            NoneType:
        """
        self.network.load_weights(path)

    def test(self) -> dict:
        """
        Evaluate model's performances against test dataset.

        Returns:
            dict: The loss value & metrics values for the model in test mode.
        """
        self.network.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        test_gen = self.data_img.test_generator
        return self.network.evaluate(test_gen)
