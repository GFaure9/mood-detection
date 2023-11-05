import tensorflow as tf


class DataFacesImages:
    """
    Main class to build training and test datasets from faces images database.
    """
    def __init__(
            self,
            path_to_train: str = None,
            path_to_test: str = None,
            train_generator: tf.keras.preprocessing.image.DirectoryIterator = None,
            validation_generator: tf.keras.preprocessing.image.DirectoryIterator = None,
            test_generator: tf.keras.preprocessing.image.DirectoryIterator = None,
    ):
        self.path_to_train = path_to_train
        self.path_to_test = path_to_test
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.test_generator = test_generator

    def train_prepro(self, target_size: tuple = (224, 224), batch_size: int = 32, val_split=0.2) -> None:
        """
        Prepare DirectoryIterator for training and validation datasets.

        Args:
            target_size (Tuple): Target size for resizing images.
            batch_size (int): Size of the batches to separate the data.
            val_split (float): Ratio of the dataset that will be used for validation.

        Returns:
            NoneType:
        """
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0/255.0,  # Normalize pixel values to [0, 1]
            rotation_range=20,  # Randomly rotate images by up to 20 degrees
            width_shift_range=0.2,  # Randomly shift image width by up to 20%
            height_shift_range=0.2,  # Randomly shift image height by up to 20%
            horizontal_flip=True,  # Randomly flip images horizontally
            validation_split=val_split,  # Set val_split ratio of the data as validation
        )

        train_gen = datagen.flow_from_directory(
            self.path_to_train,
            target_size=target_size,  # resize images to target size
            batch_size=batch_size,
            class_mode="categorical",  # for classification tasks with one-hot encoding,
            subset="training"  # set as training data
        )

        validation_gen = datagen.flow_from_directory(
            self.path_to_train,
            target_size=target_size,  # resize images to target size
            batch_size=batch_size,
            class_mode="categorical",  # for classification tasks with one-hot encoding,
            subset="validation"  # set as validation data
        )

        self.train_generator = train_gen
        self.validation_generator = validation_gen

    def test_prepro(self, target_size: tuple = (224, 224), batch_size: int = 32) -> None:
        """
        Prepare DirectoryIterator for test datasets.

        Args:
            target_size (Tuple): Target size for resizing images.
            batch_size (int): Size of the batches to separate the data.

        Returns:
            NoneType:
        """
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
        )

        test_gen = datagen.flow_from_directory(
            self.path_to_test,
            target_size=target_size,
            batch_size=batch_size,
        )

        self.test_generator = test_gen

    # todo: add a method to retrieve x_train, y_train, x_test, y_test from generator


