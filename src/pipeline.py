from src.cnn import CNN, CNNStruct
from src.data import DataFacesImages


class Pipeline:
    """
    Class defining pipelines to run for building, training and testing face expression classification models.
    """
    def __init__(
            self,
            # Data pre-processing parameters
            path_to_train: str,
            path_to_test: str,
            batch_size: int,

            # CNN structure
            activation_type: str,
            conv_pool_type: str,
            n_conv: int,

            # Training parameters
            n_epoch: int,

            # Filepath to save CNN architecture
            archi_save_path: str,

            # Filepath to save CNN weights
            weights_save_path: str,

            # CNN instance
            model: CNN = None
    ):
        self.path_to_train = path_to_train
        self.path_to_test = path_to_test
        self.batch_size = batch_size
        self.activation_type = activation_type
        self.conv_pool_type = conv_pool_type
        self.n_conv = n_conv
        self.n_epoch = n_epoch
        self.archi_save_path = archi_save_path
        self.weights_save_path = weights_save_path
        self.model = model

    def run_train(self) -> None:
        """
        Run the full training pipeline (from image preprocessing to CNN construction and training).

        Returns:
             NoneType:
        """
        print("\nStarting training pipeline...")

        # #################### Pre-process images ####################
        data_image = DataFacesImages(path_to_train=self.path_to_train)
        data_image.train_prepro(batch_size=self.batch_size)
        print("Dataset pre-processed")

        # ################### Define CNN structure ###################
        cnn_structure = CNNStruct(
            activation_type=self.activation_type,
            conv_pool_type=self.conv_pool_type,
            n_conv=self.n_conv
        )
        print("CNN structure created")

        # ################### Build and train CNN ###################
        cnn = CNN(architecture=cnn_structure, data_img=data_image)
        cnn.build()
        print("CNN built")
        cnn.train(n_epoch=self.n_epoch)
        print("CNN trained")

        # ################## Save CNN architecture ##################
        cnn.save_archi(save_path=self.archi_save_path)
        print(f"CNN architecture saved at: {self.archi_save_path}")

        # ##################### Save CNN weights #####################
        cnn.save_weights(save_path=self.weights_save_path)
        print(f"CNN weights saved at: {self.weights_save_path}")

        self.model = cnn

    def trained_model_from_file(self) -> None:
        """
        Load already trained model from its architecture and its weights (saved in .json and .h5
        files respectively).

        Returns:
             NoneType:
        """
        # ################# Load CNN architecture #################
        cnn = CNN()
        cnn.load_archi(self.archi_save_path)
        print(f"CNN built from architecture loaded at: {self.archi_save_path}")

        # ################### Load CNN weights ###################
        cnn.load_weights(self.weights_save_path)
        print(f"CNN weights loaded from file: {self.weights_save_path}")

        self.model = cnn

    def run_test(self):
        print("\nStarting testing pipeline...")

        # #################### Pre-process images ####################
        data_image = DataFacesImages(path_to_test=self.path_to_test)
        data_image.test_prepro(batch_size=self.batch_size)
        self.model.data_img = data_image
        print("Dataset pre-processed")

        # ###################### Test the model ######################
        self.model.test()
