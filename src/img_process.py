import cv2 as cv
import matplotlib.pyplot as plt

from typing import Tuple


class ImageProcessor:
    """
    Main class to help with image processing basic operations.
    """
    def __init__(self, path: str = None, image: cv.typing.MatLike = None):
        self.path = path
        if image is None and path is not None:
            image = self.load()
        self.image = image

    def load(self, path: str = None) -> cv.typing.MatLike:
        """
        Load the image using OpenCV.

        Args:
            path (str): Path to the image file.

        Returns:
            cv.typing.MatLike: The loaded image.
        """
        if path is None:
            path = self.path
        img = cv.imread(path)
        return img

    def show(self):
        """
        Show the image.

        Returns:
            NoneType:
        """
        img = self.image
        if img is None:
            raise ValueError(
                "You must provide an image or path to an image first."
            )
        else:
            image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            plt.imshow(image_rgb)
            plt.axis("off")  # turn off axis labels and ticks
            plt.show()

    def shape(self) -> Tuple:
        """
        Return the shape of the loaded image.

        Returns:
            Tuple: The shape of the image.
        """
        img = self.image
        if img is None:
            raise ValueError(
                "You must provide an image or path to an image first."
            )
        return img.shape


if __name__ == "__main__":
    pth = "../datasets/train/angry/0.jpg"
    processor = ImageProcessor(path=pth)
    print("Shape: ", processor.shape())
    processor.show()
