# Building CNNs for __Facial Expression Recognition__

---

This repository contains a project with the aim of building a _simple_ CNN to classify faces images
into 7 expressions categories: "angry", "disgust", "fear", "happy", "sad", "surprise and "neutral".\
It uses as train/test data the following [Kaggle.com](https://www.kaggle.com/) dataset: [Facial Emotion Expressions](https://www.kaggle.com/datasets/samaneheslamifar/facial-emotion-expressions).

---

This project is a work in progress and is not yet completed.\
However, it provides a basis Python framework to prepare Kaggle's downloaded data and to customize, build, train and 
test Convolutional Neural Networks to perform human facial expression classification on images.

Feel free to fork it and make your own changes/improvements (and eventually create a pull request).

---
## Prerequisites/Dependencies

- Python 3.10 or higher (this is what I used but probably older versions also work, typically >=3.8)

List of needed dependencies:
- TensorFlow (for all the DL pipelines)
- OpenCV (to modify/visualize images)
- Matplotlib (not mandatory, to plot curves...)

## Installation

Clone the repo:

```commandline
git clone https://github.com/GFaure9/mood-detection.git
```

And after having created clean virtual environment in project's folder:

```commandline
pip install -r requirements.txt
```

> [!NOTE]
> Original dataset from [https://www.kaggle.com/samaneheslamifar](https://www.kaggle.com/samaneheslamifar).
> You can download it directly at: [https://www.kaggle.com/datasets/samaneheslamifar/facial-emotion-expressions](https://www.kaggle.com/datasets/samaneheslamifar/facial-emotion-expressions).

---
## About the project...

Some insights about implemented objects (in [src](./src)):
- `CNN`: main class to define, build and train a CNN using TensorFlow lib through Keras API
- `DataFacesImages`: main class to build training and test datasets from faces images database
- `Pipeline`: class defining pipelines to run for building, training and testing face expression classification models.
- `create_downsampled_datasets(_input_path, output_path sample_size_)`: function that creates a new folder from a chosen folder, with the same sub-folders but keeping only a given number of files in each sub-folder.

In [exploration][./exploration] a Jupyter Notebook was started to start testing and using these objects and
find the best configuration for the CNN.