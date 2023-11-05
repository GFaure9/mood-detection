import os
import random
import shutil


def create_downsampled_datasets(input_path: str, output_path: str, sample_size: int = 150):
    """
    Down sample chosen dataset so that each sub-folder will contain a given number of files.

    Args:
        input_path (str): Path to the dataset to down sample.
        output_path (str): Path where to create the down sampled dataset.
        sample_size (int): Number files per sub-folder in the down sampled dataset.

    Returns:
        NoneType:
    """
    # Get all folders of the directory
    all_dir_items = os.listdir(input_path)
    folders = [itm for itm in all_dir_items if os.path.isdir(os.path.join(input_path, itm))]

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    for folder in folders:
        # Path to folder
        folder_path = os.path.join(input_path, folder)

        # Get a list of all image filenames in the folder
        fnames = [fname for fname in os.listdir(folder_path) if fname.endswith(".jpg")]

        # Randomly select the desired number of images
        sampled_img_fnames = random.sample(fnames, sample_size)

        # Create the folder in the output directory if it does not exist
        output_folder_path = os.path.join(output_path, folder)
        os.makedirs(output_folder_path, exist_ok=True)

        # Copy the selected images to the output directory
        for img_fname in sampled_img_fnames:
            source_path = os.path.join(folder_path, img_fname)
            destination_path = os.path.join(output_folder_path, img_fname)
            shutil.copy(source_path, destination_path)

        print(f"Copied folder {folder} at: {output_path}")


if __name__ == "__main__":
    input_dir_train = "../../datasets/train"
    output_dir_train = "../../datasets/downsample_train"
    create_downsampled_datasets(input_dir_train, output_dir_train)
