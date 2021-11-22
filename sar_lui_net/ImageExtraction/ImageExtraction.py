import tifffile as tiff
import cv2
import os

# from here every tif will be split into its dimensions
tifPath = '../fuego/'


def extract_image(path, filename):
    img = tiff.imread(os.path.join(path, filename))

    pure_file_name = filename[: -len('.tif')]  # without tif
    path_for_extracted_files = os.path.join(path, pure_file_name)

    os.makedirs(path_for_extracted_files, exist_ok=True)

    # split into dimension
    for dimension in range(img.shape[0]):
        extracted_file_name = os.path.join(path_for_extracted_files, str(dimension) + '.jpg')
        print(extracted_file_name)
        cv2.imwrite(extracted_file_name, img[dimension, :, :])


def walk_images():
    """ walk over all tifs in the given directory"""
    for root, dirs, files in os.walk(tifPath):
        for name in files:
            if name.endswith('.tif'):
                extract_image(root, name)


if __name__ == "__main__":
    walk_images()
