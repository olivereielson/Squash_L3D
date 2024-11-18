import glob
import json
import warnings
import numpy as np
import cv2
import tqdm
from verify_images import are_images_similar
import os


def create_image_grid(images, grid_width=5):
    """
    Create a grid of images with a specified number of images per row.
    """
    # Determine grid dimensions
    grid_images = []
    for i in range(0, len(images), grid_width):
        row = images[i:i + grid_width]

        # Pad row with blank images if it has fewer images than grid_width
        while len(row) < grid_width:
            row.append(np.zeros((images[0].shape[0], images[0].shape[1], 3), dtype=np.uint8))  # Black image

        # Concatenate images horizontally for each row
        grid_images.append(cv2.hconcat(row))
    # Concatenate rows vertically to form the full grid
    return cv2.vconcat(grid_images)


def validate_local_mappings(folder_path,display_images=True,image_size=(360,360),silent=False):


    if not silent:
        print(f"Validating local mappings for {folder_path}")

    assert os.path.exists(folder_path), "Folder does not exist..idoit"
    local_mappings = json.load(open(f"{folder_path}/local_mappings.json", "r"))
    # Check that the local mappings is not empty
    assert len(local_mappings) > 0, "Local mappings is empty"


    #To many unique images in the folder might idicate an error
    if len(local_mappings) > 6:
        warnings.warn(f"Waring: {len(local_mappings)} unique images found in {folder_path}....Extra Validation required")

    courts = []

    # Check that all images in the local mappings exist
    for key in tqdm.tqdm(local_mappings, desc="Checking images exist", disable=silent):
        courts.append(key)
        for image in local_mappings[key]:
            assert os.path.exists(image), f"Image {image} does not exist"


    # Display the images in the local mappings
    if display_images:
        images = [cv2.imread(image) for image in courts]
        images = [cv2.resize(image, (360, 360)) for image in images]

        grid = create_image_grid(images)
        cv2.imshow("Court", grid)
        cv2.waitKey(0)


def _show_images(folder_path):
    assert os.path.exists(folder_path), "Folder does not exist..idoit"
    local_mappings = json.load(open(f"{folder_path}/local_mappings.json", "r"))


    for key in local_mappings:
        for image in local_mappings[key]:
            img = cv2.imread(image)
            cv2.imshow("Image", img)
            cv2.waitKey(1)


def gen_local_mappings(folder_path):
    print(f"Generating local mappings for {folder_path}")
    assert os.path.exists(folder_path), "Folder does not exist..idoit"
    all_images = glob.glob(f"{folder_path}/*.jpg")
    unique_images = {}
    for image in tqdm.tqdm(all_images):
        found_similar = False
        for unique_image in unique_images:
            if are_images_similar(image, unique_image, threshold=0.80):
                found_similar = True
                unique_images[unique_image].append(image)
                break

        # If no similar image was found, add it to the unique images
        if not found_similar:
            unique_images[image] = [image]


    json.dump(unique_images, open(f"{folder_path}/local_mappings.json", "w"))
    assert os.path.exists(f"{folder_path}/local_mappings.json"), "Failed to generate local mappings"
    print(f"Generated local mappings for {folder_path}, {len(unique_images)} unique images found")

    if len(unique_images) == 0:
        warnings.warn(f"No unique images found in {folder_path}")

    elif len(unique_images) > 10:
        warnings.warn(f"Waring: {len(unique_images)} unique images found in {folder_path}....Extra Validation required")


if __name__ == "__main__":
    ssd_path = "/Volumes/Capstone"
    assert os.path.exists(ssd_path), "SSD path does not exist...idiot"
    #loop through all the folders in the data folder to generate local mappings
    for folder in glob.glob(f"{ssd_path}/*"):
        if os.path.isdir(folder) and not folder == "/Volumes/Capstone/System Volume Information":
            gen_local_mappings(folder)
            validate_local_mappings(folder,display_images=False,image_size=(360,360))
            # _show_images(folder)