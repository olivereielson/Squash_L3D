import glob

import cv2
import numpy as np
import os


def are_images_similar(img1_path, img2_path, threshold=0.6):
    # Read images in grayscale
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if not img1.shape == img2.shape:
        return False

    diff = cv2.absdiff(img1, img2)

    # Apply threshold to focus on significant pixel changes
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)  # Adjust the threshold value as needed

    # Count non-zero pixels (pixels that are significantly different)
    significant_pixels = cv2.countNonZero(thresh)

    # Calculate the percentage of different pixels
    total_pixels = thresh.size
    percentage_diff = (significant_pixels / total_pixels)
    # print(f"Percentage difference: {percentage_diff}")

    return (1-percentage_diff) > threshold



if __name__ == "__main__":
    folder = "/Volumes/Capstone/2021-04-24"
    files = glob.glob(f"{folder}/*.jpg")

    for path1,path2 in zip(files, files[1:]):

        if not are_images_similar(f"{path1}", f"{path2}", threshold=0.85):
            print(f"Images {path1} and {path2} are not similar")
            cv2.imshow("Image 1", cv2.imread(path1))
            cv2.imshow("Image 2", cv2.imread(path2))
            cv2.waitKey(1)
