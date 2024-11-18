import glob
import os
import json

import cv2
import numpy as np
from tqdm import tqdm

from local_mappings import validate_local_mappings
from verify_images import are_images_similar



def sanity_check(mappings,size=(360,360)):
    for key in mappings:
        key_image = cv2.imread(key)
        assert key_image is not None, f"Image {key} is empty"
        key_image = cv2.resize(key_image, size)

        for image in mappings[key]:
            assert os.path.exists(image), f"Image {image} does not exist"
            frame = cv2.imread(image)
            assert frame is not None, f"Image {image} is empty"
            frame = cv2.resize(frame, size)
            side_by_side = np.hstack((key_image, frame))
            cv2.imshow(key, side_by_side)
            cv2.waitKey(1)



def summary(global_mappings):
    print(f"Total unique images: {len(global_mappings)}")
    total_images = 0
    for key in global_mappings:
        total_images += len(global_mappings[key])
    print(f"Total images: {total_images}")


    for key in global_mappings:
        print(f"Unique Image: {key}")
        print(f"Total Images: {len(global_mappings[key])}")




ssd_path = "/Volumes/Capstone"
assert os.path.exists(ssd_path), "SSD path does not exist...idiot"

all_folders = glob.glob(f"{ssd_path}/*")


global_unique_images = {}


for folder in tqdm(all_folders, desc="Merging Local Mappings"):
    if os.path.isdir(folder) and not folder == "/Volumes/Capstone/System Volume Information":
        validate_local_mappings(folder, display_images=False,silent=True)
        local_mappings = json.load(open(f"{folder}/local_mappings.json", "r"))


        for key in local_mappings:
            found_similar = False

            if len(local_mappings[key])<100:
                continue

            for global_key in global_unique_images:
                if are_images_similar(key, global_key, threshold=0.90):

                    found_similar = True
                    global_unique_images[global_key].extend(local_mappings[key])
                    break

            if not found_similar:
                global_unique_images[key] = local_mappings[key]



json.dump(global_unique_images, open(f"{ssd_path}/global_mappings.json", "w"))
assert os.path.exists(f"{ssd_path}/global_mappings.json"), "Failed to generate global mappings"
summary(global_unique_images)
sanity_check(global_unique_images)