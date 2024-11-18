import glob
import json
import os
import random
from numpy.random import shuffle

from create_CSV import generate_csv_from_path_list

max_num_images = 20000
test_size_dist = 0.10
valid_size_dist = 0.2
train_size_dist = 1 - test_size_dist - valid_size_dist

ssd_path = "/Volumes/Capstone"
assert os.path.exists(ssd_path), "SSD path does not exist...idiot"

master_mappings = json.load(open(f"{ssd_path}/global_mappings.json", "r"))

total_images = []
for key in master_mappings:
    total_images += [key]
    total_images += master_mappings[key]


train_size = int(len(total_images) * train_size_dist)
valid_size = int(len(total_images) * valid_size_dist)
test_size = int(len(total_images) * test_size_dist)


train_paths = []
valid_paths = []
test_paths = []


### Order images by Size.....MAYBE SHUFFEL THEM??
ordered_groups = list(master_mappings.values())
ordered_groups = sorted(ordered_groups, key=lambda x: len(x), reverse=False)

for group in ordered_groups:
    if len(test_paths)<test_size:
        test_paths += group
    elif len(valid_paths)<valid_size:
        valid_paths += group
    else:
        train_paths += group

#shuffle the paths before removing the excess images
shuffle(train_paths)
shuffle(valid_paths)
shuffle(test_paths)

# Calculate the number of images to remove
train_size = max_num_images * train_size_dist
valid_size = max_num_images * valid_size_dist
test_size = max_num_images * test_size_dist

# Remove the excess images
train_paths = train_paths[:int(train_size)]
valid_paths = valid_paths[:int(valid_size)]
test_paths = test_paths[:int(test_size)]



print(f"Train: {len(train_paths)}")
print(f"Valid: {len(valid_paths)}")
print(f"Test: {len(test_paths)}")

generate_csv_from_path_list(train_paths, f"{ssd_path}/train.csv")
generate_csv_from_path_list(valid_paths, f"{ssd_path}/valid.csv")
generate_csv_from_path_list(test_paths, f"{ssd_path}/test.csv")