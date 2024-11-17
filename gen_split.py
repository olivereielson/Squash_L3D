import glob
import json
import os
import random

ssd_path = "/Volumes/Capstone"
assert os.path.exists(ssd_path), "SSD path does not exist...idiot"


master_mappings = json.load(open(f"{ssd_path}/global_mappings.json", "r"))

total_images = []

# Shuffle the courts to ensure randomness
courts = sorted(master_mappings.keys(), key=lambda court: len(master_mappings[court]), reverse=True)
# Split the courts into training and testing


