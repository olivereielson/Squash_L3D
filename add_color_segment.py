import glob
import os

import pandas as pd

from create_CSV import generate_csv_from_path_list

ssd_path = "/Volumes/Capstone"
train_path="/Volumes/Capstone/train.csv"
tennis_path="/Volumes/Capstone/tennis"

assert os.path.exists(ssd_path), "SSD path does not exist...idiot"
assert os.path.exists(train_path), f"Train path {train_path} does not exist"
assert os.path.exists(tennis_path), f"Tennis path {tennis_path} does not exist"



tennis_images = glob.glob(f"{tennis_path}/*.jpg")
generate_csv_from_path_list(tennis_images, f"{tennis_path}/tennis_path.csv")


csv1 = pd.read_csv(f"{ssd_path}/train.csv")
csv2 = pd.read_csv(f"{tennis_path}/tennis_path.csv")
csv2 = csv2[csv2['class'] == 'tennis-ball']
# print(csv2.shape)

# Combine the two DataFrames by appending rows
merged_csv = pd.concat([csv1, csv2], ignore_index=True)
merged_csv.to_csv(f"{ssd_path}/train+segmented.csv", index=False)