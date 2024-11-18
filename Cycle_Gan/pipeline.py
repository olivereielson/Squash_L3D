import shutil
from pickle import FALSE
import cv2
import numpy as np
from tqdm import tqdm

from create_CSV import generate_csv
from split_csv import split_csv_grouped
from validate_csv import validate_csv_files
import os
from VOC_Dataset import VOC, RecenterOnBall

############################################################################################################

rebuild_CSV_A = False
rebuild_CSV_B = False

train_number = 1392
test_number = 250

tennis_path = "/Users/olivereielson/Desktop/tennis-tracker"
squash_path = "/Volumes/Capstone"
output_dir = "/Users/olivereielson/Desktop/cycleGAN"

############################################################################################################

assert os.path.exists(tennis_path), f"{tennis_path} does not exist"
assert os.path.exists(squash_path), f"{squash_path} does not exist"
assert os.path.exists(output_dir), f"{output_dir} does not exist"

if os.path.exists(output_dir):
    assert output_dir != "Capstone", "You are about to delete the Capstone folder...are you sure you want to do that?"
    assert output_dir != "tennis-tracker", "You are about to delete the tennis-tracker folder...are you sure you want to do that?"
    assert "capstone" not in output_dir.lower(), "You are about to delete the Capstone folder...are you sure you want to do that?"
    shutil.rmtree(output_dir)

os.makedirs(output_dir)
os.makedirs(f"{output_dir}/trainA")
os.makedirs(f"{output_dir}/testA")
os.makedirs(f"{output_dir}/trainB")
os.makedirs(f"{output_dir}/testB")

#Open the xml files and generate the CSV files (is much faster than opening the xml files every time)
if rebuild_CSV_A:
    print("Rebuilding CSV A")
    generate_csv("/Users/olivereielson/Desktop/tennis-tracker/train/", f"{tennis_path}/train.csv",["tennis-ball"])
    generate_csv("/Users/olivereielson/Desktop/tennis-tracker/test/", f"{tennis_path}/test.csv",["tennis-ball"])
    generate_csv("/Users/olivereielson/Desktop/tennis-tracker/valid/", f"{tennis_path}/valid.csv",["tennis-ball"])


assert os.path.exists(f"{tennis_path}/train.csv"), f"{tennis_path}/train.csv does not exist"
assert os.path.exists(f"{tennis_path}/test.csv"), f"{tennis_path}test.csv does not exist"
assert os.path.exists(f"{squash_path}/train.csv"), f"{squash_path}train.csv does not exist"
assert os.path.exists(f"{squash_path}/test.csv"), f"{squash_path}test.csv does not exist"




# Validate the CSV files so prevent errors and data leakage
# Checks:
    # 1. Check for NULL values
    # 2. Check for required columns
    # 3. Check for valid bounding box coordinates
    # 4. Ensure all image paths are unique across splits
# validate_csv_files([f"{squash_path}/train.csv", f"{squash_path}/test.csv", f"{squash_path}/valid.csv",])
# validate_csv_files([f"{tennis_path}/train.csv", f"{tennis_path}/test.csv", f"{tennis_path}/valid.csv",])


transform = RecenterOnBall(crop_size=(400, 400))


# Create the dataloaders
train_tennis = VOC(f"{tennis_path}/train.csv", transform=transform)
valid_tennis = VOC(f"{tennis_path}/valid.csv", transform=transform)

train_squash = VOC(f"{squash_path}/train.csv", transform=transform)
valid_squash = VOC(f"{squash_path}/valid.csv", transform=transform)




for i in tqdm(range(train_number), desc="Saving Images for train"):
    image, target = train_tennis.__getitem__(i)
    numpy_image = np.array(image)
    numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_dir}/trainA/{i}.jpg", numpy_image)

    image, target = train_squash.__getitem__(i)
    numpy_image = np.array(image)
    numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_dir}/trainB/{i}.jpg", numpy_image)

for i in tqdm(range(test_number), desc="Saving Images for Test"):
    image, target = valid_tennis.__getitem__(i)
    numpy_image = np.array(image)
    numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_dir}/testA/{i}.jpg", numpy_image)

    image, target = valid_squash.__getitem__(i)
    numpy_image = np.array(image)
    numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_dir}/testB/{i}.jpg", numpy_image)




