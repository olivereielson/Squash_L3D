from pickle import FALSE
import cv2
import numpy as np

from create_CSV import generate_csv
from split_csv import split_csv_grouped
from validate_csv import validate_csv_files
import os
from VOC_Dataset import VOC, RecenterOnBall

############################################################################################################

rebuild_CSV_A = False
rebuild_CSV_B = False

train_number = 500
test_number = 75

############################################################################################################


#Open the xml files and generate the CSV files (is much faster than opening the xml files every time)
if rebuild_CSV_A:
    print("Rebuilding CSV A")
    generate_csv("/Users/olivereielson/Desktop/tennis-tracker/train/", "trainA.csv")
    generate_csv("/Users/olivereielson/Desktop/tennis-tracker/test/", "testA.csv")
    generate_csv("/Users/olivereielson/Desktop/tennis-tracker/valid/", "validA.csv")


if rebuild_CSV_B:
    print("Rebuilding CSV B")
    generate_csv("/Users/olivereielson/Desktop/test/", "AllB.csv")

split_csv_grouped("AllB.csv", train_ratio=0.75, output_train="trainB.csv", output_test="testB.csv")




assert os.path.exists("trainA.csv"), "trainA.csv does not exist"
assert os.path.exists("testA.csv"), "testA.csv does not exist"
assert os.path.exists("trainB.csv"), "trainB.csv does not exist"
assert os.path.exists("testB.csv"), "testB.csv does not exist"






# Validate the CSV files so prevent errors and data leakage
# Checks:
    # 1. Check for NULL values
    # 2. Check for required columns
    # 3. Check for valid bounding box coordinates
    # 4. Ensure all image paths are unique across splits
validate_csv_files(["trainA.csv", "testA.csv", "validA.csv"])
validate_csv_files(["trainB.csv", "testB.csv",])


transform = RecenterOnBall(crop_size=(360, 360))


# Create the dataloaders
trainA = VOC("trainA.csv", transform=transform)
testA = VOC("testA.csv", transform=transform)

trainB = VOC("trainB.csv", transform=transform)
testB = VOC("testB.csv", transform=transform)



for i in range(100):

    image, label, boxes= trainB.__getitem__(i)
    numpy_image = np.array(image)
    cv_mat = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", cv_mat)
    cv2.waitKey(0)
    print(trainB.images[i])


