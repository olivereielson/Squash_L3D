import glob
import os.path
import random
import shutil
from PIL import Image
from tqdm import tqdm

def copy_and_resize(source, destination, size):
    image = Image.open(source)
    image = image.resize((size, size))
    image.save(destination)


dataset_path = '/Cycle_Gan/dataset'
train_number = 1000
test_number = 10
image_size = 256


squash_all = glob.glob('/Users/olivereielson/Desktop/test/*.jpg')
assert len(squash_all) != 0, "No images found in squash folder"
assert len(squash_all) > train_number + test_number, "Not enough images in squash folder"
random.shuffle(squash_all)


squash_train = squash_all[:train_number]
squash_test = squash_all[train_number:train_number + test_number]

tennis_train = glob.glob('/Users/olivereielson/Desktop/tennis-tracker/train/*.jpg')
tennis_test = glob.glob('/Users/olivereielson/Desktop/tennis-tracker/valid/*.jpg')

print(f"Tennis Train is {len(tennis_train)}")
print(f"Tennis Test is {len(tennis_test)}")
print(f"Squash train is {len(squash_train)}")
print(f"Squash test is {len(squash_test)}")

assert len(tennis_train) != 0, "No images found in train folder"
assert len(tennis_test) != 0, "No images found in test folder"
assert len(squash_train) != 0, "No images found in train folder"
assert len(squash_test) != 0, "No images found in test folder"



if os.path.exists(dataset_path):
    shutil.rmtree(dataset_path)

os.makedirs(dataset_path)
os.makedirs(f"{dataset_path}/trainA")
os.makedirs(f"{dataset_path}/testA")
os.makedirs(f"{dataset_path}/trainB")
os.makedirs(f"{dataset_path}/testB")


print("Copying and resizing training images...")
for i in range(train_number):
    copy_and_resize(tennis_train[i], f"{dataset_path}/trainA/{i}.jpg", image_size)
    copy_and_resize(squash_train[i], f"{dataset_path}/trainB/{i}.jpg", image_size)
    if i % 100 == 0:
        print(f"Processed {i} training images")

# Copy and resize testing images
print("Copying and resizing testing images...")
for i in tqdm(range(test_number), desc="Making Testing Images"):
    copy_and_resize(squash_test[i], f"{dataset_path}/testB/{i}.jpg", image_size)
    copy_and_resize(tennis_test[i], f"{dataset_path}/testA/{i}.jpg", image_size)





