import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

from Cycle_Gan.Squash_cyclegan import load_and_preprocess_image, preprocess_image_train, preprocess_image_test

AUTOTUNE = tf.data.AUTOTUNE
image_size = [256, 256]
BUFFER_SIZE = 1000
BATCH_SIZE = 1


dataset_path = '/Users/olivereielson/Desktop/cycleGAN'
assert os.path.exists(dataset_path)


trainA_path = f"{dataset_path}/trainA/*.jpg"
trainB_path = f"{dataset_path}/trainB/*.jpg"
testA_path = f"{dataset_path}/testA/*.jpg"
testB_path = f"{dataset_path}/testB/*.jpg"

assert os.path.exists(trainA_path), "Train A path does not exist"
assert os.path.exists(trainB_path), "Train B path does not exist"
assert os.path.exists(testA_path), "Test A path does not exist"
assert os.path.exists(testB_path), "Test B path does not exist"


# Load each subset
trainA = tf.data.Dataset.list_files(trainA_path).map(lambda x: load_and_preprocess_image(x, 0), num_parallel_calls=AUTOTUNE)
trainB = tf.data.Dataset.list_files(trainB_path).map(lambda x: load_and_preprocess_image(x, 1), num_parallel_calls=AUTOTUNE)
testA = tf.data.Dataset.list_files(testA_path).map(lambda x: load_and_preprocess_image(x, 0), num_parallel_calls=AUTOTUNE)
testB = tf.data.Dataset.list_files(testB_path).map(lambda x: load_and_preprocess_image(x, 1), num_parallel_calls=AUTOTUNE)


train_horses = trainA.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

train_zebras = trainB.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_horses = testA.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_zebras = testB.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

sample_horse = next(iter(train_horses))
sample_zebra = next(iter(train_zebras))

