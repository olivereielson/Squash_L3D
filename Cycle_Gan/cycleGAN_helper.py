import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

def load_and_preprocess_image(file_path, image_size=None):
    if image_size is None:
        image_size = [256, 256]
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Adjust if using PNG
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image

def load_dataset(path):
    dataset = tf.data.Dataset.list_files(path)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset