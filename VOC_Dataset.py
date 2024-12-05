import warnings

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import os
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm
import pandas as pd
from torchvision.transforms import functional as F
from torchvision.io import read_image
from torchvision.transforms import v2 as T
import random


def resize(image, boxes, size):
    # Resize image
    image = F.resize(image, size)

    # Get the original and new dimensions
    orig_width, orig_height = image.size
    new_width, new_height = size

    # Compute scaling factors
    scale_width = new_width / orig_width
    scale_height = new_height / orig_height

    # Adjust bounding boxes
    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_width  # x_min, x_max
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_height  # y_min, y_max

    return image, boxes


label_map = {
    "Squash": 1,
    "tennis-ball": 0
}


class ResizeWithBBoxes:
    def __init__(self, target_height=640):
        """
        Initializes the resize transformation.

        Args:
            target_height (int): The target height. The width will be scaled proportionally.
        """
        self.target_height = target_height

    def __call__(self, image, boxes, labels):
        """
        Applies resizing to the image while maintaining aspect ratio
        and adjusts bounding boxes accordingly.

        Args:
            image (PIL.Image or Tensor): Input image.
            boxes (Tensor): Bounding boxes in (xmin, ymin, xmax, ymax) format, absolute coordinates.
            labels (Tensor): Corresponding labels for the bounding boxes.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Resized image, resized bounding boxes, and unchanged labels.
        """
        # Extract original image dimensions
        original_width, original_height = image.size if isinstance(image, Image.Image) else (
        image.shape[-1], image.shape[-2])

        # Calculate the new width while maintaining the aspect ratio
        scale = self.target_height / original_height
        target_width = int(original_width * scale)

        # Resize the image
        resized_image = F.resize(image, size=(self.target_height, target_width))

        # Adjust bounding boxes
        resized_boxes = boxes.clone()
        resized_boxes[:, [0, 2]] *= scale  # Scale xmin and xmax
        resized_boxes[:, [1, 3]] *= scale  # Scale ymin and ymax

        return resized_image, resized_boxes, labels


class FlipWithBBoxes:
    def __init__(self, flip_type="horizontal", probability=1.0):
        """
        Initializes the flip transformation.

        Args:
            flip_type (str): Type of flip to apply.
                             Options: "horizontal", "vertical", "both".
            probability (float): Probability of applying the transformation. Should be between 0 and 1.
        """
        if flip_type not in ["horizontal", "vertical", "both"]:
            raise ValueError("flip_type must be one of 'horizontal', 'vertical', or 'both'")
        if not (0.0 <= probability <= 1.0):
            raise ValueError("probability must be between 0 and 1")
        self.flip_type = flip_type
        self.probability = probability

    def __call__(self, image, boxes, labels):
        """
        Applies flipping to the image and adjusts bounding boxes accordingly if probability condition is met.

        Args:
            image (PIL.Image or Tensor): Input image.
            boxes (Tensor): Bounding boxes in (xmin, ymin, xmax, ymax) format, absolute coordinates.
            labels (Tensor): Corresponding labels for the bounding boxes.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Transformed image, adjusted bounding boxes, and unchanged labels.
        """
        # Apply the transformation with the specified probability
        if random.random() > self.probability:
            return image, boxes, labels  # Return as is if not applying the transformation

        # Extract original image dimensions
        original_width, original_height = image.size if isinstance(image, Image.Image) else (
        image.shape[-1], image.shape[-2])

        flipped_image = image
        flipped_boxes = boxes.clone()

        # Horizontal flip
        if self.flip_type in ["horizontal", "both"]:
            if isinstance(image, Image.Image):
                flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                flipped_image = image.flip(-1)  # Flip along width dimension

            # Adjust bounding boxes for horizontal flip
            flipped_boxes[:, [0, 2]] = original_width - boxes[:, [2, 0]]

        # Vertical flip
        if self.flip_type in ["vertical", "both"]:
            if isinstance(image, Image.Image):
                flipped_image = flipped_image.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                flipped_image = flipped_image.flip(-2)  # Flip along height dimension

            # Adjust bounding boxes for vertical flip
            flipped_boxes[:, [1, 3]] = original_height - boxes[:, [3, 1]]

        return flipped_image, flipped_boxes, labels


class VOC(Dataset):

    def __init__(self, CSV, data_dir, transform=None):
        self.path = CSV  # Path to the folder containing the images

        self.labels = []
        self.images = []
        self.boxes = []
        self.data_dir = data_dir

        self.transform = transform

        self.removed_images = []

        self._readCSV()

    def _readCSV(self):
        assert os.path.exists(self.path), f"CSV {self.path} does not exist"
        # Load the CSV file into a DataFrame
        df = pd.read_csv(self.path)

        # Iterate through each row and populate images, labels, and boxes
        for index, row in df.iterrows():
            if row['class'] != "Squash" and row['class'] != "tennis-ball":
                self.removed_images.append(row['image_path'])
                continue

            # Check for swaped xmins and xmax
            if row['xmin'] > row['xmax']:
                self.removed_images.append(row['image_path'])

            # check for swaped ymin and yamx
            if row['ymin'] > row['ymax']:
                self.removed_images.append(row['image_path'])

            # Check for negative bounding box values
            if row['xmin'] < 0 or row['ymin'] < 0 or row['xmax'] < 0 or row['ymax'] < 0:
                self.removed_images.append(row['image_path'])
                continue

            # check for too small bounding box
            width = row['xmax'] - row['xmin']
            height = row['ymax'] - row['ymin']
            if width < 3 or height < 3:
                self.removed_images.append(row['image_path'])
                continue

            image_path = row['image_path']
            print(image_path)

            # Store the image and associated data
            self.images.append(image_path)
            self.labels.append(row['class'])
            self.boxes.append([row['xmin']-5, row['ymin']-5, row['xmax']+5, row['ymax']+5])

        # print(f"When reading in data {len(self.removed_images)} images were removed")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.data_dir + "/" + self.images[idx]
        labels = [self.labels[idx]]
        boxes = [self.boxes[idx]]
        print("GET ITEM:", image)
        print(labels)
        # Convert labels to integers
        labels = [label_map[l] for l in labels]

        image = Image.open(image).convert("RGB")

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])

        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)

        iscrowd = torch.zeros((len(labels)), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target