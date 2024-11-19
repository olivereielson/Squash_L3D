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

class RecenterOnBall:
    def __init__(self, crop_size):
        """
        Args:
            crop_size (tuple): Desired output size of the crop (height, width).
        """
        self.crop_size = crop_size

    def __call__(self, image, box_tensor, label):
        """
        Args:
            image (PIL Image): The input image.
            box_tensor (list or torch.Tensor): Bounding box coordinates of the ball [xmin, ymin, xmax, ymax].
            label: Additional label information.

        Returns:
            PIL Image: Cropped and centered image around the ball, with padding if necessary.
        """
        # Convert box coordinates to integers if they're torch.Tensors
        if isinstance(box_tensor, torch.Tensor):
            box = box_tensor.int().tolist()
        else:
            box = box_tensor

        # Extract the bounding box coordinates
        xmin, ymin, xmax, ymax = box

        # Calculate the center of the ball
        ball_center_x = (xmin + xmax) // 2
        ball_center_y = (ymin + ymax) // 2

        # Calculate the crop dimensions
        crop_width, crop_height = self.crop_size

        # Determine the required padding to ensure the ball is centered in the crop
        pad_left = max(0, crop_width // 2 - ball_center_x)
        pad_top = max(0, crop_height // 2 - ball_center_y)
        pad_right = max(0, ball_center_x + crop_width // 2 - image.width)
        pad_bottom = max(0, ball_center_y + crop_height // 2 - image.height)

        # Pad the image if necessary
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            image = ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)

        # Update the ball center coordinates after padding
        ball_center_x += pad_left
        ball_center_y += pad_top

        # Calculate the crop region
        left = ball_center_x - crop_width // 2
        top = ball_center_y - crop_height // 2
        right = left + crop_width
        bottom = top + crop_height

        # Crop the image
        image = F.crop(image, top, left, crop_height, crop_width)

        return image, box_tensor, label


label_map = {
    "Squash": 1,
    "tennis-ball": 0
}


class VOC(Dataset):

    def __init__(self, CSV, transform=None):
        self.path = CSV  # Path to the folder containing the images

        self.labels = []
        self.images = []
        self.boxes = []

        self.transform = transform

        self._readCSV()

    def _readCSV(self):
        assert os.path.exists(self.path), f"CSV {self.path} does not exist"
        # Load the CSV file into a DataFrame
        df = pd.read_csv(self.path)

        # Iterate through each row and populate images, labels, and boxes
        for _, row in df.iterrows():

            if row['class']!="Squash" and row['class']!="tennis-ball":
                continue

            image_path = row['image_path']

            # Store the image and associated data
            self.images.append(image_path)
            self.labels.append(row['class'])
            self.boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        labels = [self.labels[idx]]
        boxes = [self.boxes[idx]]

        # Convert labels to integers
        labels = [label_map[l] for l in labels]

        image = Image.open(image).convert("RGB")

        boxes = torch.as_tensor(boxes, dtype=torch.int)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.zeros((len(labels)), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        image_id = torch.tensor([idx])

        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes[0], labels[0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target
