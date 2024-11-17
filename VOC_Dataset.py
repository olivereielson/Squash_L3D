import warnings

import torch
from PIL import Image
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

    def __call__(self, image, box):
        """
        Args:
            image (PIL Image): The input image.
            box (list): Bounding box coordinates of the ball [xmin, ymin, xmax, ymax].

        Returns:
            PIL Image: Cropped and centered image around the ball.
        """
        # Extract the bounding box coordinates
        xmin, ymin, xmax, ymax = box

        # Calculate the center of the ball
        ball_center_x = (xmin + xmax) // 2
        ball_center_y = (ymin + ymax) // 2

        # Calculate the crop region
        crop_width, crop_height = self.crop_size
        left = max(ball_center_x - crop_width // 2, 0)
        top = max(ball_center_y - crop_height // 2, 0)
        right = min(left + crop_width, image.width)
        bottom = min(top + crop_height, image.height)

        # Adjust the crop if it goes outside the image boundaries
        if right > image.width:
            left = max(image.width - crop_width, 0)
            right = image.width
        if bottom > image.height:
            top = max(image.height - crop_height, 0)
            bottom = image.height

        # Crop the image
        image = F.crop(image, top, left, bottom - top, right - left)

        return image



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
            image, boxes, labels = self.transform(image, boxes, labels)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target
