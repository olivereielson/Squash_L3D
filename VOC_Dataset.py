import warnings

import cv2
import numpy as np
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

class CourtTransform:
    def __init__(self, wood_color=[255, 193, 140], line_color=[255, 0, 0], ball_color=[0, 0, 0]):
        self.wood_color = wood_color
        self.line_color = line_color
        self.ball_color = ball_color

    def court_color(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height, width, _ = image.shape

        a, b, c = [], [], []
        for j in range(-20, 20):
            x, y = (width // 2) + j, (height // 2) + j
            selected_color = hsv_image[y, x]
            a.append(selected_color[0])
            b.append(selected_color[1])
            c.append(selected_color[2])

        a_med, b_med, c_med = np.median(a), np.median(b), np.median(c)
        selected_color = np.array([a_med, b_med, c_med]).astype(int)

        tolerance = 60
        lower_bound = np.array([
            max(0, selected_color[0] - tolerance),
            max(0, selected_color[1] - tolerance),
            max(0, selected_color[2] - tolerance)
        ], dtype=np.uint8)

        upper_bound = np.array([
            min(180, selected_color[0] + tolerance),
            min(255, selected_color[1] + tolerance),
            min(255, selected_color[2] + tolerance)
        ], dtype=np.uint8)

        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        new_image = image.copy()
        new_image[mask > 0] = self.wood_color
        return new_image

    def change_lines(self, og_image, new_image):
        hsv_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 120, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv_image, lower_white, upper_white)
        new_image[mask > 0] = self.line_color
        return new_image

    def change_ball(self, og_image, new_image, bnd_box):
        xmin, ymin, xmax, ymax = bnd_box
        roi = og_image[ymin:ymax, xmin:xmax]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        height, width, _ = roi.shape

        x, y = width // 2, height // 2
        selected_color = hsv_roi[y, x]
        selected_color = selected_color.astype(int)
        tolerance = 30

        lower_bound = np.array([
            max(0, selected_color[0] - tolerance),
            max(0, selected_color[1] - tolerance),
            max(0, selected_color[2] - tolerance)
        ], dtype=np.uint8)

        upper_bound = np.array([
            min(180, selected_color[0] + tolerance),
            min(255, selected_color[1] + tolerance),
            min(255, selected_color[2] + tolerance)
        ], dtype=np.uint8)

        mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
        roi[mask > 0] = self.ball_color
        roi[mask == 0] = self.wood_color
        new_image[ymin:ymax, xmin:xmax] = roi
        return new_image

    def __call__(self, image, boxes, labels):

        # print("Boxes:", boxes.tolist())
        # print("Labels:", labels.tolist())
        image = np.array(image)
        # Apply court color transformation
        new_image = self.court_color(image)
        # Apply line transformation
        new_image = self.change_lines(image, new_image)
        # Apply ball transformation for each bounding box
        # for box, label in zip(boxes.tolist(), labels.tolist()):
        if labels.tolist() == label_map["tennis-ball"]:
            new_image = self.change_ball(image, new_image, boxes.tolist())
        return new_image, boxes, labels


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
            box (list): Bounding box coordinates of the ball [xmin, ymin, xmax, ymax].

        Returns:
            PIL Image: Cropped and centered image around the ball.
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

        return image, box_tensor, label



label_map = {
    "Squash": 1,
    "tennis-ball": 0
}


class VOC(Dataset):

    def __init__(self, CSV, transform=None):
        self.path = CSV  # Path to the folder containing the images

        print(CSV)
        print()
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
            # print(image_path)

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


