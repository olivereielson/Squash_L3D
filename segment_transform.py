import warnings

import cv2
import numpy as np
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
        if labels.tolist() == label_map["tennis-ball"]:
            new_image = self.court_color(image)
            # Apply line transformation
            new_image = self.change_lines(image, new_image)
            # Apply ball transformation for each bounding box
            # for box, label in zip(boxes.tolist(), labels.tolist()):

            new_image = self.change_ball(image, new_image, boxes.tolist())
            return new_image, boxes, labels
        else:
            return image, boxes, labels

label_map = {
    "Squash": 1,
    "tennis-ball": 0
}