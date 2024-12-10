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
    def __init__(self, wood_color="/cluster/home/yezzo01/Squash_L3D/squash_floor.jpg", line_color=[255, 0, 0], ball_color=[0, 0, 0]):

        self.wood_color = cv2.imread(wood_color)
        self.wood_color = cv2.cvtColor(self.wood_color, cv2.COLOR_RGB2BGR)
        self.line_color = line_color
        self.ball_color = ball_color

    def court_color(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
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

        resized_texture = cv2.resize(self.wood_color, (width, height), interpolation=cv2.INTER_AREA)
        return_img= resized_texture.copy()

        texture_overlay = np.zeros_like(image)
        texture_overlay[mask > 0] = resized_texture[mask > 0]

        new_image = image.copy()
        # new_image[mask > 0] = self.wood_color
        new_image[mask > 0] = texture_overlay[mask > 0]
        return return_img


    def change_lines(self, og_image, new_image):
        hsv_image = cv2.cvtColor(og_image, cv2.COLOR_RGB2HSV)
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 120, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv_image, lower_white, upper_white)
        new_image[mask > 0] = self.line_color
        return new_image

    def change_ball(self, og_image, new_image, bnd_box):
        # print(bnd_box)
        xmin, ymin, xmax, ymax = map(int, bnd_box)
        if xmin < 0 or ymin < 0 or xmax > og_image.shape[1] or ymax > og_image.shape[0]:
            raise ValueError("Bounding box coordinates are out of bounds.")

        roi = og_image[ymin:ymax, xmin:xmax]
        if roi is None or roi.size == 0:
            raise ValueError("Extracted ROI is empty. Check bounding box or image data.")

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        # height, width, _ = roi.shape

        # x, y = width // 2, height // 2
        center_x = xmin + (xmax - xmin) // 2
        center_y = ymin + (ymax - ymin) // 2

        radius = max(int(min(xmax - xmin, ymax - ymin) * 0.50), 1)
        cv2.circle(new_image, (center_x, center_y), radius, (0, 0, 0), -1, lineType=cv2.LINE_4)  # -1 fills the circle

        roi_size = 10
        height, width, _ = new_image.shape

        x_start = max(center_x - roi_size, 0)
        x_end = min(center_x + roi_size, width)
        y_start = max(center_y - roi_size, 0)
        y_end = min(center_y + roi_size, height)

        circle_roi = new_image[y_start:y_end, x_start:x_end]
        circle_roi_blurred = cv2.GaussianBlur(circle_roi, (3, 3), 0)

        # Replace the original ROI with the blurred version
        new_image[y_start:y_end, x_start:x_end] = circle_roi_blurred

        # selected_color = hsv_roi[y, x]
        # selected_color = selected_color.astype(int)
        # tolerance = 30
        #
        # lower_bound = np.array([
        #     max(0, selected_color[0] - tolerance),
        #     max(0, selected_color[1] - tolerance),
        #     max(0, selected_color[2] - tolerance)
        # ], dtype=np.uint8)
        #
        # upper_bound = np.array([
        #     min(180, selected_color[0] + tolerance),
        #     min(255, selected_color[1] + tolerance),
        #     min(255, selected_color[2] + tolerance)
        # ], dtype=np.uint8)
        #
        # mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
        #
        # resized_texture = cv2.resize(self.wood_color, (width, height), interpolation=cv2.INTER_LINEAR)
        # texture_overlay = np.zeros_like(roi)
        # texture_overlay[mask > 0] = resized_texture[mask > 0]
        #
        # roi[mask > 0] = self.ball_color
        # # roi[mask == 0] = texture_overlay[mask > 0]
        # new_image[ymin:ymax, xmin:xmax] = roi
        return new_image

    def __call__(self, image, boxes, labels):

        # print("Boxes:", boxes.tolist())
        # print("Labels:", labels.tolist())

        # if isinstance(image, torch.Tensor):  # <-- Change: Handle PyTorch tensor inputs
        #     image = image.permute(1, 2, 0).numpy()  # Convert to NumPy for OpenCV
        image = np.array(image)

        if labels.tolist() == [label_map["tennis-ball"]]:
            # print(labels.tolist())
            # Labels check unchanged
            transformed_image = self.court_color(image)
            transformed_image = self.change_lines(image, transformed_image)
            transformed_image = self.change_ball(image, transformed_image, boxes[0].tolist())
        else:
            transformed_image = image

        # Convert back to PyTorch tensor <-- Change: Ensure Tensor output
        transformed_image = torch.from_numpy(transformed_image).permute(2, 0, 1).float() / 255.0

        # Return consistent Tuple[Tensor, Tensor, Tensor] <-- Change: Normalize and format output
        labels = torch.tensor([1])
        return transformed_image, boxes, labels
        #
        # image = np.array(image)
        # # Apply court color transformation
        # if (labels.tolist() == [label_map["tennis-ball"]]):
        #     # print("TRANSFORMING")
        #     new_image = self.court_color(image)
        #     # Apply line transformation
        #     new_image = self.change_lines(image, new_image)
        #     # Apply ball transformation for each bounding box
        #     # for box, label in zip(boxes.tolist(), labels.tolist()):
        #
        #     new_image = self.change_ball(image, new_image, boxes.tolist()[0])
        #     return new_image, boxes, labels
        # else:
        #     # print("PASSED")
        #     return image, boxes, labels

label_map = {
    "Squash": 1,
    "tennis-ball": 0
}