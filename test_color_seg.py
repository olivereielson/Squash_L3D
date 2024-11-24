# import cv2
# import numpy as np
# import os
# import statistics
# import xml.etree.ElementTree as ET
#
#
# def court_color(image):
#     # image = cv2.imread(image_path)
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     height, width, channels = image.shape
#
#     a = []
#     b = []
#     c = []
#     for j in range(-50, 50, 1):
#         x, y = height // 2 + j, width // 2 + j  # Example coordinates, change based on your image
#         selected_color = (hsv_image[y, x])  # Get the HSV color at the (x, y) position
#         # print(selected_color)
#         a.append(selected_color[0])
#         b.append(selected_color[1])
#         c.append(selected_color[2])
#
#     # a_med = np.median(a)
#     # b_med = np.median(b)
#     # c_med = np.median(c)
#     a_med = np.mean(a)
#     b_med = np.mean(b)
#     c_med = np.mean(c)
#
#     selected_color = np.array([a_med, b_med, c_med])
#     tolerance = 60  # You can adjust the tolerance
#
#     selected_color = selected_color.astype(int)
#
#     lower_bound = np.array([
#         max(0, selected_color[0] - tolerance),  # Hue, clamp to 0
#         max(0, selected_color[1] - tolerance),  # Saturation, clamp to 0
#         max(0, selected_color[2] - tolerance)  # Value, clamp to 0
#     ], dtype=np.uint8)
#
#     upper_bound = np.array([
#         min(180, selected_color[0] + tolerance),  # Hue, clamp to 180
#         min(255, selected_color[1] + tolerance),  # Saturation, clamp to 255
#         min(255, selected_color[2] + tolerance)  # Value, clamp to 255
#     ], dtype=np.uint8)
#
#     mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
#
#     new_image = image.copy()
#     wood_color = [140, 193, 255]  # Example wood color in BGR format
#     # Apply the color change to the selected areas
#     new_image[mask > 0] = wood_color
#
#     return new_image
#
# def change_lines(og_image, new_image):
#     hsv_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2HSV)
#
#     lower_white = np.array([0, 0, 200], dtype=np.uint8)
#     upper_white = np.array([180, 120, 255], dtype=np.uint8)
#
#     mask = cv2.inRange(hsv_image, lower_white, upper_white)
#     new_image[mask > 0] = [0, 0, 255]
#
#     return new_image
#
# def change_ball(og_image, new_image, bnd_box):
#     xmin, ymin, xmax, ymax = bnd_box
#     # print(xmin, ymin, xmax, ymax)
#     roi = og_image[ymin:ymax, xmin:xmax]
#
#     hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#
#     height, width, channels = roi.shape
#
#     # print(height, width)
#
#     x, y = width // 2, height // 2
#     selected_color = (hsv_roi[y, x])
#
#     tolerance = 20  # You can adjust the tolerance
#
#     selected_color = selected_color.astype(int)
#
#     lower_bound = np.array([
#         max(0, selected_color[0] - tolerance),  # Hue, clamp to 0
#         max(0, selected_color[1] - tolerance),  # Saturation, clamp to 0
#         max(0, selected_color[2] - tolerance)  # Value, clamp to 0
#     ], dtype=np.uint8)
#
#     upper_bound = np.array([
#         min(180, selected_color[0] + tolerance),  # Hue, clamp to 180
#         min(255, selected_color[1] + tolerance),  # Saturation, clamp to 255
#         min(255, selected_color[2] + tolerance)  # Value, clamp to 255
#     ], dtype=np.uint8)
#
#     mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
#
#     # new_image = og_image.copy()
#     # Apply the color change to the selected areas
#     roi[mask > 0] = [0, 0, 0]
#     roi[mask == 0] = [140, 193, 255]
#     new_image[ymin:ymax, xmin:xmax] = roi
#
#     return new_image
#
#
#
# def main():
#
#     folder_path = "/Users/youssefezzo/Desktop/tennis-tracker/train"
#     count = 0
#     for filename in os.listdir(folder_path):
#
#         if filename.endswith(".xml"):  # Check for image file extensions
#             xml_path = os.path.join(folder_path, filename)
#             # print("path", xml_path)
#             if xml_path is not None:
#                 tree = ET.parse(xml_path)
#                 root = tree.getroot()
#
#                 img_path = root.find("filename").text
#                 img_path = os.path.join(folder_path, img_path)
#                 image = cv2.imread(img_path)
#
#                 for bndbox in root.findall("object"):
#                     bounding_box = bndbox.find("bndbox")
#                     xmin = int(bounding_box.find("xmin").text)
#                     ymin = int(bounding_box.find("ymin").text)
#                     xmax = int(bounding_box.find("xmax").text)
#                     ymax = int(bounding_box.find("ymax").text)
#                     # boxes.append([xmax, ymin, xmin, ymax, ball_class])
#
#                     # Validate and correct bounding box coordinates
#                     if xmax < xmin:
#                         xmin, xmax = xmax, xmin
#
#                     if ymax < ymin:
#                         ymin, ymax = ymax, ymin
#
#                     if xmax > xmin and ymax > ymin:
#                         box = [xmin, ymin, xmax, ymax]
#                     else:
#                         print(f"Skipping invalid box: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
#
#                 new_image = court_color(image)
#                 new_image = change_lines(image, new_image)
#                 new_image = change_ball(image, new_image, box)
#
#                 img_path_add = "f{i}.jpg".format(i=count)
#                 print(count)
#                 count += 1
#                 new_image_path = os.path.join('/Users/youssefezzo/Desktop/trainA_segment/', img_path_add)
#                 cv2.imwrite(new_image_path, new_image)
#                 cv2.destroyAllWindows()


import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

def court_color(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width, _ = image.shape

    a, b, c = [], [], []

    for j in range(-20, 20):
        x, y = (width // 2) + j, (height // 2) + j  # Coordinates for sampling court color
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
    wood_color = [140, 193, 255]  # Example wood color in BGR
    new_image[mask > 0] = wood_color

    return new_image

def change_lines(og_image, new_image):
    hsv_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 120, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv_image, lower_white, upper_white)
    new_image[mask > 0] = [0, 0, 255]  # Red lines

    return new_image

def change_ball(og_image, new_image, bnd_box):
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
    roi[mask > 0] = [0, 0, 0]  # Black ball
    roi[mask == 0] = [140, 193, 255]
    new_image[ymin:ymax, xmin:xmax] = roi

    return new_image

def main():
    folder_path = "/Users/youssefezzo/Desktop/tennis-tracker/train"
    output_folder = "/Users/youssefezzo/Desktop/trainA_segment/"
    count = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".xml"):
            xml_path = os.path.join(folder_path, filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            img_filename = root.find("filename").text
            img_path = os.path.join(folder_path, img_filename)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Failed to load image: {img_path}")
                continue

            new_image = court_color(image)
            new_image = change_lines(image, new_image)

            for bndbox in root.findall("object"):
                if bndbox.find("name").text == "tennis-ball":
                    bounding_box = bndbox.find("bndbox")
                    xmin = int(bounding_box.find("xmin").text)
                    ymin = int(bounding_box.find("ymin").text)
                    xmax = int(bounding_box.find("xmax").text)
                    ymax = int(bounding_box.find("ymax").text)

                    # Validate and correct bounding box coordinates
                    if xmax < xmin:
                        xmin, xmax = xmax, xmin
                    if ymax < ymin:
                        ymin, ymax = ymax, ymin

                    if xmax > xmin and ymax > ymin:
                        box = [xmin, ymin, xmax, ymax]
                        new_image = change_ball(image, new_image, box)
                    else:
                        print(f"Skipping invalid box: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

            new_img_filename = f"modified_image_{count}.jpg"
            new_image_path = os.path.join(output_folder, new_img_filename)
            cv2.imwrite(new_image_path, new_image)
            count += 1

    cv2.destroyAllWindows()












    # for i in range(1392):
        # Load the image
        # image = cv2.imread('/Users/youssefezzo/Desktop/trainA/'+str(i)+'.jpg')
        #
        # new_image = court_color(image)
        # new_image = change_lines(image, new_image)
        #
        # img_path = "f{i}.jpg".format(i=i)
        # new_image_path = os.path.join('/Users/youssefezzo/Desktop/trainA_segment/', img_path)
        # cv2.imwrite(new_image_path, new_image)
        # # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(i)

        # Convert to HSV color space
        # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # height, width, channels = image.shape
        # # Assume we grab the color from a specific pixel in the image, e.g., (x, y) coordinates
        # selected_color = 0
        # a = []
        # b = []
        # c = []
        # for j in range(0, 50, 5):
        #
        #     x, y = height//2 + j, width//2 +j  # Example coordinates, change based on your image
        #     selected_color  = (hsv_image[y, x])  # Get the HSV color at the (x, y) position
        #     # print(selected_color)
        #     a.append(selected_color[0])
        #     b.append(selected_color[1])
        #     c.append(selected_color[2])
        #
        #
        # a_med = np.median(a)
        # b_med = np.median(b)
        # c_med = np.median(c)
        # # selected_color = sorted(selected_color)
        # selected_color = np.array([a_med,b_med,c_med])
        # tolerance = 60  # You can adjust the tolerance
        # # lower_bound = np.array([0,0,0], dtype=np.uint8)
        #
        # selected_color = selected_color.astype(int)
        #
        #
        # lower_bound = np.array([
        #     max(0, selected_color[0] - tolerance),  # Hue, clamp to 0
        #     max(0, selected_color[1] - tolerance),  # Saturation, clamp to 0
        #     max(0, selected_color[2] - tolerance)   # Value, clamp to 0
        # ], dtype=np.uint8)
        #
        #
        # upper_bound = np.array([
        #     min(180, selected_color[0] + tolerance),  # Hue, clamp to 180
        #     min(255, selected_color[1] + tolerance),  # Saturation, clamp to 255
        #     min(255, selected_color[2] + tolerance)   # Value, clamp to 255
        # ], dtype=np.uint8)
        #
        #
        # # Define the color range for white
        # # You can adjust these values if needed
        # lower_white = np.array([0, 0, 200], dtype=np.uint8)
        # upper_white = np.array([180, 120, 255], dtype=np.uint8)
        #
        #
        # # Create a mask for white color
        # mask = cv2.inRange(hsv_image, lower_white, upper_white)
        #
        # mask2 = cv2.inRange(hsv_image, lower_bound, upper_bound)
        #
        # # Define the wood color in BGR format (for example, a brownish color)
        # wood_color = [140,193,255]  # Example wood color in BGR format
        # # Apply the color change to the selected areas
        #
        #
        # # Change the white lines to red in the original image
        # image[mask > 0] = [0, 0, 255]  # BGR format for red
        # image[mask2 > 0] = wood_color
        # # Save or display the modified image
        # # cv2.imshow('Modified Image', image)
        # img_path = "f{i}.jpg".format(i=i)
        # new_image_path = os.path.join('/Users/youssefezzo/Desktop/trainA_segment/', img_path)
        # cv2.imwrite(new_image_path, image)
        # # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(i)


if __name__ == "__main__":
    main()

# import cv2
# import numpy as np

# def change_line_color(image, new_color):
#     """Changes the color of lines in an image.
#
#     Args:
#         image (numpy.ndarray): The input image.
#         new_color (tuple): The new color for the lines (BGR format).
#
#     Returns:
#         numpy.ndarray: The image with lines in the new color.
#     """
#
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Apply Canny edge detection to find edges
#     edges = cv2.Canny(gray, 50, 150)
#
#     # Create a color image with the same dimensions as the input
#     color_image = np.zeros_like(image)
#
#     # Set the color of the edges to the new color
#     image[edges != 0] = new_color
#
#     return image
#
# def fill_red_lines(image):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Apply Canny edge detection to find edges
#     edges = cv2.Canny(gray, 50, 150)
#
#     # Find contours from the edges
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Fill the contours with red color
#     cv2.drawContours(image, contours, -1, (0, 0, 255), thickness=cv2.FILLED)
#
#     return image
#
# # Load the image
# image = cv2.imread('/Users/youssefezzo/Desktop/1383.jpg')
#
# # Change the color of lines to red
# new_color = (0, 0, 255)  # BGR format for red
# result = change_line_color(image, new_color)
# result = fill_red_lines(result)
# # Display the result
# cv2.imshow("Original Image", image)
# cv2.imshow("Lines in Red", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# POTENTIALLYY
# import cv2
# import numpy as np
#
# # Load the image
# image = cv2.imread('/Users/youssefezzo/Desktop/1383.jpg')
#
# # Convert the image to the HSV color space
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
# # Define the range for detecting white color in the HSV space
# lower_white = np.array([0, 0, 200], dtype=np.uint8)
# upper_white = np.array([180,105, 255], dtype=np.uint8)
#
# # Create a mask for white color
# mask = cv2.inRange(hsv, lower_white, upper_white)
#
# # Find contours of the white lines
# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # Draw the contours on the original image in red
# for contour in contours:
#     cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)  # Red color in BGR
#
# # Save or display the result
# cv2.imshow('Result', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
