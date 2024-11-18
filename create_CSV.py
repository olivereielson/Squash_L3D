import os
import csv
import warnings
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm

def parse_voc_annotation(xml_file):
    """Parse a Pascal VOC XML file and return bounding boxes and labels."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    labels = []

    for obj in root.findall("object"):
        label = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    return boxes, labels


def generate_csv_from_path_list(path_list, output_csv):

    assert len(path_list) > 0, "No paths provided"


    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_path", "xmin", "ymin", "xmax", "ymax", "class"])

        # Process XML files and corresponding images
        for jpg in tqdm(path_list, desc=f"Processing Annotations for {output_csv}"):

            if jpg.endswith(".xml"):
                warnings.warn(f"There is an xml file in your path list: {jpg}.... that should not happen")
            assert jpg.endswith(".jpg"), f"File {jpg} is not a jpg file"

            xml_path = jpg.replace(".jpg", ".xml")
            image_path = jpg

            assert os.path.exists(xml_path), f"XML file {xml_path} does not exist"
            assert os.path.exists(image_path), f"Image file {image_path} does not exist"

            # Parse XML to get boxes and labels
            boxes, labels = parse_voc_annotation(xml_path)

            for box, label in zip(boxes, labels):
                writer.writerow([image_path, *box, label])

    print(f"CSV file created: {output_csv}")






def generate_csv(folder_path, output_csv,included_labels=[]):
    """Generates a CSV file listing image paths, bounding boxes, and labels."""
    assert os.path.exists(folder_path), f"Folder {folder_path} does not exist"

    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_path", "xmin", "ymin", "xmax", "ymax", "class"])

        # Process XML files and corresponding images
        for xml_file in tqdm(os.listdir(folder_path), desc="Processing Annotations"):
            if not xml_file.endswith(".xml"):
                continue
            xml_path = os.path.join(folder_path, xml_file)
            image_file = xml_file.replace(".xml", ".jpg")
            image_path = os.path.join(folder_path, image_file)

            # Verify image file exists
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_file} not found for annotation {xml_file}")
                continue

            # Parse XML to get boxes and labels
            boxes, labels = parse_voc_annotation(xml_path)


            for box, label in zip(boxes, labels):
                if label in included_labels:
                    writer.writerow([image_path, *box, label])

    print(f"CSV file created: {output_csv}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CSV from Pascal VOC annotations")
    parser.add_argument("folder", help="Path to the folder containing images and annotations")
    parser.add_argument("output", help="Output CSV file name")
    args = parser.parse_args()

    generate_csv(args.folder, args.output)