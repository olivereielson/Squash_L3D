
import pandas as pd
import os

def validate_csv_files(csv_paths):
    """
    Validates a list of CSV files containing bounding box data.
    Checks for NULL values, data integrity, and ensures no overlap
    between train, test, and validation splits.

    Args:
        csv_paths (list): List of paths for the CSV files.

    Raises:
        ValueError: If any issues are found in the CSV files.

    """
    all_image_paths = set()  # To ensure there is no overlap in image paths across splits

    for csv_path in csv_paths:




        if not os.path.exists(csv_path):
            raise ValueError(f"CSV file {csv_path} does not exist.")

        # Load CSV file
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Error reading {csv_path}: {e}")

        # Check for NULL values
        if df.isnull().values.any():
            raise ValueError(f"CSV file {csv_path} contains NULL values.")

        # Check for required columns
        required_columns = {"image_path", "xmin", "ymin", "xmax", "ymax", "class"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV file {csv_path} is missing required columns: {required_columns - set(df.columns)}")

        # Check for valid bounding box coordinates
        if not (df["xmin"] <= df["xmax"]).all() or not (df["ymin"] <= df["ymax"]).all():
            raise ValueError(f"CSV file {csv_path} has invalid bounding box coordinates.")

        # Ensure all image paths are unique across splits
        current_image_paths = set(df["image_path"])
        if all_image_paths & current_image_paths:
            raise ValueError(f"Overlap found between image paths in {csv_path} and previous CSV files.")
        all_image_paths.update(current_image_paths)

    print("All CSV files passed validation!")