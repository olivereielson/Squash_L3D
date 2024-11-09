import pandas as pd
from sklearn.model_selection import train_test_split

def split_csv_grouped(file_path, train_ratio, output_train, output_test):
    """
    Splits a CSV file into training and testing CSV files while keeping all bounding boxes
    for each image in the same split.

    Args:
        file_path (str): Path to the original CSV file.
        train_ratio (float): Ratio of data to use for the training set (default is 0.8).
        output_train (str): Filename for the training CSV (default is "trainA.csv").
        output_test (str): Filename for the testing CSV (default is "testA.csv").

    Returns:
        None: The function saves two CSV files: one for training and one for testing.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Get unique image paths
    image_paths = df['image_path'].unique()

    # Split image paths into train and test sets
    train_image_paths, test_image_paths = train_test_split(image_paths, train_size=train_ratio)

    # Split the DataFrame based on the image paths
    train_df = df[df['image_path'].isin(train_image_paths)]
    test_df = df[df['image_path'].isin(test_image_paths)]

    # Save the splits to new CSV files
    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)

    print(f"Data split complete. Training data saved to {output_train}, testing data saved to {output_test}.")