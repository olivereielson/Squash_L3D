import glob
import os

import pandas as pd

from create_CSV import generate_csv_from_path_list

def create_tennis_csv(direc_path, tennis_path):

    assert os.path.exists(direc_path), "SSD path does not exist...idiot"
    assert os.path.exists(tennis_path), f"Tennis path {tennis_path} does not exist"



    tennis_images = glob.glob(f"{tennis_path}/*.jpg")

    # print(relative_paths)
    generate_csv_from_path_list(tennis_images, f"{tennis_path}/tennis_path.csv")


    # csv1 = pd.read_csv(f"{direc_path}/train.csv")
    csv2 = pd.read_csv(f"{tennis_path}/tennis_path.csv")
    csv2 = csv2[csv2['class'] == 'tennis-ball']
    # print(csv2)

    csv2["image_path"] = csv2["image_path"].str.replace(f"{direc_path}/", "", regex=False)
    csv2.to_csv(f"{direc_path}/train_tennis.csv", index=False)

    # Combine the two DataFrames by appending rows
    # merged_csv = pd.concat([csv1, csv2], ignore_index=True)
    # merged_csv.to_csv(f"{direc_path}/train+segmented.csv", index=False)

    return csv2

def main():
    direc_path = "/cluster/tufts/cs152l3dclass/oeiels01"
    tennis_path = "/cluster/tufts/cs152l3dclass/oeiels01/tennis"

    create_tennis_csv(direc_path, tennis_path)

if __name__ == "__main__":
    main()