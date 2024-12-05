import os
import json

def find_best_map50_results(base_dir):
    best_map50 = -1
    best_hyperparams = None
    best_file = None

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("train_history.json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Get the maximum mAP_50 value and its index
                        if "map" in data:
                            max_map50 = max(data["map"])
                            if max_map50 > best_map50:
                                best_map50 = max_map50
                                best_hyperparams = data.get("hyperparams", {})
                                best_file = file_path
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    if best_hyperparams is not None:
        print(f"Best mAP_50: {best_map50}")
        print(f"File: {best_file}")
        print("Hyperparameters:")
        for key, value in best_hyperparams.items():
            print(f"  {key}: {value}")
    else:
        print("No valid JSON files with mAP_50 found.")

if __name__ == "__main__":
    # Replace 'Results-baseline' with the path to your directory
    base_dir = "Results_squash+segmented"
    find_best_map50_results(base_dir)