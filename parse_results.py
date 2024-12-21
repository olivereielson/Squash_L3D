import os
import json
import argparse

def find_best_metric_results(base_dir, metric):
    best_metric = -1
    best_hyperparams = None
    best_file = None

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("train_history.json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Get the maximum value of the specified metric and its index
                        if metric in data:
                            max_metric = max(data[metric])
                            if max_metric > best_metric:
                                best_metric = max_metric
                                best_hyperparams = data.get("hyperparams", {})
                                best_file = file_path
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    if best_hyperparams is not None:
        print(f"Best {metric}: {best_metric}")
        print(f"File: {best_file}")
        print("Hyperparameters:")
        for key, value in best_hyperparams.items():
            print(f"  {key}: {value}")
    else:
        print(f"No valid JSON files with {metric} found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the best results based on a specified metric.")
    parser.add_argument(
        "--Folder",
        type=str,
        default="Results",
        help="The base directory to search for train_history.json files."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="map_50",
        help="The metric to evaluate for the best results (default: map_50)."
    )
    args = parser.parse_args()

    find_best_metric_results(args.Folder, args.metric)