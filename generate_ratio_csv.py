import pandas as pd
import argparse

def process_csv(file1, file2, output_file, start_row, end_row, subset_only):
    """
    Processes CSV files based on the `subset_only` flag.
    If `subset_only` is True, saves a subset of file1 to output_file.
    Otherwise, merges a range of rows from file1 into file2 and saves the result.

    Args:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.
        output_file (str): Path to save the resulting CSV file.
        start_row (int): Starting row index (inclusive) for slicing from file1.
        end_row (int): Ending row index (exclusive) for slicing from file1.
        subset_only (bool): Flag to determine whether to save only a subset of file1.
    """
    # Load the first CSV file
    try:
        df1 = pd.read_csv(file1)
    except Exception as e:
        print(f"Error loading file1: {e}")
        return

    # Ensure range is valid
    if start_row < 0 or end_row < 0 or start_row >= len(df1):
        print(f"Invalid range: start_row={start_row}, end_row={end_row}, file1 rows={len(df1)}")
        return

    # Adjust end_row to be within bounds
    end_row = min(end_row, len(df1))

    # Extract the range of rows from the first CSV
    sliced_df1 = df1.iloc[start_row:end_row]

    if subset_only:
        # Save only the subset of df1
        try:
            sliced_df1.to_csv(output_file, index=False)
            print(f"Subset of file1 saved to {output_file}")
        except Exception as e:
            print(f"Error saving subset: {e}")
        return

    # Load the second CSV file
    try:
        df2 = pd.read_csv(file2)
    except Exception as e:
        print(f"Error loading file2: {e}")
        return

    # Append the subset of df1 to df2
    merged_df = pd.concat([df2, sliced_df1], ignore_index=True)

    # Save the merged result
    try:
        merged_df.to_csv(output_file, index=False)
        print(f"Merged CSV saved to {output_file}")
    except Exception as e:
        print(f"Error saving merged CSV: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files by merging or extracting a subset.")
    parser.add_argument("--file1", type=str, help="Path to the first CSV file.", default="/cluster/tufts/cs152l3dclass/oeiels01/train.csv")
    parser.add_argument("--file2", type=str, help="Path to the second CSV file.", default="/cluster/tufts/cs152l3dclass/oeiels01/synthetic.csv")
    parser.add_argument("--output_file", type=str, help="Path to save the resulting CSV file.", default="ratio.csv")
    parser.add_argument("--start_row", type=int, help="Starting row index (inclusive) from file1.", default=0)
    parser.add_argument("--end_row", type=int, help="Ending row index (exclusive) from file1.", default=-1)
    parser.add_argument("--subset_only", action="store_true", help="If set, saves only a subset of file1 to the output file.")

    args = parser.parse_args()

    process_csv(args.file1, args.file2, args.output_file, args.start_row, args.end_row, args.subset_only)