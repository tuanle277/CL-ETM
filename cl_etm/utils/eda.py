import os
import polars as pl
import torch
import argparse

def shorten(path, num_row=0):
    """
    Shortens CSV files in the specified directory by reducing the number of rows.

    Parameters:
    - path (str): The directory containing the CSV files.
    - num_row (int): The number of rows to keep in each shortened CSV file.
    """
    new_folder_path = os.path.join(os.path.dirname(path), '../MIMIC-IV-short/hosp')
    os.makedirs(new_folder_path, exist_ok=True)

    files = os.listdir(path)
    csv_files = [file for file in files if file.endswith('.csv')]

    for csv_file in csv_files:
        original_file_path = os.path.join(path, csv_file)
        df = pl.read_csv(original_file_path, n_rows=num_row, infer_schema_length=num_row*2)
        print(f"Processing {csv_file}...")

        short_file_name = os.path.splitext(csv_file)[0] + '.csv'
        short_file_path = os.path.join(new_folder_path, short_file_name)
        df.write_csv(short_file_path)

    print("Processed all CSV files and saved the short versions in 'MIMIC-IV-short' folder.")

def get_files_in_dir(dir, recursive=False):
    """
    Get a list of all files in a directory.

    Parameters:
    - dir (str): The directory path to search.
    - recursive (bool): If True, includes files in subdirectories.

    Returns:
    - List[str]: A list of file paths.
    """
    file_list = []
    if recursive:
        for root, dirs, files in os.walk(dir):
            for file in files:
                file_list.append(os.path.join(root, file))
    else:
        file_list = [os.path.join(dir, file) for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))]

    return file_list    

def save_all_graphs(graph_data_list, file_path='saved_graphs/all_graphs.pt'):
    """
    Save all graphs in the list to a single file.

    Parameters:
    - graph_data_list (dict): The dictionary containing all graph data.
    - file_path (str): The path where the file will be saved.
    """
    torch.save(graph_data_list, file_path)
    print(f"Saved all graphs to {file_path}")

def load_all_graphs(file_path='saved_graphs/all_graphs.pt'):
    """
    Load all graphs from the specified file.

    Parameters:
    - file_path (str): The path to the file containing the saved graph data.

    Returns:
    - dict: The dictionary containing all loaded graph data.
    """
    return torch.load(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shorten CSV files by reducing the number of rows.")
    parser.add_argument("path", type=str, help="The path to the directory containing CSV files.")
    parser.add_argument("--num_rows", type=int, default=0, help="The number of rows to keep in each shortened CSV file.")

    args = parser.parse_args()

    # Shrink original data to num_row rows as specified by the user
    shorten(args.path, num_row=args.num_rows)
