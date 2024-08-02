import os 
import polars as pl

path = 'data/MIMIC-IV/icu'

# Define the new folder path at the same level as the original folder
new_folder_path = os.path.join(os.path.dirname(path), '../MIMIC-IV-short/icu')

# Create the new folder if it doesn't exist
os.makedirs(new_folder_path, exist_ok=True)

# List all files in the original folder
files = os.listdir(path)

# Filter out only .csv files
csv_files = [file for file in files if file.endswith('.csv')]

# Process each .csv file
for csv_file in csv_files:
    # Construct the full path to the .csv file in the original folder
    original_file_path = os.path.join(path, csv_file)
    
    # Read the .csv file using polars
    df = pl.read_csv(original_file_path, n_rows=10)

    print(f"Processing {csv_file}...")
    
    # Construct the new file name
    short_file_name = os.path.splitext(csv_file)[0] + '.csv'
    short_file_path = os.path.join(new_folder_path, short_file_name)
    
    # Save the smaller version of the file in the new folder
    df.write_csv(short_file_path)

print("Processed all CSV files and saved the short versions in 'MIMIC-IV-short' folder.")