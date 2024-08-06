import os 
import polars as pl
import dask.dataframe as dd 

def shorten(path, num_row=0):
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
        df = pl.read_csv(original_file_path, n_rows=num_row)

        print(f"Processing {csv_file}...")
        
        # Construct the new file name
        short_file_name = os.path.splitext(csv_file)[0] + '.csv'
        short_file_path = os.path.join(new_folder_path, short_file_name)
        
        # Save the smaller version of the file in the new folder
        df.write_csv(short_file_path)

    print("Processed all CSV files and saved the short versions in 'MIMIC-IV-short' folder.")


# Function to check if a column can be converted to datetime
def check_datetime_convertible(df, col):
    try:
        # Attempt to convert the column to datetime and compute a small part to test
        _ = dd.to_datetime(df[col], errors='raise').head()
        
        # If conversion is successful, return True
        return True
    except:
        # If conversion fails, return False
        return False

if __name__ == "__main__":

    icu_tables = ["chartevents", "d_items", "ingredientevents", "inputevents", "outputevents", "procedureevents"]
    hosp_tables = ["patients", "diagnoses_icd", "labevents", "microbiologyevents", "admissions", "d_icd_diagnoses", "d_icd_procedures", "procedures_icd"]

    # Load all tables with Dask, using lazy evaluation, and handle missing values with assume_missing=True
    data = {
        table: dd.read_csv(f"./data/MIMIC-IV-short/icu/{table}.csv" if table in icu_tables
                        else f"./data/MIMIC-IV-short/hosp/{table}.csv", assume_missing=True)
        for table in icu_tables + hosp_tables
    }

    time_names = {"starttime", "startdate", "endtime", "stopdate", "charttime", "chartdate"}

    # Iterate through each DataFrame
    for key, df in data.items():
        print(f"DataFrame {key}", end=' ')
        
        datetime_columns = []
        for col in df.columns:
            if col in time_names:
                datetime_columns.append(col)

        if len(datetime_columns) > 0:

            print(datetime_columns)