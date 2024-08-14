import os
import polars as pl
import torch
import json
import argparse 

def shorten(path, num_row=0):
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
    file_list = []
    if recursive:
        for root, _, files in os.walk(dir):
            for file in files:
                file_list.append(os.path.join(root, file))
    else:
        file_list = [os.path.join(dir, file) for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))]

    return file_list    



def save_all_graphs(graph_data_list, file_path='data/graph_data/patient_graphs.pt'):
    torch.save(graph_data_list, file_path)
    print(f"Saved all graphs to {file_path}")



def load_all_graphs(file_path='data/graph_data/patient_graphs.pt'):
    return torch.load(file_path)



def create_unified_index(data):
    microbiologyevents_df, labevents_df, prescriptions_df, procedures_icd_df, diagnoses_icd_df = data["microbiologyevents"], data["labevents"], data["prescriptions"], data["procedures_icd"], data["diagnoses_icd"]
    # Function to construct the unified index
    def construct_unified_index(prefix, original_id):
        return f"{prefix}{str(original_id).zfill(7)}"
    
    print(microbiologyevents_df)

    # Create dictionaries for each table
    microbiologyevents_mapping = {row['microevent_id']: construct_unified_index(1, row['microevent_id']) 
                                for row in microbiologyevents_df.to_dicts()}

    labevents_mapping = {row['labevent_id']: construct_unified_index(2, row['labevent_id']) 
                        for row in labevents_df.to_dicts()}

    prescriptions_mapping = {row['poe_id']: construct_unified_index(3, row['poe_id']) 
                            for row in prescriptions_df.to_dicts()}

    procedures_icd_mapping = {f"{row['icd_code']}{row['icd_version']}": construct_unified_index(4, f"{row['icd_code']}{row['icd_version']}") 
                            for row in procedures_icd_df.to_dicts()}

    diagnoses_icd_mapping = {f"{row['icd_code']}{row['icd_version']}": construct_unified_index(4, f"{row['icd_code']}{row['icd_version']}") 
                            for row in diagnoses_icd_df.to_dicts()}

    # Create a dictionary with table names as keys and mappings as values
    unified_index_mapping = {
        'microbiologyevents': microbiologyevents_mapping,
        'labevents': labevents_mapping,
        'prescriptions': prescriptions_mapping,
        'procedures_icd': procedures_icd_mapping,
        'diagnoses_icd': diagnoses_icd_mapping
    }

    # Create the reverse mapping
    reverse_mapping = {}
    for table, mapping in unified_index_mapping.items():
        for original_id, unified_index in mapping.items():
            reverse_mapping[unified_index] = {
                'table': table,
                'original_id': original_id
            }

    # Save the mappings to JSON files
    with open('data/unified_index_mapping.json', 'w') as f:
        json.dump(unified_index_mapping, f, indent=4)

    with open('data/reverse_mapping.json', 'w') as f:
        json.dump(reverse_mapping, f, indent=4)

'''
MIMIC-IV
{
    "hosp": 
    {
        "admissions": DataFrame,
        "patients": DataFrame,
        ...
    },

    "icu":
    {
        "chartevents": DataFrame,
        "datetimeevents": DataFrame,
        ...
    }
}
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shorten CSV files by reducing the number of rows.")
    parser.add_argument("path", type=str, help="The path to the directory containing CSV files.")
    parser.add_argument("--num_rows", type=int, default=0, help="The number of rows to keep in each shortened CSV file.")

    args = parser.parse_args()

    # Shrink original data to num_row rows as specified by the user
    shorten(args.path, num_row=args.num_rows)

    # print(load_all_graphs("data/graph_data/patient_graphs.pt")[10051825].hyperedges)