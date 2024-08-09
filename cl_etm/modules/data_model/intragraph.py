import torch
import polars as pl
import pandas as pd
import logging
from tqdm import tqdm
from torch_geometric.data import Data
import os
import argparse
import random
from collections import defaultdict

from cl_etm.utils.eda import get_files_in_dir, save_all_graphs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define mappings for special relationships
drug_disease_relations = {
    # Example: 'DrugName': ['Disease1', 'Disease2']
    'Aspirin': ['Heart Disease', 'Stroke'],
    # Add more drug-disease relationships here
}

class IntraPatientHypergraphModule:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.merged_df = None 
        self.graph_data_list = None 

    def load_data(self):
        # Load CSV files and map them to table names
        tables = get_files_in_dir(f"{self.data_dir}/icu/") + get_files_in_dir(f"{self.data_dir}/hosp/")
        table_names = [os.path.splitext(os.path.basename(table))[0] for table in tables]
        data = {name: pl.read_csv(file, infer_schema_length=10**7) for name, file in zip(table_names, tables)}

        # Merge admissions and patients tables on 'subject_id'
        merged_df = data['admissions'].join(data['patients'], on='subject_id')
        self.merged_df = merged_df
        graph_data_list = {}

        # Dictionary to store co-occurrence counts
        co_occurrence_dict = defaultdict(int)

        # Iterate over each unique subject_id
        for subject_id in tqdm(merged_df['subject_id'].unique(), desc="Creating patient hypergraphs..."):
            nodes, edge_index, edge_attr, hyperedges = [], [], [], []
            node_index, current_index = {}, 0

            # Process events from each relevant table
            for table_name in table_names:
                df = data[table_name]
                if "subject_id" in df.columns:
                    df = df.filter(pl.col('subject_id') == subject_id)
                    # Check for time-related columns and iterate through rows

                    if any(col in df.columns for col in {"starttime", "startdate", "charttime", "chartdate"}):

                        for i, row in enumerate(df.iter_rows(named=True)):
                            # Convert the first available time column to a timestamp
                            time_value = next((row[col] for col in {"starttime", "startdate", "charttime", "chartdate"} if col in row and row[col]), None)
                            previous_treatment = None
                            
                            # Convert to timestamp if a valid time value exists
                            if time_value:
                                node_time = pd.to_datetime(time_value).timestamp()

                                # Create nodes and index them
                                if i not in node_index:
                                    nodes.append([current_index])
                                    node_index[i] = current_index
                                    current_index += 1

                                # Create temporal edges between consecutive nodes
                                if i > 0 and (i - 1) in node_index:
                                    edge_index.append([node_index[i - 1], node_index[i]])
                                    edge_attr.append([node_time])

                                # Collect hyperedges based on admission IDs
                                if 'hadm_id' in row and row['hadm_id'] is not None:
                                    hyperedges.append(int(row['hadm_id']))

                                # Sequential Treatment Hyperedges
                                treatment = row.get('medication') or row.get('procedure')
                                if previous_treatment and treatment:
                                    hyperedges.append([previous_treatment, treatment])
                                
                                previous_treatment = treatment  # Update previous treatment

                                # Additional Hyperedges
                                if 'medication' in df.columns:
                                    medications = df['medication'].drop_nulls()
                                    for med1 in medications:
                                        for med2 in medications:
                                            if med1 != med2:
                                                pair = tuple(sorted([med1, med2]))
                                                co_occurrence_dict[pair] += 1

                                    # Creating hyperedges between co-occurring medications based on frequency
                                    for pair, freq in co_occurrence_dict.items():
                                        if freq > 1:  # Only create hyperedge if co-occurrence is above a threshold
                                            hyperedges.append(list(pair))

                                if 'diagnosis' in df.columns:
                                    # Creating hyperedges between drugs and diseases
                                    for drug, diseases in drug_disease_relations.items():
                                        if drug in df['medication'].unique():
                                            for disease in diseases:
                                                if disease in df['diagnosis'].unique():
                                                    hyperedges.append([drug, disease])

            # Create a PyTorch Geometric Data object if nodes exist
            if nodes:
                graph_data_list[subject_id] = Data(
                    x=torch.tensor(nodes),
                    edge_index=torch.tensor(edge_index).t().contiguous(),
                    edge_attr=torch.tensor(edge_attr) if edge_attr else torch.tensor([]),
                    hyperedges=torch.tensor(hyperedges) if hyperedges else torch.tensor([])
                )
            else:
                logger.info(f"No data for patient {subject_id}")

        self.graph_data_list = graph_data_list
        # Display the first part of each patient's hypergraph
        for k, v in graph_data_list.items():
            if v:
                print(k, str(v)[:90])

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process MIMIC-IV data and create patient graphs.")
    parser.add_argument("--data_dir", type=str, default="data/MIMIC-IV-short", help="Directory containing MIMIC-IV data.")

    args = parser.parse_args()

    # Initialize MIMICIVDataModule with the provided data directory
    mimic = IntraPatientHypergraphModule(data_dir=args.data_dir)

    # Load data and create graphs
    mimic.load_data()

    # Save all generated graphs to the specified file path
    save_all_graphs(mimic.graph_data_list, file_path="data/graph_data/patient_graphs.pt")

    # Inspect the merged data and optionally a specific subject's graph
    print("Merged DataFrame:")
    print(mimic.merged_df.head())
    
    subject_id = random.randrange(mimic.graph_data_list.keys())
    if subject_id is not None and subject_id in mimic.graph_data_list:
        print(f"Graph for subject {subject_id}:")
        print(mimic.graph_data_list[subject_id])
    else:
        print(f"No specific subject ID provided or subject ID {args.subject_id} not found.")
