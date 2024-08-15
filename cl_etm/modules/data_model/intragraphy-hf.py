from tqdm import tqdm
from torch_geometric.data import Data
from collections import defaultdict
from datasets import load_dataset
from huggingface_hub import HfApi

import torch
import polars as pl
import pandas as pd
import logging
import os
import argparse
import random

from cl_etm.utils.eda import save_all_graphs, create_unified_index

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message=s')
logger = logging.getLogger(__name__)

# Define mappings for special relationships
special_relations = {
    'Aspirin': ['Heart Disease', 'Stroke'],
    # Add more drug-disease relationships here
}

class IntraPatientHypergraphModule:
    def __init__(self, organization, cache_dir, num_samples):
        self.organization = organization
        self.cache_dir = cache_dir
        self.num_samples = num_samples

        self.merged_df = None
        self.graph_data_list = None

    def create_streaming_data(self):
        # Get all datasets from the organization
        api = HfApi()
        datasets = api.list_datasets(author=self.organization)

        # Load the dataset with streaming and caching enabled
        data = {}
        for dataset_info in datasets:
            dataset_name = dataset_info.id
            print(f"Processing dataset: {dataset_name}")

            dataset = load_dataset(dataset_name, streaming=True, cache_dir=self.cache_dir)['train']
            data[dataset_name.split("/")[-1]] = dataset.take(self.num_samples)
        return data

    def convert_to_polars(self, stream_dataset):
        processed_data = [each for each in stream_dataset]
        return pl.from_dicts(processed_data, infer_schema_length=1000)

    def load_data(self):
        data = self.create_streaming_data()
        data = {name: self.convert_to_polars(data[name]) for name in data.keys()}
        
        data_admissions = data['admissions']
        data_patients = data['patients']
        
        # Merge admissions and patients tables on 'subject_id'
        merged_df = data_admissions.join(data_patients, on='subject_id')
        self.merged_df = merged_df
        graph_data_list = {}

        create_unified_index(data)
        exit()

        # Dictionary to store co-occurrence counts
        co_occurrence_dict = defaultdict(int)

        #  Iterate over each unique subject_id
        for subject_id in tqdm(merged_df['subject_id'].unique(), desc="Creating patient hypergraphs..."):
            nodes, edge_index, edge_attr, hyperedges = [], [], [], []
            node_index, current_index = {}, 0
            admission_hyperedge = set()
            sequential_hyperedge = set()
            co_occur_hyperedges = defaultdict(set)

            # Process events from each relevant table
            for table_name in data.keys():
                df = data[table_name]

                if "subject_id" in df.columns:
                    df = df.filter(pl.col('subject_id') == subject_id)
                    if any(col in df.columns for col in {"starttime", "startdate", "charttime", "chartdate"}):

                        previous_treatment_node = None
                        previous_treatment = None

                        for i, row in enumerate(df.iter_rows(named=True)):
                            # Convert the first available time column to a timestamp
                            time_value = next((row[col] for col in {"starttime", "startdate", "charttime", "chartdate"} if col in row and row[col]), None)

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

                                # Add to admission-based hyperedge
                                if 'hadm_id' in row and row['hadm_id'] is not None:
                                    admission_hyperedge.add(node_index[i])

                                # Sequential Treatment Hyperedges
                                treatment = row.get('medication') or row.get('procedure')
                                if previous_treatment and treatment:
                                    sequential_hyperedge.update([previous_treatment_node, node_index[i]])

                                previous_treatment_node = node_index[i]  # Update previous treatment node

                                # Co-occurring Medications
                                if 'medication' in df.columns:
                                    medications = df['medication'].drop_nulls()
                                    for med1 in medications:
                                        for med2 in medications:
                                            if med1 != med2:
                                                co_occur_hyperedges[frozenset([med1, med2])].update([node_index[i]])

                                # Drug-Disease Relationships
                                if 'diagnosis' in df.columns:
                                    for drug, diseases in special_relations.items():
                                        if drug in df['medication'].unique():
                                            for disease in diseases:
                                                if disease in df['diagnosis'].unique():
                                                    hyperedges.append([drug, disease])

            # Finalize hyperedges for the patient
            if admission_hyperedge:
                hyperedges.append(list(admission_hyperedge))
            if sequential_hyperedge:
                hyperedges.append(list(sequential_hyperedge))
            for co_occur_edge in co_occur_hyperedges.values():
                hyperedges.append(list(co_occur_edge))

            # Create a PyTorch Geometric Data object if nodes exist
            if nodes:
                graph_data_list[subject_id] = Data(
                    x=torch.tensor(nodes),
                    edge_index=torch.tensor(edge_index).t().contiguous(),
                    edge_attr=torch.tensor(edge_attr) if edge_attr else torch.tensor([]),
                    hyperedges=hyperedges  # Store as a list of sets/lists
                )
            else:
                logger.info(f"No data for patient {subject_id}")
        self.graph_data_list = graph_data_list

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process MIMIC-IV data and create patient graphs.")
    parser.add_argument("--organization", type=str, default="CL-ETM", help="Directory containing MIMIC-IV data.")
    parser.add_argument('--cache_dir', type=str, default='.datasets', help='Directory to cache datasets')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples to load')
    args = parser.parse_args()

    # Initialize MIMICIVDataModule with the provided data directory
    mimic = IntraPatientHypergraphModule(args.organization, args.cache_dir, args.num_samples)

    # Load data and create graphs
    mimic.load_data()

    # Save all generated graphs to the specified file path
    save_all_graphs(mimic.graph_data_list, file_path="data/graph_data/patient_graphs.pt")

    # Inspect the merged data and optionally a specific subject's graph
    print("Merged DataFrame:")
    print(mimic.merged_df.head())
    
    subject_id = random.choice(list(mimic.graph_data_list.keys()))
    if subject_id is not None and subject_id in mimic.graph_data_list:
        print(f"Graph for subject {subject_id}:")
        print(mimic.graph_data_list[subject_id])
    else:
        print(f"No specific subject ID provided or subject ID {subject_id} not found.")
