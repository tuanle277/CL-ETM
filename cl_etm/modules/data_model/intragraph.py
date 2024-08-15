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

class IntraPatientHypergraphModule:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.merged_df = None
        self.graph_data_list = None

    def load_data(self):
        tables = self._load_tables()
        self.merged_df = self._merge_tables(tables)
        self.graph_data_list = self._create_patient_hypergraphs(tables)

    def _load_tables(self):
        # Load CSV files and map them to table names
        tables = get_files_in_dir(f"{self.data_dir}/icu/") + get_files_in_dir(f"{self.data_dir}/hosp/")
        table_names = [os.path.splitext(os.path.basename(table))[0] for table in tables]
        data = {name: pl.read_csv(file, infer_schema_length=10**7) for name, file in zip(table_names, tables)}
        return data

    def _merge_tables(self, data):
        # Merge admissions and patients tables on 'subject_id'
        merged_df = data['admissions'].join(data['patients'], on='subject_id')
        return merged_df

    def _create_patient_hypergraphs(self, data):
        graph_data_list = {}

        for subject_id in tqdm(self.merged_df['subject_id'].unique(), desc="Creating patient hypergraphs..."):
            nodes, edge_index, edge_attr, hyperedges = self._process_patient_events(subject_id, data)
            if nodes:
                graph_data_list[subject_id] = Data(
                    x=torch.tensor(nodes),
                    edge_index=torch.tensor(edge_index).t().contiguous(),
                    edge_attr=torch.tensor(edge_attr) if edge_attr else torch.tensor([]),
                    hyperedges=hyperedges  # Store as a list of sets/lists
                )
            else:
                logger.info(f"No data for patient {subject_id}")


            print(graph_data_list[10000032].hyperedges)
        return graph_data_list

    def _process_patient_events(self, subject_id, data):
        nodes, edge_index, edge_attr, hyperedges = [], [], [], []
        node_index, current_index = {}, 0
        admission_hyperedge = set()
        sequential_hyperedge = set()
        co_occur_hyperedges = defaultdict(set)

        for _, df in data.items():
            if "subject_id" in df.columns:
                df = df.filter(pl.col('subject_id') == subject_id)
                if any(col in df.columns for col in {"starttime", "startdate", "charttime", "chartdate"}):
                    nodes, edge_index, edge_attr, node_index, current_index = self._create_nodes_edges(
                        df, nodes, edge_index, edge_attr, node_index, current_index, admission_hyperedge, sequential_hyperedge, co_occur_hyperedges
                    )

        hyperedges = self._finalize_hyperedges(admission_hyperedge, sequential_hyperedge, co_occur_hyperedges)

        return nodes, edge_index, edge_attr, hyperedges

    def _create_nodes_edges(self, df, nodes, edge_index, edge_attr, node_index, current_index, admission_hyperedge, sequential_hyperedge, co_occur_hyperedges):
        previous_treatment_node = None
        treatment = None  # Initialize the treatment variable

        for i, row in enumerate(df.iter_rows(named=True)):
            node_time, nodes, node_index, current_index = self._create_node(i, row, nodes, node_index, current_index)
            edge_index, edge_attr = self._create_temporal_edges(i, node_index, node_time, edge_index, edge_attr)

            if 'hadm_id' in row and row['hadm_id'] is not None:
                admission_hyperedge.add(node_index[i])

            treatment, sequential_hyperedge, previous_treatment_node = self._update_sequential_hyperedge(
                row, i, treatment, sequential_hyperedge, previous_treatment_node, node_index
            )

            co_occur_hyperedges = self._update_co_occur_hyperedges(df, node_index, i, co_occur_hyperedges)
        
        return nodes, edge_index, edge_attr, node_index, current_index

    def _create_node(self, i, row, nodes, node_index, current_index):
        time_value = next((row[col] for col in {"starttime", "startdate", "charttime", "chartdate"} if col in row and row[col]), None)
        if time_value:
            node_time = pd.to_datetime(time_value).timestamp()

            if i not in node_index:
                nodes.append([current_index])
                node_index[i] = current_index
                current_index += 1

        return node_time, nodes, node_index, current_index

    def _create_temporal_edges(self, i, node_index, node_time, edge_index, edge_attr):
        if i > 0 and (i - 1) in node_index:
            edge_index.append([node_index[i - 1], node_index[i]])
            edge_attr.append([node_time])
        
        return edge_index, edge_attr

    def _update_sequential_hyperedge(self, row, i, treatment, sequential_hyperedge, previous_treatment_node, node_index):
        treatment = row.get('medication') or row.get('procedure')
        if previous_treatment_node and treatment:
            sequential_hyperedge.update([previous_treatment_node, node_index[i]])

        previous_treatment_node = node_index[i]  # Update previous treatment node

        return treatment, sequential_hyperedge, previous_treatment_node

    def _update_co_occur_hyperedges(self, df, node_index, i, co_occur_hyperedges):
        if 'medication' in df.columns:
            medications = df['medication'].drop_nulls()
            for med1 in medications:
                for med2 in medications:
                    if med1 != med2:
                        co_occur_hyperedges[frozenset([med1, med2])].update([node_index[i]])
        
        return co_occur_hyperedges

    def _finalize_hyperedges(self, admission_hyperedge, sequential_hyperedge, co_occur_hyperedges):
        hyperedges = []

        '''
        
        _update_sequential_hyperedge: Updates the sequential hyperedge by linking treatments that occurred one after another.
        _update_co_occur_hyperedges: Updates co-occurrence hyperedges for medications that occurred together.
        _finalize_hyperedges: Finalizes the hyperedges, converting them into lists and storing them for each patient.

        '''
        
        if admission_hyperedge:
            hyperedges.append(list(admission_hyperedge))
        if sequential_hyperedge:
            hyperedges.append(list(sequential_hyperedge))
        for co_occur_edge in co_occur_hyperedges.values():
            hyperedges.append(list(co_occur_edge))

        return hyperedges


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process MIMIC-IV data and create patient graphs.")
    parser.add_argument("--data_dir", type=str, default="data/MIMIC-IV-short", help="Directory containing MIMIC-IV data.")

    args = parser.parse_args()

    # Initialize IntraPatientHypergraphModule with the provided data directory
    mimic = IntraPatientHypergraphModule(data_dir=args.data_dir)

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
