import torch
import polars as pl
import os
from collections import defaultdict
from torch_geometric.data import Data
from cl_etm.utils.eda import get_files_in_dir

class InterPatientHypergraphModule:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.shared_diagnosis_hyperedges = defaultdict(list)
        self.comorbidity_hyperedges = defaultdict(list)
        self.graph_data = None

    def load_data(self):
        # Load CSV files and map them to table names
        tables = get_files_in_dir(f"{self.data_dir}/hosp/")
        data = {os.path.splitext(os.path.basename(file))[0]: pl.read_csv(file, infer_schema_length=10**7) for file in tables}
        return data

    def create_shared_diagnosis_hyperedges(self, diagnoses_df):
        """
        Create hyperedges that connect patients sharing the same diagnosis.
        """
        diagnoses_groups = diagnoses_df.groupby('icd_code')

        for _, group in diagnoses_groups:
            patients = group['subject_id'].unique()
            for i, patient_id in enumerate(patients):
                for other_patient_id in patients[i+1:]:
                    self.shared_diagnosis_hyperedges[patient_id].append(other_patient_id)

    def create_comorbidity_hyperedges(self, diagnoses_df):
        """
        Create hyperedges that connect patients with the same combination of diagnoses.
        """
        patient_diagnoses = diagnoses_df.groupby('subject_id')['icd_code'].apply(list)

        for subject_id, diagnoses in patient_diagnoses.items():
            for other_subject_id, other_diagnoses in patient_diagnoses.items():
                if subject_id != other_subject_id and set(diagnoses) == set(other_diagnoses):
                    self.comorbidity_hyperedges[subject_id].append(other_subject_id)

    def create_inter_patient_data_object(self):
        """
        Create a PyTorch Geometric Data object for the inter-patient hypergraph.
        """
        nodes = []
        edge_index = []
        hyperedges = []

        # Create nodes (each patient is a node)
        patient_ids = list(set(list(self.shared_diagnosis_hyperedges.keys()) + list(self.comorbidity_hyperedges.keys())))
        node_index = {patient_id: i for i, patient_id in enumerate(patient_ids)}
        nodes = torch.arange(len(patient_ids)).view(-1, 1)

        # Create edges for shared diagnoses
        for patient_id, connected_patients in self.shared_diagnosis_hyperedges.items():
            for other_patient_idn in connected_patients:
                edge_index.append([node_index[patient_id], node_index[other_patient_id]])

        # Create edges for comorbidities
        for patient_id, connected_patients in self.comorbidity_hyperedges.items():
            for other_patient_id in connected_patients:
                edge_index.append([node_index[patient_id], node_index[other_patient_id]])

        # Convert edge_index to tensor and transpose to match PyTorch Geometric format
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Combine all hyperedges into a single list (list of lists)
        hyperedges = [set(edge) for edge in self.shared_diagnosis_hyperedges.values()] + [set(edge) for edge in self.comorbidity_hyperedges.values()]

        # Create the Data object
        self.graph_data = Data(
            x=nodes,
            edge_index=edge_index,
            hyperedges=hyperedges
        )

    def save_shared_diagnosis_hyperedges(self, file_path="data/graph_data/shared_diagnosis_hyperedges.pt"):
        # Save shared diagnosis hyperedges
        torch.save(self.shared_diagnosis_hyperedges, file_path)
        print(f"Saved shared diagnosis hyperedges to {file_path}")

    def save_comorbidity_hyperedges(self, file_path="data/graph_data/comorbidity_hyperedges.pt"):
        # Save comorbidity hyperedges
        torch.save(self.comorbidity_hyperedges, file_path)
        print(f"Saved comorbidity hyperedges to {file_path}")

    def create_and_save_inter_hypergraphs(self, data):
        self.create_shared_diagnosis_hyperedges(data['diagnoses_icd'])
        self.save_shared_diagnosis_hyperedges()

        self.create_comorbidity_hyperedges(data['diagnoses_icd'])
        self.save_comorbidity_hyperedges()

        self.create_inter_patient_data_object()
        print("Created inter-patient hypergraph data object.")

