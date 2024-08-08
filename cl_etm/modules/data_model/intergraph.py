import torch
import polars as pl

import os 

from cl_etm.utils.eda import get_files_in_dir

class InterPatientHypergraphModule:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.shared_diagnosis_hyperedges = {}
        self.comorbidity_hyperedges = {}

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
                if patient_id not in self.shared_diagnosis_hyperedges:
                    self.shared_diagnosis_hyperedges[patient_id] = []
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
                    if subject_id not in self.comorbidity_hyperedges:
                        self.comorbidity_hyperedges[subject_id] = []
                    self.comorbidity_hyperedges[subject_id].append(other_subject_id)

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
