import torch
import polars as pl
import os 
from datasets import load_dataset
from huggingface_hub import HfApi

class InterPatientHypergraphModule:
    def __init__(self, organization, cache_dir, num_samples):
        self.organization = organization
        self.cache_dir = cache_dir
        self.num_samples = num_samples

        self.shared_diagnosis_hyperedges = {}
        self.comorbidity_hyperedges = {}

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