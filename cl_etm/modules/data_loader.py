import torch
from torch.utils.data import DataLoader
import numpy as np
from torch_geometric.data import Batch

from cl_etm.utils.eda import load_all_graphs

class MIMICIVDataLoader:
    def __init__(self, patient_graphs, batch_size=32, validation_split=0.1, test_split=0.1):
        self.patient_graphs = patient_graphs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split

    def create_dataloaders(self):
        """Creates dataloaders for training, validation, and testing."""
        patient_keys = list(self.patient_graphs.keys())  # If patient_graphs is a dictionary
        num_patients = len(patient_keys)
        num_val = int(num_patients * self.validation_split)
        num_test = int(num_patients * self.test_split)
        
        np.random.shuffle(patient_keys)
        
        # Splitting keys
        train_keys = patient_keys[:-num_val-num_test]
        val_keys = patient_keys[-num_val-num_test:-num_test]
        test_keys = patient_keys[-num_test:]

        # Creating DataLoaders
        train_loader = DataLoader([self.patient_graphs[key] for key in train_keys], batch_size=self.batch_size, shuffle=True, collate_fn=self.collate)
        val_loader = DataLoader([self.patient_graphs[key] for key in val_keys], batch_size=self.batch_size, shuffle=False, collate_fn=self.collate)
        test_loader = DataLoader([self.patient_graphs[key] for key in test_keys], batch_size=self.batch_size, shuffle=False, collate_fn=self.collate)

        return train_loader, val_loader, test_loader

    @staticmethod
    def collate(batch):
        """Custom collate function for PyTorch Geometric's Data objects."""
        return Batch.from_data_list(batch)

# Sample run 

from data import MIMICIVDataModule

if __name__ == "__main__":
    # Initialize data module and load data
    # mimic_data_module = MIMICIVDataModule("data/MIMIC-IV-short")
    # mimic_data_module.load_data()

    # Construct patient graphs
    # mimic_data_module.construct_patient_hypergraphs()

    graph_data = load_all_graphs("./data/graph_data/patient_graphs.pt")

    # Initialize data loader with patient graphs
    mimic_data_loader = MIMICIVDataLoader(graph_data)
    train_loader, val_loader, test_loader = mimic_data_loader.create_dataloaders()

    # Print a sample batch from the training DataLoader
    for batch in train_loader:
        print(batch)
        break  # Print only the first batch
