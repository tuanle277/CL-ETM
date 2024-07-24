import torch
from torch.utils.data import DataLoader
import numpy as np
import dgl

class MIMICIVDataLoader:
    def __init__(self, patient_graphs, batch_size=32, validation_split=0.1, test_split=0.1):
        self.patient_graphs = patient_graphs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split

    def create_dataloaders(self):
        """Creates dataloaders for training, validation, and testing."""
        num_patients = len(self.patient_graphs)
        num_val = int(num_patients * self.validation_split)
        num_test = int(num_patients * self.test_split)
        
        patient_ids = list(range(num_patients))
        np.random.shuffle(patient_ids)
        train_ids = patient_ids[:-num_val-num_test]
        val_ids = patient_ids[-num_val-num_test:-num_test]
        test_ids = patient_ids[-num_test:]

        train_loader = DataLoader([self.patient_graphs[i] for i in train_ids], batch_size=self.batch_size, shuffle=True, collate_fn=dgl.batch)
        val_loader = DataLoader([self.patient_graphs[i] for i in val_ids], batch_size=self.batch_size, shuffle=False, collate_fn=dgl.batch)
        test_loader = DataLoader([self.patient_graphs[i] for i in test_ids], batch_size=self.batch_size, shuffle=False, collate_fn=dgl.batch)

        return train_loader, val_loader, test_loader


# Sample run 

from data import MIMICIVDataModule

if __name__ == "__main__":
    # Initialize data module and load data
    mimic_data_module = MIMICIVDataModule()
    patients, admissions, events = mimic_data_module.load_data()
    mimic_data_module.mimic_data = {'patients': patients, 'admissions': admissions, 'events': events}

    # Construct graphs
    mimic_data_module.construct_patient_hypergraphs()
    mimic_data_module.construct_patient_disease_graph()

    # Initialize data loader with patient graphs
    mimic_data_loader = MIMICIVDataLoader(mimic_data_module.patient_graphs)
    train_loader, val_loader, test_loader = mimic_data_loader.create_dataloaders()

    # # Visualize the first patient hypergraph
    # mimic_data_module.visualize_patient_hypergraph(mimic_data_module.patient_graphs[0])

    # # Visualize the patient-disease graph
    # mimic_data_module.visualize_patient_disease_graph(mimic_data_module.patient_disease_graph)

    # Print a sample graph from the training DataLoader
    for batch in train_loader:
        print(batch)
        break  # Print only the first batch
