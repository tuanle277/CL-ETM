import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

class IntraPatientDataLoader:
    def __init__(self, graph_data_list, batch_size=32, validation_split=0.1, test_split=0.1):
        self.graph_data_list = graph_data_list
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split

    def create_dataloaders(self):
        """Creates dataloaders for training, validation, and testing."""
        patient_keys = list(self.graph_data_list.keys())
        num_patients = len(patient_keys)
        num_val = int(num_patients * self.validation_split)
        num_test = int(num_patients * self.test_split)
        
        torch.manual_seed(42)
        shuffled_keys = torch.randperm(num_patients).tolist()

        train_keys = [patient_keys[i] for i in shuffled_keys[:-num_val-num_test]]
        val_keys = [patient_keys[i] for i in shuffled_keys[-num_val-num_test:-num_test]]
        test_keys = [patient_keys[i] for i in shuffled_keys[-num_test:]]

        train_loader = DataLoader([self.graph_data_list[key] for key in train_keys],
                                  batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        val_loader = DataLoader([self.graph_data_list[key] for key in val_keys],
                                batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        test_loader = DataLoader([self.graph_data_list[key] for key in test_keys],
                                 batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

        return train_loader, val_loader, test_loader

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for PyTorch Geometric's Data objects."""
        return Batch.from_data_list(batch)

class InterPatientDataLoader:
    def __init__(self, hyperedge_data, batch_size=32, validation_split=0.1, test_split=0.1):
        self.hyperedge_data = hyperedge_data
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split

    def create_dataloaders(self):
        """Creates dataloaders for training, validation, and testing."""
        patient_keys = list(self.hyperedge_data.keys())
        num_patients = len(patient_keys)
        num_val = int(num_patients * self.validation_split)
        num_test = int(num_patients * self.test_split)

        torch.manual_seed(42)
        shuffled_keys = torch.randperm(num_patients).tolist()

        train_keys = [patient_keys[i] for i in shuffled_keys[:-num_val-num_test]]
        val_keys = [patient_keys[i] for i in shuffled_keys[-num_val-num_test:-num_test]]
        test_keys = [patient_keys[i] for i in shuffled_keys[-num_test:]]

        train_loader = DataLoader([self.create_graph(key) for key in train_keys],
                                  batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        val_loader = DataLoader([self.create_graph(key) for key in val_keys],
                                batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        test_loader = DataLoader([self.create_graph(key) for key in test_keys],
                                 batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

        return train_loader, val_loader, test_loader

    def create_graph(self, key):
        """Create a PyTorch Geometric Data object for each inter-patient hypergraph."""
        # For this example, we're assuming that hyperedges are represented as lists of patient indices
        hyperedges = self.hyperedge_data[key]
        edge_index = torch.tensor(hyperedges, dtype=torch.long).t().contiguous()

        # The node features are assumed to be simple identity (1-hot) encoding for simplicity
        x = torch.eye(len(edge_index), dtype=torch.float)

        return Data(x=x, edge_index=edge_index)

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for PyTorch Geometric's Data objects."""
        return Batch.from_data_list(batch)
