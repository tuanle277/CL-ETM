import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit

from cl_etm.modules.node_splitter import NodeSplitter

class IntraPatientDataLoader:
    def __init__(self, patient_graph, num_neighbors=10, batch_size=32, validation_split=0.1, test_split=0.1):
        self.patient_graph = patient_graph  # Single patient's continuous EHR hypergraph
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split

        # Split nodes into anchor, positive, and negative sets
        splitter = NodeSplitter(self.patient_graph)
        self.patient_graph = splitter.split_nodes()

        # Create the actual DataLoaders
        self.train_loader, self.val_loader, self.test_loader = self.create_dataloaders()

    def create_dataloaders(self):
        # Define the transformation with RandomLinkSplit
        transform = RandomLinkSplit(
            num_val=self.validation_split, 
            num_test=self.test_split,
            is_undirected=True,  # Assuming undirected graph, set False if directed
            add_negative_train_samples=True  # Set to False if you don't want to add negative samples to the train set
        )

        train_data, val_data, test_data = transform(self.patient_graph)

        # Create NeighborLoaders for each split
        train_loader = NeighborLoader(train_data,
                                      num_neighbors=[self.num_neighbors],
                                      batch_size=self.batch_size, shuffle=True)
        val_loader = NeighborLoader(val_data,
                                    num_neighbors=[self.num_neighbors],
                                    batch_size=self.batch_size, shuffle=False)
        test_loader = NeighborLoader(test_data,
                                     num_neighbors=[self.num_neighbors],
                                     batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader


class InterPatientDataLoader:
    def __init__(self, hyperedge_data, num_neighbors=10, batch_size=32, validation_split=0.1, test_split=0.1):
        self.hyperedge_data = hyperedge_data  # Hyperedge data between patients
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split

        # Merge individual HeteroData objects into a single HeteroData object
        self.full_hetero_data = self._merge_hetero_data()

        # Create the actual DataLoaders
        self.train_loader, self.val_loader, self.test_loader = self.create_dataloaders()

    def _merge_hetero_data(self):
        """Merge all patient HeteroData objects into a single HeteroData object."""
        full_hetero_data = HeteroData()

        for i, (_, hetero_data) in enumerate(self.hyperedge_data.items()):
            # Assign a unique patient index to each node
            print(hetero_data)
            for node_type in hetero_data.node_types:
                hetero_data[node_type].batch = torch.full(
                    (hetero_data[node_type].num_nodes,), i, dtype=torch.long
                )
            full_hetero_data = full_hetero_data + hetero_data

        return full_hetero_data

    def create_dataloaders(self):
        """Creates dataloaders for training, validation, and testing."""
        num_patients = len(self.hyperedge_data)
        num_val = int(num_patients * self.validation_split)
        num_test = int(num_patients * self.test_split)

        torch.manual_seed(42)
        shuffled_keys = torch.randperm(num_patients).tolist()

        # Create train, validation, and test splits based on the patient indices
        train_idx = shuffled_keys[:-num_val-num_test]
        val_idx = shuffled_keys[-num_val-num_test:-num_test]
        test_idx = shuffled_keys[-num_test:]

        train_loader = NeighborLoader(
            self.full_hetero_data,
            num_neighbors=[self.num_neighbors],
            input_nodes=("patient", train_idx),
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = NeighborLoader(
            self.full_hetero_data,
            num_neighbors=[self.num_neighbors],
            input_nodes=("patient", val_idx),
            batch_size=self.batch_size,
            shuffle=False,
        )

        test_loader = NeighborLoader(
            self.full_hetero_data,
            num_neighbors=[self.num_neighbors],
            input_nodes=("patient", test_idx),
            batch_size=self.batch_size,
            shuffle=False,
        )

        return train_loader, val_loader, test_loader

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader
