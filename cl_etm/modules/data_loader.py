import torch
import random
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Batch, HeteroData


class IntraPatientDataLoader:
    def __init__(self, graph_data_list, num_neighbors=10, batch_size=32, validation_split=0.1, test_split=0.1):
        self.graph_data_list = graph_data_list
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split

        # Create the actual DataLoaders
        self.train_loader, self.val_loader, self.test_loader = self.create_dataloaders()

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

        train_loader = NeighborLoader(self._create_contrastive_dataset(train_keys),
                                      num_neighbors=[self.num_neighbors],
                                      batch_size=self.batch_size, shuffle=True, input_nodes=None)
        val_loader = NeighborLoader(self._create_contrastive_dataset(val_keys),
                                    num_neighbors=[self.num_neighbors],
                                    batch_size=self.batch_size, shuffle=False, input_nodes=None)
        test_loader = NeighborLoader(self._create_contrastive_dataset(test_keys),
                                     num_neighbors=[self.num_neighbors],
                                     batch_size=self.batch_size, shuffle=False, input_nodes=None)

        return train_loader, val_loader, test_loader

    def _create_contrastive_dataset(self, keys):
        """Create a dataset of anchor-positive-negative triplets for contrastive learning."""
        triplets = []
        for key in keys:
            anchor = self.graph_data_list[key]

            # Create positive example (same patient, different time/condition)
            positive = self._create_positive_example(anchor, key)

            # Create negative example (different patient)
            negative = self._create_negative_example(key)

            triplets.append((anchor, positive, negative))

        print(triplets)

        return triplets

    def _create_positive_example(self, anchor, key):
        """Create a positive example based on the same patient (but different event)."""
        # Here, you could select a different time step, different feature, etc.
        positive = self.graph_data_list[key].clone()  # Simple clone for example purposes
        return positive

    def _create_negative_example(self, key):
        """Create a negative example from a different patient."""
        available_keys = list(self.graph_data_list.keys())
        available_keys.remove(key)
        negative_key = random.choice(available_keys)
        negative = self.graph_data_list[negative_key]
        return negative

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
