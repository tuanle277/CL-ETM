import torch
from torch_geometric.data import Data

class NodeSplitter:
    def __init__(self, data: Data, positive_ratio=0.2, negative_ratio=0.2):
        """
        Initializes the NodeSplitter.

        :param data: The PyG Data object containing the graph.
        :param positive_ratio: The ratio of nodes to be selected as positive examples.
        :param negative_ratio: The ratio of nodes to be selected as negative examples.
        """
        self.data = data
        self.positive_ratio = positive_ratio
        self.negative_ratio = negative_ratio
        self.num_nodes = data.num_nodes

    def split_nodes(self):
        """
        Splits nodes into anchor, positive, and negative sets and creates corresponding masks.

        :return: Updated Data object with anchor_mask, positive_mask, and negative_mask.
        """
        # Generate random indices for anchor, positive, and negative nodes
        all_indices = torch.randperm(self.num_nodes)

        num_positive = int(self.num_nodes * self.positive_ratio)
        num_negative = int(self.num_nodes * self.negative_ratio)

        positive_indices = all_indices[:num_positive]
        negative_indices = all_indices[num_positive:num_positive + num_negative]
        anchor_indices = all_indices[num_positive + num_negative:]

        # Create masks
        anchor_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        positive_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        negative_mask = torch.zeros(self.num_nodes, dtype=torch.bool)

        anchor_mask[anchor_indices] = True
        positive_mask[positive_indices] = True
        negative_mask[negative_indices] = True

        # Add the masks to the Data object
        self.data.anchor_mask = anchor_mask
        self.data.positive_mask = positive_mask
        self.data.negative_mask = negative_mask

        return self.data
