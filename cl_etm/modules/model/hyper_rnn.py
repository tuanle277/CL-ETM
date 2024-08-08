import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, global_mean_pool
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TemporalCausalModel(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, rnn_hidden_dim, output_dim, num_heads=4):
        super(TemporalCausalModel, self).__init__()

        # Hypergraph Convolutional Layer
        self.hypergraph_conv = HypergraphConv(in_channels=node_feature_dim, 
                                              out_channels=hidden_dim, 
                                              heads=num_heads, 
                                              use_attention=True)
        
        # RNN for Temporal Sequence Modeling (can be replaced with Transformer)
        self.rnn = nn.LSTM(input_size=hidden_dim, 
                           hidden_size=rnn_hidden_dim, 
                           num_layers=2, 
                           batch_first=True, 
                           bidirectional=True)

        # Causal Inference Layer
        self.causal_layer = nn.Linear(rnn_hidden_dim * 2, hidden_dim)  # bidirectional output

        # Link Prediction Layer
        self.link_prediction_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, data: Batch):
        # Step 1: Hypergraph Convolution
        x, edge_index, hyperedge_index, batch = data.x, data.edge_index, data.hyperedges, data.batch
        x = F.relu(self.hypergraph_conv(x, hyperedge_index))

        # Step 2: Pooling to get a dense representation for each patient
        x_pooled = global_mean_pool(x, batch)

        # Step 3: Temporal Sequence Modeling
        x_seq, seq_lengths = to_dense_batch(x_pooled, batch)
        x_seq_packed = pack_padded_sequence(x_seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        rnn_out, (hn, cn) = self.rnn(x_seq_packed)
        rnn_out_padded, _ = pad_packed_sequence(rnn_out, batch_first=True)

        # Step 4: Causal Inference
        causal_out = F.relu(self.causal_layer(rnn_out_padded))

        # Step 5: Link Prediction
        link_pred = torch.sigmoid(self.link_prediction_layer(causal_out))

        return link_pred

    def predict_links(self, data: Batch):
        link_scores = self.forward(data)
        return link_scores

    def loss(self, data: Batch, target_links):
        link_scores = self.forward(data)
        loss = F.binary_cross_entropy(link_scores, target_links.float())
        return loss