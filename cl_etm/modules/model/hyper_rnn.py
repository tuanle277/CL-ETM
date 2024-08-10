import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, global_mean_pool
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TemporalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super(TemporalEncoding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timestamps, first_timestamp):
        absolute_time = self._absolute_time_encoding(timestamps)
        relative_time = self._relative_time_encoding(timestamps, first_timestamp)
        positional_encoding = self._sinusoidal_encoding(timestamps)
        return absolute_time + relative_time + positional_encoding

    def _absolute_time_encoding(self, timestamps):
        return F.relu(timestamps.unsqueeze(-1).expand(-1, self.embedding_dim))

    def _relative_time_encoding(self, timestamps, first_timestamp):
        relative_time = timestamps - first_timestamp
        return F.relu(relative_time.unsqueeze(-1).expand(-1, self.embedding_dim))

    def _sinusoidal_encoding(self, timestamps):
        encoding = torch.zeros_like(timestamps).unsqueeze(-1).expand(-1, self.embedding_dim)
        for k in range(self.embedding_dim):
            if k % 2 == 0:
                encoding[:, k] = torch.sin(timestamps / (10000 ** (2 * k / self.embedding_dim)))
            else:
                encoding[:, k] = torch.cos(timestamps / (10000 ** (2 * k / self.embedding_dim)))
        return encoding

class RelationalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super(RelationalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.cause_embedding = nn.Parameter(torch.randn(embedding_dim))
        self.effect_embedding = nn.Parameter(torch.randn(embedding_dim))
        self.associated_embedding = nn.Parameter(torch.randn(embedding_dim))

    def forward(self, event_roles):
        Rijk = torch.zeros(len(event_roles), self.embedding_dim, device=event_roles.device)
        for i, role in enumerate(event_roles):
            if role == "cause":
                Rijk[i] += self.cause_embedding
            elif role == "effect":
                Rijk[i] += self.effect_embedding
            else:  # associated
                Rijk[i] += self.associated_embedding
        return Rijk

class HyperGNNModel(nn.Module):
    def __init__(
                self, 
                node_feature_dim, 
                hidden_dim, 
                rnn_hidden_dim, 
                embedding_dim, 
                num_heads=4, 
                temperature=0.5
            ):
        
        super(HyperGNNModel, self).__init__()

        # Hypergraph Convolutional Layer
        self.hypergraph_conv = HypergraphConv(
                                                in_channels=node_feature_dim, 
                                                out_channels=hidden_dim, 
                                                heads=num_heads, 
                                                use_attention=True
                                              )

        # Temporal Encoding Layer
        self.temporal_encoding = TemporalEncoding(hidden_dim)

        # RNN for Temporal Sequence Modeling (can be replaced with Transformer)
        self.rnn = nn.LSTM(
                                input_size=hidden_dim, 
                                hidden_size=rnn_hidden_dim, 
                                num_layers=2, 
                                batch_first=True, 
                                bidirectional=True
                           )

        # Causal Inference Layer
        self.causal_layer = nn.Linear(rnn_hidden_dim * 2, hidden_dim)  # bidirectional output

        # Relational Encoding Layer
        self.relational_encoding = RelationalEncoding(hidden_dim)

        # Embedding Layer to produce final patient embeddings
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)

        # Temperature for contrastive learning
        self.temperature = temperature

    def forward(
                self, 
                data: Batch, 
                timestamps, 
                first_timestamp=None, 
                event_roles=None
            ):
        
        # Step 1: Hypergraph Convolution
        x, _, hyperedge_index, batch = data.x, data.edge_index, data.hyperedges, data.batch
        x = F.relu(self.hypergraph_conv(x, hyperedge_index))

        # Step 2: Temporal Encoding
        P_t = self.temporal_encoding(timestamps, first_timestamp)
        x = x + P_t  # Augment the node embeddings with temporal encoding

        # Step 3: Pooling to get a dense representation for each patient
        x_pooled = global_mean_pool(x, batch)

        # Step 4: Temporal Sequence Modeling
        x_seq, seq_lengths = to_dense_batch(x_pooled, batch)
        x_seq_packed = pack_padded_sequence(x_seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        rnn_out, (_, _) = self.rnn(x_seq_packed)
        rnn_out_padded, _ = pad_packed_sequence(rnn_out, batch_first=True)

        # Step 5: Causal Inference
        causal_out = F.relu(self.causal_layer(rnn_out_padded))

        # Step 6: Relational Encoding
        Rijk = self.relational_encoding(event_roles)
        causal_out = causal_out + Rijk  # Augment the embeddings with relational encoding

        # Step 7: Generate Patient Embeddings
        patient_embeddings = self.embedding_layer(causal_out)

        return patient_embeddings

    def embed_nodes(self, data):
        # Return embeddings for all nodes, forward propagation
        return self.forward(data.x, data.edge_index)