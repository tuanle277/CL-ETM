import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, global_mean_pool
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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

class HierarchicalAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(HierarchicalAttention, self).__init__()
        self.attn_layer = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        attn_output, _ = self.attn_layer(x, x, x)
        return F.relu(self.fc(attn_output))

class HypergraphTransformer(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, num_heads, num_layers):
        super(HypergraphTransformer, self).__init__()
        self.graph_conv_layers = nn.ModuleList([
            GATConv(node_feature_dim if i == 0 else hidden_dim, hidden_dim, heads=num_heads) 
            for i in range(num_layers)
        ])
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(
                TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
                num_layers=1) for _ in range(num_layers)
        ])

        # Temporal and Relational Encoding
        self.temporal_encoding = TemporalEncoding(hidden_dim)
        self.relational_encoding = RelationalEncoding(hidden_dim)

    def forward(self, x, edge_index, timestamps, first_timestamp, event_roles):
        for conv, transformer in zip(self.graph_conv_layers, self.transformer_layers):
            x = conv(x, edge_index)
            x = x.unsqueeze(1)  # For transformer input shape (seq_len, batch, feature_dim)
            
            # Apply temporal and relational encoding
            P_t = self.temporal_encoding(timestamps, first_timestamp)
            Rijk = self.relational_encoding(event_roles)
            x = x + P_t + Rijk
            
            x = transformer(x)
            x = x.squeeze(1)
        return x

class MultiViewAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(MultiViewAttention, self).__init__()
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads) for _ in range(3)
        ])
        self.fc = nn.Linear(input_dim * 3, output_dim)

    def forward(self, x):
        attn_outputs = []
        for attn in self.attn_layers:
            attn_out, _ = attn(x, x, x)
            attn_outputs.append(attn_out)
        combined = torch.cat(attn_outputs, dim=-1)
        return F.relu(self.fc(combined))

class MetaLearnedEmbeddings(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MetaLearnedEmbeddings, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.meta_fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, meta_data):
        x = F.relu(self.fc(x))
        meta_embedding = F.relu(self.meta_fc(meta_data))
        return x + meta_embedding  # Adapt embeddings with meta-data

class PatientGraphTransformer(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, num_heads=4, num_layers=2, temperature=0.5):
        super(PatientGraphTransformer, self).__init__()

        # Hypergraph Convolutional Layer
        self.hypergraph_conv = HypergraphConv(in_channels=node_feature_dim, 
                                              out_channels=hidden_dim, 
                                              heads=num_heads, 
                                              use_attention=True)

        # Hypergraph Transformer with Temporal and Relational Encoding
        self.hypergraph_transformer = HypergraphTransformer(hidden_dim, hidden_dim, num_heads, num_layers)

        # Multi-View Attention for different hyperedges
        self.multi_view_attention = MultiViewAttention(hidden_dim, hidden_dim, num_heads)

        # Meta-Learned Embeddings
        self.meta_embeddings = MetaLearnedEmbeddings(hidden_dim, hidden_dim)

        # Hierarchical Attention Mechanism
        self.hierarchical_attention = HierarchicalAttention(hidden_dim, num_heads)

        # Temperature for contrastive learning
        self.temperature = temperature

        # Link Prediction Layer
        self.link_prediction_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, data: Batch, meta_data):
        x, edge_index, hyperedge_index, batch = data.x, data.edge_index, data.hyperedges, data.batch
        timestamps = data.timestamps
        first_timestamp = data.first_timestamp
        event_roles = data.event_roles

        # Step 1: Hypergraph Convolution
        x = F.relu(self.hypergraph_conv(x, hyperedge_index))

        # Step 2: Hypergraph Transformer with Temporal and Relational Encoding
        x = self.hypergraph_transformer(x, edge_index, timestamps, first_timestamp, event_roles)

        # Step 3: Multi-View Attention
        x = self.multi_view_attention(x)

        # Step 4: Meta-Learned Embeddings
        x = self.meta_embeddings(x, meta_data)

        # Step 5: Hierarchical Attention
        x = self.hierarchical_attention(x)

        return x

    def contrastive_loss(self, anchor, positive, negative):
        # Compute the dot products for anchor-positive and anchor-negative pairs
        pos_dot_product = torch.sum(anchor * positive, dim=-1)
        neg_dot_product = torch.sum(anchor * negative, dim=-1)

        # Compute the contrastive loss
        pos_term = torch.exp(pos_dot_product / self.temperature)
        neg_term = torch.exp(neg_dot_product / self.temperature)
        loss = -torch.log(pos_term / (pos_term + neg_term)).mean()

        return loss

    def loss(self, anchor_data, positive_data, negative_data, meta_data):
        anchor_embeddings = self.forward(anchor_data, meta_data)
        positive_embeddings = self.forward(positive_data, meta_data)
        negative_embeddings = self.forward(negative_data, meta_data)

        return self.contrastive_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
