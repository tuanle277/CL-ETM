import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, global_mean_pool
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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

class HierarchicalGraphTransformer(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, num_heads, num_layers):
        super(HierarchicalGraphTransformer, self).__init__()
        self.graph_conv_layers = nn.ModuleList([
            GATConv(node_feature_dim if i == 0 else hidden_dim, hidden_dim, heads=num_heads) 
            for i in range(num_layers)
        ])
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads), num_layers=num_layers)

    def forward(self, x, edge_index):
        for conv in self.graph_conv_layers:
            x = conv(x, edge_index)
        x = x.unsqueeze(1)  # For transformer input shape (seq_len, batch, feature_dim)
        x = self.transformer(x)
        return x.squeeze(1)

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
    def __init__(self, node_feature_dim, hidden_dim, output_dim, num_heads=4, num_layers=2):
        super(PatientGraphTransformer, self).__init__()

        # Hypergraph Convolutional Layer
        self.hypergraph_conv = HypergraphConv(in_channels=node_feature_dim, 
                                              out_channels=hidden_dim, 
                                              heads=num_heads, 
                                              use_attention=True)

        # Hierarchical Graph Transformer
        self.hierarchical_transformer = HierarchicalGraphTransformer(hidden_dim, hidden_dim, num_heads, num_layers)

        # Multi-View Attention for different hyperedges
        self.multi_view_attention = MultiViewAttention(hidden_dim, hidden_dim, num_heads)

        # Meta-Learned Embeddings
        self.meta_embeddings = MetaLearnedEmbeddings(hidden_dim, hidden_dim)

        # Link Prediction Layer
        self.link_prediction_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, data: Batch, meta_data):
        x, edge_index, hyperedge_index, batch = data.x, data.edge_index, data.hyperedges, data.batch

        # Step 1: Hypergraph Convolution
        x = F.relu(self.hypergraph_conv(x, hyperedge_index))

        # Step 2: Hierarchical Graph Transformer
        x = self.hierarchical_transformer(x, edge_index)

        # Step 3: Multi-View Attention
        x = self.multi_view_attention(x)

        # Step 4: Meta-Learned Embeddings
        x = self.meta_embeddings(x, meta_data)

        # Step 5: Pooling and Link Prediction
        x_pooled = global_mean_pool(x, batch)
        link_pred = torch.sigmoid(self.link_prediction_layer(x_pooled))

        return link_pred

    def loss(self, data: Batch, meta_data, target_links):
        link_scores = self.forward(data, meta_data)
        loss = F.binary_cross_entropy(link_scores, target_links.float())
        return loss