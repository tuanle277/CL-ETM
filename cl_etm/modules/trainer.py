import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import HypergraphConv
from tqdm import tqdm
from cl_etm.utils.eda import load_all_graphs

class HypergraphNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_attention=False, heads=1):
        super(HypergraphNet, self).__init__()
        self.conv1 = HypergraphConv(input_dim, hidden_dim, use_attention=use_attention, heads=heads)
        self.conv2 = HypergraphConv(hidden_dim * heads if heads > 1 else hidden_dim, output_dim, use_attention=use_attention, heads=heads)

    def forward(self, x, hyperedge_index, hyperedge_weight=None, hyperedge_attr=None):
        x = self.conv1(x, hyperedge_index, hyperedge_weight, hyperedge_attr)
        x = torch.relu(x)
        x = self.conv2(x, hyperedge_index, hyperedge_weight, hyperedge_attr)
        return x

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, num_epochs=10):
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index)  # Assuming edge_index represents hyperedge_index
                loss = self.criterion(out, batch.y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1} Loss: {epoch_loss / len(self.train_loader)}")

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index)
                _, predicted = torch.max(out.data, 1)
                total += batch.y.size(0)
                correct += (predicted == batch.y).sum().item()
        return correct / total

    def test(self):
        test_acc = self.evaluate(self.test_loader)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Sample run using the trainer

if __name__ == "__main__":
    from data_loader import MIMICIVDataLoader

    graph_data = load_all_graphs("./data/graph_data/patient_graphs.pt")

    # Initialize data loader with patient graphs
    mimic_data_loader = MIMICIVDataLoader(graph_data)
    train_loader, val_loader, test_loader = mimic_data_loader.create_dataloaders()

    # Model initialization
    input_dim = 1  # Assumes all graphs have the same input_dim
    hidden_dim = 64
    output_dim = 2  # Example: binary classification
    model = HypergraphNet(input_dim, hidden_dim, output_dim, use_attention=True, heads=4)

    # Initialize the trainer and start training
    trainer = Trainer(model, train_loader, val_loader, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
    trainer.train(num_epochs=20)
    trainer.test()
