import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from cl_etm.utils.eda import load_all_graphs

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, lr=0.001, weight_decay=0.0001):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=10, verbose=True)

        self.model = self.model.to(self.device)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.loss(data, data.target_links)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                loss = self.model.loss(data, data.target_links)
                total_loss += loss.item()

                pred = self.model.predict_links(data)
                correct += ((pred > 0.5).float() == data.target_links.float()).sum().item()
                total += data.target_links.numel()

        accuracy = correct / total
        return total_loss / len(loader), accuracy

    def fit(self, epochs):
        best_val_loss = float('inf')

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_accuracy = self.evaluate(self.val_loader)

            print(f'Epoch {epoch + 1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

            self.scheduler.step(val_loss)

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')

    def test(self):
        test_loss, test_accuracy = self.evaluate(self.test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        return test_loss, test_accuracy
    
# Sample run using the trainer

# if __name__ == "__main__":
    # Assuming you have prepared data loaders
    # train_loader = IntraPatientDataLoader(...)
    # val_loader = IntraPatientDataLoader(...)
    # test_loader = IntraPatientDataLoader(...)

    # node_feature_dim = 128
    # hidden_dim = 64
    # rnn_hidden_dim = 64
    # output_dim = 1  # Binary link prediction

    # model = TemporalCausalModel(node_feature_dim, hidden_dim, rnn_hidden_dim, output_dim)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # trainer = Trainer(model, train_loader, val_loader, test_loader, device)

    # # Training the model
    # trainer.fit(epochs=100)

    # # Testing the model
    # trainer.test()