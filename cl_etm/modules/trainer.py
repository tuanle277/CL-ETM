import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
            
            # Assuming the data contains anchor, positive, and negative pairs
            anchor_data, positive_data, negative_data = data.anchor, data.positive, data.negative
            meta_data = data.meta_data

            # Calculate contrastive loss
            loss = self.model.loss(anchor_data, positive_data, negative_data, meta_data)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                
                # Assuming the data contains anchor, positive, and negative pairs
                anchor_data, positive_data, negative_data = data.anchor, data.positive, data.negative
                meta_data = data.meta_data

                # Calculate contrastive loss
                loss = self.model.loss(anchor_data, positive_data, negative_data, meta_data)
                total_loss += loss.item()

        return total_loss / len(loader)

    def fit(self, epochs):
        best_val_loss = float('inf')

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.evaluate(self.val_loader)

            print(f'Epoch {epoch + 1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

            self.scheduler.step(val_loss)

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')

    def test(self):
        test_loss = self.evaluate(self.test_loader)
        print(f'Test Loss: {test_loss:.4f}')
        return test_loss
