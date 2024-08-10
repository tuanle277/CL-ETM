import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, lr=0.001, weight_decay=0.0001):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=10)
        self.model = self.model.to(self.device)

    def contrastive_loss(self, anchor, positive, negative, margin=1.0):
        """
        Computes contrastive loss for triplets of nodes.
        
        :param anchor: Embedding for anchor nodes.
        :param positive: Embedding for positive nodes.
        :param negative: Embedding for negative nodes.
        :param margin: Margin for contrastive loss.
        :return: Loss value.
        """
        positive_distance = F.pairwise_distance(anchor, positive)
        negative_distance = F.pairwise_distance(anchor, negative)
        loss = torch.mean(F.relu(positive_distance - negative_distance + margin))
        return loss

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Get embeddings for anchor, positive, and negative nodes
            anchor_emb = self.model.embed_nodes(batch)[batch.anchor_mask]
            positive_emb = self.model.embed_nodes(batch)[batch.positive_mask]
            negative_emb = self.model.embed_nodes(batch)[batch.negative_mask]

            # Compute contrastive loss
            loss = self.contrastive_loss(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                # Get embeddings for anchor, positive, and negative nodes
                anchor_emb = self.model.embed_nodes(batch)[batch.anchor_mask]
                positive_emb = self.model.embed_nodes(batch)[batch.positive_mask]
                negative_emb = self.model.embed_nodes(batch)[batch.negative_mask]

                # Compute contrastive loss
                loss = self.contrastive_loss(anchor_emb, positive_emb, negative_emb)
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
                torch.save(self.model.state_dict(), f'{epochs}_model.pth')

    def test(self):
        test_loss = self.evaluate(self.test_loader)
        print(f'Test Loss: {test_loss:.4f}')
        return test_loss
