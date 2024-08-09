
from cl_etm.modules.data_loader import IntraPatientDataLoader, InterPatientDataLoader
from cl_etm.modules.model.hyper_rnn import HyperGNNModel
from cl_etm.modules.trainer import Trainer

from cl_etm.utils.eda import load_all_graphs

import torch 

if __name__ == "__main__":

    # Load data 
    patient_graphs = load_all_graphs()

    # Create data loaders
    train_loader, val_loader, test_loader = InterPatientDataLoader(patient_graphs).create_dataloaders()

    # Define model parameters
    node_feature_dim = 128
    hidden_dim = 64
    rnn_hidden_dim = 64
    output_dim = 1  # Binary link prediction

    # Initialize the model
    model = HyperGNNModel(node_feature_dim, hidden_dim, rnn_hidden_dim, output_dim)

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the trainer with model, loaders, and device
    trainer = Trainer(model, train_loader, val_loader, test_loader, device)

    # Train the model
    trainer.fit(epochs=100)

    # Test the model
    trainer.test()