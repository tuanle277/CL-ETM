from cl_etm.modules.data_loader import IntraPatientDataLoader
from cl_etm.modules.model.hyper_rnn import HyperGNNModel
from cl_etm.modules.trainer import Trainer
from cl_etm.utils.eda import load_all_graphs

import torch

def test_intra_patient_data_loader():
    try:
        # Load data
        print("Loading patient graphs...")
        patient_graphs = load_all_graphs()

        # Check if graphs loaded correctly
        assert len(patient_graphs) > 0, "No patient graphs were loaded."
        print(f"Loaded {len(patient_graphs)} patient graphs.")

        # Create data loaders for the first patient graph
        print("Creating data loaders for the first patient graph...")
        intra_patient_loader = IntraPatientDataLoader(patient_graphs[list(patient_graphs.keys())[0]])
        train_loader, val_loader, test_loader = intra_patient_loader.create_dataloaders()

        # Test if data loaders are created correctly
        assert len(train_loader) > 0, "Train loader is empty."
        assert len(val_loader) > 0, "Validation loader is empty."
        assert len(test_loader) > 0, "Test loader is empty."
        print("Data loaders created successfully.")

        # Test iterating through the train loader
        print("Testing iteration through the train loader...")
        for batch in train_loader:
            print(f"Train batch: {batch}")
            break  # Just to test one batch

    except Exception as e:
        print(f"Test failed: {e}")
        raise e

def test_model_initialization():
    try:
        # Define model parameters
        node_feature_dim = 128
        hidden_dim = 64
        rnn_hidden_dim = 64
        output_dim = 1  # Binary link prediction

        # Initialize the model
        print("Initializing the HyperGNNModel...")
        model = HyperGNNModel(node_feature_dim, hidden_dim, rnn_hidden_dim, output_dim)
        print("Model initialized successfully.")

        # Check if the model is on the correct device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        # assert model.device == device, "Model is not on the correct device."
        print(f"Model is on device: {device}")

    except Exception as e:
        print(f"Test failed: {e}")
        raise e

def test_trainer():
    try:
        # Load data and model
        print("Loading patient graphs...")
        patient_graphs = load_all_graphs()
        print("Creating data loaders...")
        train_loader, val_loader, test_loader = IntraPatientDataLoader(patient_graphs[list(patient_graphs.keys())[0]]).create_dataloaders()

        print(train_loader.data)
        print(val_loader.data)
        print(test_loader.data)

        # Initialize the model
        print("Initializing the model...")
        node_feature_dim = 128
        hidden_dim = 64
        rnn_hidden_dim = 64
        output_dim = 1  # Binary link prediction
        model = HyperGNNModel(node_feature_dim, hidden_dim, rnn_hidden_dim, output_dim)

        # Set device to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Initialize the trainer with model, loaders, and device
        print("Initializing the trainer...")
        trainer = Trainer(model, train_loader, val_loader, test_loader, device)

        # Train the model for a few epochs to test
        print("Training the model...")
        trainer.fit(epochs=5)  # Reduced to 5 epochs for testing
        print("Training completed.")

        # Test evaluation
        print("Evaluating the model on test data...")
        test_loss, test_accuracy = trainer.test()
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    except Exception as e:
        print(f"Test failed: {e}")
        raise e

if __name__ == "__main__":
    # Run all tests
    print("Running IntraPatientDataLoader test...")
    test_intra_patient_data_loader()

    print("Running model initialization test...")
    test_model_initialization()

    print("Running trainer test...")
    test_trainer()

    print("All tests completed successfully.")
