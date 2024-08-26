# CL-ETM: Modeling Patient EHR Representation as Continuous Hypergraph for Enhancing Disease Detection

This repository proposes the codebase for research on modeling the patient EHR dataset MIMIC-IV as Continuous Hypergraphs and learning a link-prediction model between out-of-graph diseases and patients. 

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Scripts](#scripts)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This repository contains scripts for:
- Shortening MIMIC-IV CSV files by reducing the number of rows.
- Constructing patient graphs from the processed data using Polars and PyTorch Geometric.
- Saving and loading graphs for further analysis.

## Requirements

- Python 3.11 or higher

## Setup

- Clone the repository:

```bash
git clone https://github.com/tuanle277/cl_etm.git
cd cl_etm
```

- Install the required packages using `pip`:

```bash
pip install -e .
```

- Ensure your MIMIC-IV data is properly placed:

`Place the MIMIC-IV data in the ./data/MIMIC-IV/hosp and ./data/MIMIC-IV/icu directories.`

## Usage
### Shortening CSV Files
You can shorten the CSV files in the MIMIC-IV dataset by reducing the number of rows using the shorten.py script located in the `cl_etm/utils/` directory.

```bash
python .\cl_etm\utils\eda.py "./data/MIMIC-IV/icu" --num_rows 50
python .\cl_etm\utils\eda.py "./data/MIMIC-IV/hosp" --num_rows 50
```

+ path: The directory containing the original MIMIC-IV CSV files.
+ num_rows: The number of rows to keep in each shortened CSV file. If not specified, the entire file is processed.
  
The shortened files will be saved in a new directory at the same level as the original folder, named e.g. `MIMIC-IV-short/hosp`.

### Data Creation
The data.py script in the `cl_etm/modules/` directory is used for loading the MIMIC-IV data, creating patient graphs, and performing data transformations.

```bash
python cl_etm/modules/data.py --data_dir "./data/MIMIC-IV-short"
```

+ data_dir: Directory containing MIMIC-IV data. Defaults to `data/MIMIC-IV-short`.

This script processes the data, saves the generated graphs in `./data/graph_data/patient_graphs.pt`, and optionally prints details about a specific subject's graph.

### Data Loader 
The data_loader.py script in the `cl_etm/modules/` directory is used to create a DataLoader object with the MIMIC-IV data.

sample run
```bash
python cl_etm/modules/data_loader.py 
```

### Model
### Trainer

## Scripts
```markdown
cl_etm/
│
├── modules/                   # Core modules of your project
│   ├── __init__.py            # Makes the directory a Python package
│   ├── data_loader.py         # DataLoader implementations
│   ├── data_model/            # Subdirectory for data modeling
│   │   ├── __init__.py
│   │   ├── embedding.py       # Embedding features module
│   │   ├── intragraph.py      # Intra-patient hypergraph module
│   │   ├── intergraph.py      # Inter-patient hypergraph module
│   ├── model/                 # Model definitions
│   │   ├── __init__.py
│   │   ├── hyper_rnn.py       # HyperGNN model (and related models)
│   │   ├── transformer.py     # Hierarchical Graph Transformer model
│   ├── trainer.py             # Trainer class for model training
│   ├── node_splitter.py       # NodeSplitter class for splitting nodes into anchor/positive/negative
│
├── utils/                     # Utility functions and scripts
│   ├── __init__.py
│   ├── eda.py                 # Functions for exploratory data analysis, loading/saving data
│   ├── metrics.py             # Custom evaluation metrics
│   ├── config.py              # Configuration file for global constants
│
├── data/                      # Data directory (you might want to exclude this from version control)
│   ├── MIMIC-IV/              # Original dataset
│   ├── MIMIC-IV-short/        # Preprocessed/smaller dataset for quick tests
│   ├── graph_data/            # Directory to store processed graphs
│
├── notebooks/                 # Jupyter notebooks for experiments and EDA
├── scripts/                   # Standalone scripts for running experiments, tests, etc.
│   ├── __init__.py
│   ├── run_training.py        # Script to run training from the command line
│   ├── test.py                # Script for testing and debugging
│   ├── preprocess_data.py     # Script to preprocess the MIMIC-IV data
│
├── tests/                     # Unit tests and test scripts
│   ├── __init__.py
│   ├── test_data_loader.py    # Tests for data loader modules
│   ├── test_model.py          # Tests for model modules
│   ├── test_trainer.py        # Tests for the trainer
│
├── README.md                  # Project README with instructions, description, etc.
├── requirements.txt           # Python dependencies
├── setup.py                   # Setup script for packaging the project
├── .gitignore                 # Git ignore file
├── pyproject.toml             # Project metadata (optional, can be used instead of setup.py)
└── LICENSE                    # License file (if applicable)
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have improvements or new features you'd like to add.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

markdown

### Explanation:
- **Introduction**: Provides an overview of the project and its functionalities.
- **Directory Structure**: Outlines the structure of the project directories and files.
- **Requirements**: Lists the Python version and packages needed to run the scripts, along with installation instructions.
- **Setup**: Instructions on how to clone the repository and organize the data.
- **Usage**: Detailed explanations and commands for shortening CSV files, loading and processing data, performing EDA, and creating sample datasets.
- **Contributing**: Encourages contributions and explains how to get involved.
- **License**: Specifies the licensing information.
