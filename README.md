# CL-ETM: Modeling Patient EHR Representation as Continuous Hypergraph for Enhancing Disease Detection

This repository provides___

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

You can install the required packages using `pip`:

```bash
pip install -e .
```

## Setup

- Clone the repository:

```bash
git clone https://github.com/tuanle277/cl_etm.git
cd cl_etm
```
Ensure your MIMIC-IV data is properly placed:

Place the MIMIC-IV data in the ./data/MIMIC-IV/hosp and ./data/MIMIC-IV/icu directories.

## Usage
### Shortening CSV Files
You can shorten the CSV files in the MIMIC-IV dataset by reducing the number of rows using the shorten.py script located in the cl_etm/utils/ directory.

```bash
python cl_etm/utils/shorten.py "./data/MIMIC-IV/hosp" --num_rows 5000
```

+ path: The directory containing the original MIMIC-IV CSV files.
+ num_rows: The number of rows to keep in each shortened CSV file. If not specified, the entire file is processed.
  
The shortened files will be saved in a new directory at the same level as the original folder, named MIMIC-IV-short/hosp.

### Data Creation
The data.py script in the cl_etm/modules/ directory is used for loading the MIMIC-IV data, creating patient graphs, and performing data transformations.

```bash
python cl_etm/modules/data.py --data_dir "./data/MIMIC-IV-short" --save_path "./data/graph_data/graphs.pt" --subject_id 10058834
```

+ data_dir: Directory containing MIMIC-IV data. Defaults to data/MIMIC-IV-short.
+ save_path: Path to save the generated graphs. Defaults to data/graph_data/patient_graphs.pt.
+ subject_id: (Optional) Specific subject ID to inspect after processing.

This script processes the data, saves the generated graphs, and optionally prints details about a specific subject's graph.

### Data Loader 
The data_loader.py  script in the cl_etm/modules/ directory is used to create DataLoader object with the MIMIC-IV data.

sample run
```bash
python cl_etm/modules/data_loader.py 
```

### Model
### Trainer

## Scripts
```markdown
CL_ETM/
├── cl_etm/
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── trainer.py
│   │   └── data.py
│   └── utils/
│       ├── __init__.py
│       ├── create_sample_data.py
│       ├── eda.py
│       └── misc.py
├── data/
│   ├── graph_data/
│   └── MIMIC-IV/
│       ├── hosp/
│       └── icu/
├── MIMIC-IV-short/
├── .gitignore
├── notes.txt
├── pyproject.toml
└── README.md
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
