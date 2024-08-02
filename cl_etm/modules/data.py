import torch
import pandas as pd
from datetime import timedelta
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
import dgl
import logging
import numpy as np
from torch.utils.data import DataLoader
from cl_etm.utils.misc import display_dglgraph_info
from typing import List, Dict
import dask.dataframe as dd 
from tqdm import tqdm 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MIMICIVDataModule:
    def __init__(self, data_dir='data', time_window=7):
        self.data_dir = data_dir
        self.time_window = time_window
        self.mimic_data = None
        self.patient_graphs = None
        self.patient_disease_graph = None

    def load_data(self, chunksize: int = 1000000, timestamp_col_map: Dict[str, List[str]] = None) -> tuple:
        """Loads and preprocesses MIMIC-IV tables efficiently with Dask."""

        icu_tables = ["chartevents", "d_items", "datetimeevents", "icustays", "ingredientevents", "inputevents", "outputevents", "procedureevents"]
        hosp_tables = ["patients", "diagnoses_icd", "labevents", "microbiologyevents"]

        if timestamp_col_map is None:
            timestamp_col_map = {
                'admissions': ['admittime', 'dischtime', 'deathtime'],
                'prescriptions': ['starttime', 'stoptime'],
                'labevents': ['charttime'],
                'microbiologyevents': ['charttime'],
                'chartevents': ['charttime'],
                'inputevents': ['starttime', 'endtime'],
                'outputevents': ['charttime'],
                'procedureevents': ['starttime', 'endtime'],
            }

        # Load all tables with Dask, using lazy evaluation, and handle missing values with assume_missing=True
        data = {
            table: dd.read_csv(f"{self.data_dir}/icu/{table}.csv" if table in icu_tables
                            else f"{self.data_dir}/hosp/{table}.csv", assume_missing=True)
            for table in icu_tables + hosp_tables
        }

        # Load prescriptions with specific dtypes
        data["prescriptions"] = dd.read_csv(
            f"{self.data_dir}/hosp/prescriptions.csv",
            dtype={
                "dose_val_rx": "object",
                "form_rx": "object",
                "form_val_disp": "object",
                "gsn": "object",
                "ndc": "float64",
                "poe_seq": "float64",
            },
            assume_missing=True
        )
        # Load admissions with specific dtypes
        data["admissions"] = dd.read_csv(
            f"{self.data_dir}/hosp/admissions.csv",
            dtype={
                "deathtime": "object",
            },
        )
        # Convert date columns to datetime
        for table, columns in timestamp_col_map.items():
            for column in columns:
                if column != 'deathtime':
                    data[table][column] = dd.to_datetime(data[table][column], errors='coerce')
                else:
                    # Convert deathtime to datetime separately
                    data['admissions'][column] = dd.to_datetime(data['admissions'][column], errors='coerce', format="%Y-%m-%d %H:%M:%S")  

        # Label event types
        event_types = {
            'diagnoses_icd': 'diagnosis',
            'procedureevents': 'procedure',
            'prescriptions': 'medication',
            'labevents': 'lab',
            'microbiologyevents': 'microbiology',
            'chartevents': 'chart',
            'inputevents': 'input',
            'outputevents': 'output',
        }

        # Initialize empty dataframes for results
        patients_df = pd.DataFrame()
        admissions_df = pd.DataFrame()
        events_df = pd.DataFrame()

        # Process each event table incrementally
        with tqdm(total=len(event_types), desc="Loading tables") as pbar_tables:
            for table, event_type in event_types.items():

                # Filter patients and admissions dataframes for each table
                # Load unique patients and admissions in chunks
                unique_patient_ids = set()
                unique_admission_ids = set()
                for chunk in data[table].to_delayed():
                    chunk = chunk.compute()
                    unique_patient_ids.update(chunk['subject_id'].unique())
                    unique_admission_ids.update(chunk['hadm_id'].unique())

                # Filter patients and admissions based on the current table
                patients_table = data['patients'].loc[
                    data['patients']['subject_id'].isin(unique_patient_ids)
                ].compute()
                admissions_table = data['admissions'].loc[
                    data['admissions']['hadm_id'].isin(unique_admission_ids)
                ].compute()

                # Process events for the current table in chunks
                all_events_dfs = []
                for chunk in tqdm(data[table].to_delayed(), desc=f"Loading {table}", leave=False):
                    chunk = chunk.compute()
                    chunk['event_type'] = event_type
                    # Handle timestamp column
                    timestamp_column = next((col for col in ["charttime", "starttime", "endtime"] if col in chunk.columns), None)
                    if timestamp_column is not None:
                        # Rename the timestamp column to 'eventtime' after extraction
                        all_events_dfs.append(
                            chunk[["subject_id", "hadm_id", "event_type"] +
                                [col for col in chunk.columns if
                                col != "subject_id" and col != "hadm_id" and col != "event_type"]]
                                .rename(columns={timestamp_column: "eventtime"})
                        )
                    else:
                        logging.warning(f"No relevant timestamp column found for table '{table}'")

                        # If no relevant timestamp column, still include the chunk but set eventtime to NaN.
                        chunk["eventtime"] = pd.NaT
                        all_events_dfs.append(
                            chunk[["subject_id", "hadm_id", "event_type"] +
                                [col for col in chunk.columns if
                                col != "subject_id" and col != "hadm_id" and col != "event_type"]]
                        )
                # Concatenate the events for the current table
                events_table = pd.concat(all_events_dfs)

                # Process the table data 
                patients_df = pd.concat([patients_df, patients_table], ignore_index=True)
                admissions_df = pd.concat([admissions_df, admissions_table], ignore_index=True)
                events_df = pd.concat([events_df, events_table], ignore_index=True)

                pbar_tables.update(1)

            # Drop duplicates from the result dataframes
            patients_df.drop_duplicates(subset='subject_id', inplace=True)
            admissions_df.drop_duplicates(subset='hadm_id', inplace=True)

            return patients_df, admissions_df, events_df
    
        # Extract events using Dask delayed function to avoid intermediate DataFrames
        # events_delayed = [
        #     data[table].assign(event_type=event_type).map_partitions(
        #         lambda df: df[['subject_id', 'hadm_id', 'event_type'] +
        #                     [col for col in df.columns
        #                         if col not in ['subject_id', 'hadm_id', 'event_type']]]
        #     )
        #     for table, event_type in event_types.items()
        # ]
        # events = dd.concat(events_delayed).compute()  # Trigger computation for events DataFrame

        # # Handle timestamp columns
        # for table, columns in timestamp_col_map.items():
        #     timestamp_column = next((col for col in ['charttime', 'starttime', 'endtime'] if col in columns), None)
        #     if timestamp_column:
        #         events.loc[events['event_type'] == event_types.get(table), 'eventtime'] = data[table][timestamp_column].compute() 

        # return data['patients'].compute(), data['admissions'].compute(), events  # compute at the end for the needed dataframes


    def construct_patient_hypergraphs(self):
        """Construct heterogeneous DGL graphs for each patient's medical history."""
        self.patient_graphs = {}
        patients, admissions, events = self.mimic_data

        # Encode categorical features
        encoders = {
            'icd_code': LabelEncoder(),
            'drug': LabelEncoder(),
            'itemid': LabelEncoder()
        }

        for feature in encoders:
            if events[feature].dtype == 'object' or events[feature].dtype == 'string':  # Assuming string columns are of dtype 'object'
                events[feature] = encoders[feature].fit_transform(events[feature].fillna('0'))
            else:  # Numeric columns
                events[feature] = encoders[feature].fit_transform(events[feature].fillna(0))


        for patient_id in tqdm(patients['subject_id'].unique(), desc="Creating patient graphs"):
            patient_events = events[events['subject_id'] == patient_id].sort_values('eventtime')
            g = create_heterograph_with_events(patient_events)

            # Add diagnosis cooccurrence hyperedges
            g = create_diagnosis_cooccurrence_hypergraph(patient_events, g)

            # Add medication combination hyperedges
            g = create_medication_combination_hypergraph(patient_events, g)

            self.patient_graphs[patient_id] = g

        logger.info("Graph details:")
        display_dglgraph_info(self.patient_graphs)

    def construct_patient_disease_graph(self):
        """Construct a bipartite patient-disease graph."""
        patients, _, events = self.mimic_data
        patient_ids = patients['subject_id'].unique()
        disease_codes = events[events['event_type'] == 'diagnosis']['icd_code'].unique()

        patient_index = {p: i for i, p in enumerate(patient_ids)}
        disease_index = {d: i for i, d in enumerate(disease_codes)}

        g = dgl.heterograph({
            ('patient', 'has_disease', 'disease'): ([], []),
            ('disease', 'disease_of', 'patient'): ([], [])
        })

        g.add_nodes(len(patient_ids), ntype='patient')
        g.add_nodes(len(disease_codes), ntype='disease')

        edges = [
            (patient_index[row['subject_id']], disease_index[row['icd_code']]) 
            for _, row in events[events['event_type'] == 'diagnosis'].iterrows()
        ]

        src, dst = zip(*edges)
        g.add_edges(src, dst, etype='has_disease')
        g.add_edges(dst, src, etype='disease_of')

        patient_feats = torch.tensor([
            [patients[patients['subject_id'] == pid].iloc[0]['gender'] == 'M'] 
            for pid in patient_ids
        ], dtype=torch.float32)

        g.nodes['patient'].data['feat'] = patient_feats
        g.nodes['disease'].data['feat'] = torch.zeros((len(disease_codes), 1), dtype=torch.float32)

        self.patient_disease_graph = g

    def prepare_data(self):
        """Load data and create patient graphs."""
        self.mimic_data = self.load_data()
        self.construct_patient_hypergraphs()
        self.construct_patient_disease_graph()

    def train_dataloader(self, graph_type):
        """Create a DataLoader for training."""
        graph_dict = {
            'coocurrence': self.patient_graphs_coocurrence_hypergraph,
            'medication': self.patient_graphs_medication_hypergraph,
            'misc': self.patient_graphs_misc_hypergraph
        }
        graphs = graph_dict.get(graph_type, self.patient_graphs)
        return DataLoader(list(graphs.values()), batch_size=self.batch_size, shuffle=True)

def create_heterograph_with_events(patient_events):
    # """Create a heterogeneous DGL graph with event nodes and temporal edges."""
    # graph_data = {  # Define node and edge types
    #     ('patient', f'has_{et}', et): ([], []) for et in ['diagnosis', 'procedure', 'medication', 'lab', 
    #                                                       'microbiology', 'chart', 'input', 'output', 'procedure_event']
    # }

    # for event_type in graph_data.keys():
    #     event_df = patient_events[patient_events['event_type'] == event_type[2]]
    #     if not event_df.empty:
    #         event_ids = torch.arange(len(event_df))
    #         graph_data[event_type][0].extend([0] * len(event_df))  # Single patient node connection
    #         graph_data[event_type][1].extend(event_ids)
    
    # g = dgl.heterograph(graph_data)
    # return g
    """
    Creates a heterogeneous graph with event nodes and temporal edges.
    """

    # Define node types
    event_types = ['diagnosis', 'procedure', 'medication', 'lab', 'microbiology', 
                'chart', 'input', 'output', 'procedure_event']
    # Create graph_data 
    graph_data = {
        ('patient', f'has_{et}', et): ([], []) for et in event_types
    }
    graph_data.update({
        (et, f'{et}_patient', 'patient'): ([], []) for et in event_types
    })
    graph_data.update({
        ('event', 'occurs_after', 'event'): ([], []),
        ('event', 'cooccurs_in_admission_with', 'event'): ([], []),
        ('medication', 'cooccurs_with', 'medication'): ([], [])
    })

    # Add nodes and edges for each event type
    for event_type in ['diagnosis', 'procedure', 'medication', 'lab', 'microbiology', 'chart', 'input', 'output', 'procedure_event']:
        event_df = patient_events[patient_events['event_type'] == event_type]
        if not event_df.empty:
            event_ids = torch.arange(len(event_df))
            graph_data[('patient', f'has_{event_type}', event_type)][0].extend([0] * len(event_df))  # Connect to single patient node
            graph_data[('patient', f'has_{event_type}', event_type)][1].extend(event_ids)
            graph_data[(event_type, f'{event_type}_patient', 'patient')][0].extend(event_ids)  # Reverse edge for bi-directional
            graph_data[(event_type, f'{event_type}_patient', 'patient')][1].extend([0] * len(event_df))  # Connect to single patient node
    

            g = dgl.heterograph(graph_data)

            # for event_type in ['diagnosis', 'procedure', 'medication', 'lab', 'microbiology', 'chart', 'input', 'output', 'procedure_event']:

            # Add node features
            if event_type in ['diagnosis', 'procedure']:
                g.nodes[event_type].data['feat'] = torch.from_numpy(event_df['icd_code'].values).long()
            elif event_type == 'medication':
                g.nodes[event_type].data['feat'] = torch.from_numpy(event_df['drug'].values).long()
            elif event_type == 'lab':
                scaler = StandardScaler()
                g.nodes[event_type].data['feat'] = torch.from_numpy(scaler.fit_transform(event_df[['itemid', 'valuenum']].fillna(0).values)).float()
            # elif event_type == 'microbiology':
            #     g.nodes[event_type].data['feat'] = torch.from_numpy(le_item.transform(event_df['org_itemid'].fillna('0').values)).long()
            else:
                g.nodes[event_type].data['feat'] = torch.from_numpy(event_df['itemid'].values).long()
            # Add timestamp feature (normalize it to [0, 1] for better model performance)
            # Calculate time difference in days
            min_eventtime = event_df['eventtime'].min().to_datetime64()  # Convert to datetime64
            time_diffs = event_df['eventtime'].values - min_eventtime
            time_diffs_days = time_diffs / np.timedelta64(1, 'D')
            normalized_time_diffs = time_diffs_days / time_diffs_days.max()

            g.nodes[event_type].data['timestamp'] = torch.from_numpy(normalized_time_diffs).float()

    # Create temporal edges
    for event_type in ['diagnosis', 'procedure', 'medication', 'lab', 'microbiology', 'chart', 'input', 'output', 'procedure_event']:
        event_df = patient_events[patient_events['event_type'] == event_type]
        if not event_df.empty:
            event_ids = torch.arange(len(event_df))
            
            event_times = event_df['eventtime'].values
            for i in range(len(event_times) - 1):
                graph_data[('event', 'occurs_after', 'event')][0].append(event_ids[i])
                graph_data[('event', 'occurs_after', 'event')][1].append(event_ids[i + 1])


    g = dgl.heterograph(graph_data)

    # patient_data = g[g['subject_id'] == patient_id].iloc[0]
    # Split anchor_year_group into start and end year
    # anchor_year_start, _ = patient_data['anchor_year_group'].split(' - ')

    # Calculate real year of birth (anchor_year_start - anchor_age)
    # patient_data['dob'] = int(anchor_year_start) - patient_data['anchor_age']
    # g.nodes['patient'].data['feat'] = torch.tensor([[patient_data['gender'] == 'M', 2024 - pd.to_datetime(patient_data['dob']).year]], dtype=torch.float32)

    return g

def create_diagnosis_cooccurrence_hypergraph(patient_events, g):
    """Create hyperedges based on co-occurrence of diagnoses and other events."""
    admission_groups = patient_events.groupby('hadm_id')
    edges = []

    for _, group in admission_groups:
        diag_events = group[group['event_type'] == 'diagnosis'].index
        other_events = group[group['event_type'] != 'diagnosis'].index
        if not diag_events.empty and not other_events.empty:
            for diag_event in diag_events:
                for other_event in other_events:
                    # Ensure unique edges by sorting node pairs
                    src, dst = sorted([diag_event, other_event])
                    edges.append((src, dst))

    if edges:
        src, dst = zip(*edges)
        g.add_edges(torch.tensor(src), torch.tensor(dst), etype=('event', 'cooccurs_in_admission_with', 'event'))
    else:
        print("No edges to add.")  # Debugging print if edges list is empty

    return g

def create_medication_combination_hypergraph(patient_events, g):
    """Create hyperedges for common medication combinations within the same admission."""
    medication_groups = patient_events[patient_events['event_type'] == 'medication'].groupby('hadm_id')
    edges = []

    for _, group in medication_groups:
        if len(group) > 1:
            for i, med_i in enumerate(group.index):
                for med_j in group.index[i+1:]:
                    edges.append(sorted([med_i, med_j]))

    if edges:
        src, dst = zip(*edges)
        g.add_edges(torch.tensor(src), torch.tensor(dst), etype=('medication', 'cooccurs_with', 'medication'))
    else:
        print("No edges to add.")  # Debugging print if edges list is empty    

    return g

if __name__ == "__main__":
    mimic = MIMICIVDataModule('data/MIMIC-IV-short')
    mimic.prepare_data()