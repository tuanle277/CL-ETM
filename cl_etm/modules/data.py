import torch
from torch.utils.data import DataLoader
import networkx as nx
import matplotlib.pyplot as plt
import dgl
import pandas as pd
from datetime import timedelta
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Assuming we have the sample data as created previously in dataframes
# patients_df, admissions_df, diagnoses_icd_df, procedures_icd_df, prescriptions_df, labevents_df

class MIMICIVDataModule:
    def __init__(self, data_dir='data', batch_size=32, time_window=7, max_events_per_visit=50, validation_split=0.1, test_split=0.1):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.time_window = time_window
        self.max_events_per_visit = max_events_per_visit
        self.validation_split = validation_split
        self.test_split = test_split
        self.mimic_data = None
        self.patient_graphs = None
        self.patient_disease_graph = None

    def load_data(self):

        """Loads and preprocesses MIMIC-IV data."""
        # Load relevant MIMIC-IV tables (adjust paths as needed)
        patients = pd.read_csv(self.data_dir + '/patients.csv')
        admissions = pd.read_csv(self.data_dir + '/admissions.csv')
        diagnoses_icd= pd.read_csv(self.data_dir + '/diagnoses.csv')
        procedures_icd = pd.read_csv(self.data_dir + '/procedures.csv')
        prescriptions = pd.read_csv(self.data_dir + '/prescriptions.csv')
        labevents = pd.read_csv(self.data_dir + '/labevents.csv')
        noteevents = pd.read_csv(self.data_dir + '/noteevents.csv')

        # ... Load other relevant tables 
        # Ensure date columns are datetime objects
        admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
        admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'])
        admissions['DEATHTIME'] = pd.to_datetime(admissions['DEATHTIME'], errors='coerce')
        diagnoses_icd['CHARTTIME'] = pd.to_datetime(diagnoses_icd['CHARTTIME'], errors='coerce')
        procedures_icd['CHARTTIME'] = pd.to_datetime(procedures_icd['CHARTTIME'], errors='coerce')
        prescriptions['STARTDATE'] = pd.to_datetime(prescriptions['STARTDATE'], errors='coerce')
        prescriptions['ENDDATE'] = pd.to_datetime(prescriptions['ENDDATE'], errors='coerce')
        labevents['CHARTTIME'] = pd.to_datetime(labevents['CHARTTIME'], errors='coerce')
        noteevents['CHARTTIME'] = pd.to_datetime(noteevents['CHARTTIME'], errors='coerce')

        # Preprocess data (handle missing values, normalization, etc.)
        # ... (Impute/remove missing values, standardize numerical features, tokenize notes)

        # Combine and label events
        diagnoses_icd['event_type'] = 'diagnosis'
        procedures_icd['event_type'] = 'procedure'
        prescriptions['event_type'] = 'medication'
        labevents['event_type'] = 'lab'
        noteevents['event_type'] = 'note'

        events = pd.concat([
            diagnoses_icd[['SUBJECT_ID', 'HADM_ID', 'ICD_CODE', 'CHARTTIME', 'event_type']],
            procedures_icd[['SUBJECT_ID', 'HADM_ID', 'ICD_CODE', 'CHARTTIME', 'event_type']],
            prescriptions[['SUBJECT_ID', 'HADM_ID', 'DRUG', 'STARTDATE', 'ENDDATE', 'event_type']],
            labevents[['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'event_type']],
            noteevents[['HADM_ID', 'SUBJECT_ID','CHARTDATE','CHARTTIME','CATEGORY','TEXT', 'event_type']]
        ])

        return patients, admissions, events

    def construct_patient_hypergraphs(self):
        """
        Constructs a list of heterogeneous dgl graphs, each representing a patient's medical history as a hypergraph.
        """
        patient_graphs = []

        # Encode categorical features
        le_icd = LabelEncoder()
        le_drug = LabelEncoder()
        le_item = LabelEncoder()

        all_icd_codes = self.mimic_data['events']['ICD_CODE'].fillna('0').unique()
        all_drugs = self.mimic_data['events']['DRUG'].fillna('0').unique()
        all_items = self.mimic_data['events']['ITEMID'].fillna('0').unique()

        le_icd.fit(all_icd_codes)
        le_drug.fit(all_drugs)
        le_item.fit(all_items)
        
        self.mimic_data['events']['ICD_CODE'] = le_icd.fit_transform(self.mimic_data['events']['ICD_CODE'].fillna('0'))
        self.mimic_data['events']['DRUG'] = le_drug.fit_transform(self.mimic_data['events']['DRUG'].fillna('0'))
        self.mimic_data['events']['ITEMID'] = le_item.fit_transform(self.mimic_data['events']['ITEMID'].fillna('0'))

        for patient_id in tqdm(self.mimic_data['patients']['SUBJECT_ID'].unique(), desc="Creating patient graphs"):
            patient_events = self.mimic_data['events'][self.mimic_data['events']['SUBJECT_ID'] == patient_id].sort_values('CHARTTIME')

            # Create Heterogeneous Graph
            graph_data = {
                ('patient', 'has_diagnosis', 'diagnosis'): ([], []),
                ('patient', 'has_procedure', 'procedure'): ([], []),
                ('patient', 'has_medication', 'medication'): ([], []),
                ('patient', 'has_lab', 'lab'): ([], []),
                ('patient', 'has_note', 'note'): ([], []),
                ('diagnosis', 'diagnosis_patient', 'patient'): ([], []),
                ('procedure', 'procedure_patient', 'patient'): ([], []),
                ('medication', 'medication_patient', 'patient'): ([], []),
                ('lab', 'lab_patient', 'patient'): ([], []),
                ('note', 'note_patient', 'patient'): ([], []),
                ('event', 'occurs_after', 'event'): ([], []),
                ('event', 'cooccurs_with_diagnosis', 'event'): ([], []),
                ('event', 'cooccurs_with_procedure', 'event'): ([], []),
                ('event', 'cooccurs_with_medication', 'event'): ([], []),
                ('event', 'cooccurs_with_lab', 'event'): ([], []),
                ('event', 'cooccurs_with_note', 'event'): ([], [])

            }

            # Add nodes and edges for each event type
            for event_type in ['diagnosis', 'procedure', 'medication', 'lab', 'note']:
                event_df = patient_events[patient_events['event_type'] == event_type]
                if not event_df.empty:
                    event_ids = torch.arange(len(event_df))
                    graph_data[('patient', f'has_{event_type}', event_type)][0].extend([0] * len(event_df))
                    graph_data[('patient', f'has_{event_type}', event_type)][1].extend(event_ids)
                    graph_data[(event_type, f'{event_type}_patient', 'patient')][0].extend(event_ids)
                    graph_data[(event_type, f'{event_type}_patient', 'patient')][1].extend([0] * len(event_df))


            # Create temporal edges
            event_times = patient_events['CHARTTIME'].values
            for i in range(len(event_times) - 1):
                graph_data[('event', 'occurs_after', 'event')][0].append(i)
                graph_data[('event', 'occurs_after', 'event')][1].append(i + 1)

            # Create hyperedges based on co-occurrence within time_window
            for i, event in patient_events.iterrows():
                start_time = event['CHARTTIME']
                end_time = event['CHARTTIME'] + timedelta(days=self.time_window)

                related_events = patient_events[
                    (patient_events['CHARTTIME'] > start_time) & 
                    (patient_events['CHARTTIME'] < end_time) &
                    (patient_events['event_type'] != event['event_type'])
                ]

                for related_event in related_events.itertuples():
                    graph_data[('event', f'cooccurs_with_{event["event_type"]}', 'event')][0].append(i)
                    graph_data[('event', f'cooccurs_with_{event["event_type"]}', 'event')][1].append(related_event.Index)
                    graph_data[('event', f'cooccurs_with_{related_event.event_type}', 'event')][0].append(related_event.Index)
                    graph_data[('event', f'cooccurs_with_{related_event.event_type}', 'event')][1].append(i)  # Make the hyperedge undirected

            g = dgl.heterograph(graph_data)

            # g.add_nodes(1, ntype='patient')

            # Add node features
            for event_type in ['diagnosis', 'procedure', 'medication', 'lab']:
                event_df = patient_events[patient_events['event_type'] == event_type]
                if not event_df.empty:
                    if event_type == 'diagnosis' or event_type == 'procedure':
                        g.nodes[event_type].data['feat'] = torch.from_numpy(event_df['ICD_CODE'].values).long()
                    elif event_type == 'medication':
                        g.nodes[event_type].data['feat'] = torch.from_numpy(event_df['DRUG'].values).long()
                    elif event_type == 'lab':
                        scaler = StandardScaler()
                        g.nodes[event_type].data['feat'] = torch.from_numpy(scaler.fit_transform(event_df[['ITEMID', 'VALUENUM']].fillna(0).values)).float()
                    elif event_type == 'note':
                        g.nodes[event_type].data['feat'] = event_df["TEXT"]

            # Add patient features
            patient_data = self.mimic_data['patients'][self.mimic_data['patients']['SUBJECT_ID'] == patient_id].iloc[0]
            # print(g.nodes['patient'].data, torch.tensor(
            #     [[patient_data['GENDER'] == 'M', 2024 - pd.to_datetime(patient_data['DOB']).year]],
            #     dtype=torch.float32
            # ))
            # g.nodes['patient'].data['feat'] = torch.tensor(
            #     [[patient_data['GENDER'] == 'M', 2024 - pd.to_datetime(patient_data['DOB']).year]],
            #     dtype=torch.float32
            # )
            patient_graphs.append(g)
            
        self.patient_graphs = patient_graphs

    def construct_patient_disease_graph(self):
        """Constructs a bipartite patient-disease graph."""
        patient_ids = self.mimic_data['patients']['SUBJECT_ID'].unique()
        disease_codes = self.mimic_data['events'][self.mimic_data['events']['event_type'] == 'diagnosis']['ICD_CODE'].unique()

        patient_index = {p: i for i, p in enumerate(patient_ids)}
        disease_index = {d: i for i, d in enumerate(disease_codes)}

        # Initialize the bipartite graph with patient and disease nodes
        num_patients = len(patient_ids)
        num_diseases = len(disease_codes)
        
        g = dgl.heterograph({
            ('patient', 'has_disease', 'disease'): ([], []),
            ('disease', 'disease_of', 'patient'): ([], [])
        })

        # Add patient and disease nodes
        g.add_nodes(num_patients, ntype='patient')
        g.add_nodes(num_diseases, ntype='disease')

        # Add edges
        patient_to_disease_edges = []
        disease_to_patient_edges = []

        for _, row in self.mimic_data['events'][self.mimic_data['events']['event_type'] == 'diagnosis'].iterrows():
            patient_id = row['SUBJECT_ID']
            disease_code = row['ICD_CODE']
            patient_to_disease_edges.append((patient_index[patient_id], disease_index[disease_code]))
            disease_to_patient_edges.append((disease_index[disease_code], patient_index[patient_id]))

        patient_to_disease_src, patient_to_disease_dst = zip(*patient_to_disease_edges)
        disease_to_patient_src, disease_to_patient_dst = zip(*disease_to_patient_edges)

        g.add_edges(patient_to_disease_src, patient_to_disease_dst, etype='has_disease')
        g.add_edges(disease_to_patient_src, disease_to_patient_dst, etype='disease_of')

        # Add features for patient nodes
        patient_feats = []
        for patient_id in patient_ids:
            patient_data = self.mimic_data['patients'][self.mimic_data['patients']['SUBJECT_ID'] == patient_id].iloc[0]
            patient_feat = [patient_data['GENDER'] == 'M', 2024 - pd.to_datetime(patient_data['DOB']).year]
            patient_feats.append(patient_feat)
        
        g.nodes['patient'].data['feat'] = torch.tensor(patient_feats, dtype=torch.float32)

        # No features for disease nodes in this example
        g.nodes['disease'].data['feat'] = torch.zeros((num_diseases, 1), dtype=torch.float32)
        
        self.patient_disease_graph = g
        
    def visualize_patient_hypergraph(self, g):
        """Visualizes a patient hypergraph."""
        nx_g = dgl.to_networkx(g, node_attrs=['feat'])
        pos = nx.spring_layout(nx_g)
        plt.figure(figsize=(10, 10))
        nx.draw(nx_g, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
        plt.title("Patient Hypergraph")
        plt.show()

    def visualize_patient_disease_graph(self, g):
        """Visualizes the patient-disease bipartite graph."""
        nx_g = dgl.to_networkx(g, node_attrs=['feat'])
        pos = nx.spring_layout(nx_g)
        plt.figure(figsize=(10, 10))
        nx.draw(nx_g, pos, with_labels=True, node_size=500, node_color="lightgreen", font_size=10, font_weight="bold", edge_color="gray")
        plt.title("Patient-Disease Bipartite Graph")
        plt.show()

    def create_dataloaders(self):
        """Creates dataloaders for training, validation, and testing."""
        # Split patient IDs into train, validation, and test sets
        num_patients = len(self.patient_graphs)
        num_val = int(num_patients * self.validation_split)
        num_test = int(num_patients * self.test_split)
        
        patient_ids = list(range(num_patients))
        np.random.shuffle(patient_ids)
        train_ids = patient_ids[:-num_val-num_test]
        val_ids = patient_ids[-num_val-num_test:-num_test]
        test_ids = patient_ids[-num_test:]

        # Create DataLoader objects for each split
        train_loader = DataLoader([self.patient_graphs[i] for i in train_ids], batch_size=self.batch_size, shuffle=True, collate_fn=dgl.batch)
        val_loader = DataLoader([self.patient_graphs[i] for i in val_ids], batch_size=self.batch_size, shuffle=False, collate_fn=dgl.batch)
        test_loader = DataLoader([self.patient_graphs[i] for i in test_ids], batch_size=self.batch_size, shuffle=False, collate_fn=dgl.batch)

        return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Usage
    mimic_data_module = MIMICIVDataModule()

    # Load and preprocess data
    patients, admissions, events = mimic_data_module.load_data()
    mimic_data_module.mimic_data = {'patients': patients, 'admissions': admissions, 'events': events}

    # Construct the patient hypergraphs
    mimic_data_module.construct_patient_hypergraphs()

    # Construct the patient-disease graph
    mimic_data_module.construct_patient_disease_graph()

    # Visualize the first patient hypergraph
    mimic_data_module.visualize_patient_hypergraph(mimic_data_module.patient_graphs[0])

    # Visualize the patient-disease graph
    mimic_data_module.visualize_patient_disease_graph(mimic_data_module.patient_disease_graph)


    print(mimic_data_module.patient_graphs)
    print("+++++++++++++++++")
    print(mimic_data_module.patient_disease_graph)

    # Create DataLoaders for training, validation, and testing
    train_loader, val_loader, test_loader = mimic_data_module.create_dataloaders()

    # Print a sample graph from the training DataLoader
    for batch in train_loader:
        print(batch)
        break  # Print only the first batch

