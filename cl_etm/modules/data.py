import torch
import pandas as pd
from datetime import timedelta
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
import dgl
import logging

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

    def load_data(self):
        """Loads and preprocesses MIMIC-IV data."""
        tables = ['patients', 'admissions', 'diagnoses', 'procedures', 'prescriptions', 'labevents', 'noteevents']
        data = {table: pd.read_csv(f"{self.data_dir}/{table}.csv") for table in tables}
        
        # Convert date columns to datetime objects
        date_columns = {
            'admissions': ['ADMITTIME', 'DISCHTIME', 'DEATHTIME'],
            'diagnoses': ['CHARTTIME'],
            'procedures': ['CHARTTIME'],
            'prescriptions': ['STARTDATE', 'ENDDATE'],
            'labevents': ['CHARTTIME'],
            'noteevents': ['CHARTTIME']
        }
        
        for table, columns in date_columns.items():
            for column in columns:
                data[table][column] = pd.to_datetime(data[table][column], errors='coerce')

        # Label events
        data['diagnoses']['event_type'] = 'diagnosis'
        data['procedures']['event_type'] = 'procedure'
        data['prescriptions']['event_type'] = 'medication'
        data['labevents']['event_type'] = 'lab'
        data['noteevents']['event_type'] = 'note'

        events = pd.concat([
            data['diagnoses'][['SUBJECT_ID', 'HADM_ID', 'ICD_CODE', 'CHARTTIME', 'event_type']],
            data['procedures'][['SUBJECT_ID', 'HADM_ID', 'ICD_CODE', 'CHARTTIME', 'event_type']],
            data['prescriptions'][['SUBJECT_ID', 'HADM_ID', 'DRUG', 'STARTDATE', 'ENDDATE', 'event_type']],
            data['labevents'][['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'event_type']],
            data['noteevents'][['HADM_ID', 'SUBJECT_ID', 'CHARTDATE', 'CHARTTIME', 'CATEGORY', 'TEXT', 'event_type']]
        ])
        
        return data['patients'], data['admissions'], events

    def construct_patient_hypergraphs(self):
        """Constructs a list of heterogeneous DGL graphs, each representing a patient's medical history as a hypergraph."""
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
        
        self.mimic_data['events']['ICD_CODE'] = le_icd.transform(self.mimic_data['events']['ICD_CODE'].fillna('0'))
        self.mimic_data['events']['DRUG'] = le_drug.transform(self.mimic_data['events']['DRUG'].fillna('0'))
        self.mimic_data['events']['ITEMID'] = le_item.transform(self.mimic_data['events']['ITEMID'].fillna('0'))

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
                    graph_data[('event', f'cooccurs_with_{related_event.event_type}', 'event')][1].append(i)

            g = dgl.heterograph(graph_data)

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

            patient_data = self.mimic_data['patients'][self.mimic_data['patients']['SUBJECT_ID'] == patient_id].iloc[0]
            # g.nodes['patient'].data['feat'] = torch.tensor([[patient_data['GENDER'] == 'M', 2024 - pd.to_datetime(patient_data['DOB']).year]], dtype=torch.float32)

            patient_graphs.append(g)
        
        self.patient_graphs = patient_graphs

    def construct_patient_disease_graph(self):
        """Constructs a bipartite patient-disease graph."""
        patient_ids = self.mimic_data['patients']['SUBJECT_ID'].unique()
        disease_codes = self.mimic_data['events'][self.mimic_data['events']['event_type'] == 'diagnosis']['ICD_CODE'].unique()

        patient_index = {p: i for i, p in enumerate(patient_ids)}
        disease_index = {d: i for i, d in enumerate(disease_codes)}

        g = dgl.heterograph({
            ('patient', 'has_disease', 'disease'): ([], []),
            ('disease', 'disease_of', 'patient'): ([], [])
        })

        g.add_nodes(len(patient_ids), ntype='patient')
        g.add_nodes(len(disease_codes), ntype='disease')

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

        patient_feats = []
        for patient_id in patient_ids:
            patient_data = self.mimic_data['patients'][self.mimic_data['patients']['SUBJECT_ID'] == patient_id].iloc[0]
            patient_feat = [patient_data['GENDER'] == 'M', 2024 - pd.to_datetime(patient_data['DOB']).year]
            patient_feats.append(patient_feat)
        
        g.nodes['patient'].data['feat'] = torch.tensor(patient_feats, dtype=torch.float32)
        g.nodes['disease'].data['feat'] = torch.zeros((len(disease_codes), 1), dtype=torch.float32)
        
        self.patient_disease_graph = g


'''
Graph Details:
==============

Node Counts:
------------
- Diagnosis: 170
- Event: 5471
- Lab: 181
- Medication: 141
- Note: 105
- Patient: 28
- Procedure: 111

Edge Counts:
------------
- (Diagnosis, diagnosis_patient, Patient): 170
- (Event, cooccurs_with_diagnosis, Event): 40
- (Event, cooccurs_with_lab, Event): 35
- (Event, cooccurs_with_medication, Event): 0
- (Event, cooccurs_with_note, Event): 25
- (Event, cooccurs_with_procedure, Event): 20
- (Event, occurs_after, Event): 680
- (Lab, lab_patient, Patient): 181
- (Medication, medication_patient, Patient): 141
- (Note, note_patient, Patient): 105
- (Patient, has_diagnosis, Diagnosis): 170
- (Patient, has_lab, Lab): 181
- (Patient, has_medication, Medication): 141
- (Patient, has_note, Note): 105
- (Patient, has_procedure, Procedure): 111
- (Procedure, procedure_patient, Patient): 111

Metagraph:
----------
- (Diagnosis, diagnosis_patient, Patient)
- (Patient, has_diagnosis, Diagnosis)
- (Patient, has_lab, Lab)
- (Patient, has_medication, Medication)
- (Patient, has_note, Note)
- (Patient, has_procedure, Procedure)
- (Event, cooccurs_with_diagnosis, Event)
- (Event, cooccurs_with_lab, Event)
- (Event, cooccurs_with_medication, Event)
- (Event, cooccurs_with_note, Event)
- (Event, cooccurs_with_procedure, Event)
- (Event, occurs_after, Event)
- (Lab, lab_patient, Patient)
- (Medication, medication_patient, Patient)
- (Note, note_patient, Patient)
- (Procedure, procedure_patient, Patient)

Continuous Hypergraph:
(Patient) -> (Diagnosis) -> (Event1) -> (Event2) -> (Lab)
  |             |           |             |           |
  v             v           v             v           v
(Medication) -> (Event3) -> (Note) -> (Procedure) -> (Event4)


'''