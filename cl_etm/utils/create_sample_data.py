import pandas as pd
import numpy as np
import os
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Generate PATIENTS data
patients_data = []
for _ in range(10000):
    subject_id = fake.unique.uuid4()
    gender = random.choice(['M', 'F'])
    dob = fake.date_of_birth(minimum_age=0, maximum_age=90)
    dod = fake.date_between(start_date=dob, end_date='today') if random.choice([True, False]) else None
    dod_hosp = dod if random.choice([True, False]) else None
    dod_ssn = dod if random.choice([True, False]) else None
    expire_flag = 1 if dod else 0
    patients_data.append([subject_id, gender, dob, dod, dod_hosp, dod_ssn, expire_flag])

patients_df = pd.DataFrame(patients_data, columns=['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'DOD_HOSP', 'DOD_SSN', 'EXPIRE_FLAG'])

# Generate ADMISSIONS data
admissions_data = []
for _ in range(20000):
    hadm_id = fake.unique.uuid4()
    subject_id = random.choice(patients_df['SUBJECT_ID'])
    admittime = fake.date_time_this_decade()
    dischtime = fake.date_time_between(start_date=admittime, end_date='now')
    deathtime = fake.date_time_between(start_date=admittime, end_date=dischtime) if random.choice([True, False]) else None
    admission_type = random.choice(['emergency', 'urgent', 'elective'])
    admission_location = fake.city()
    discharge_location = fake.city()
    insurance = random.choice(['Medicare', 'Medicaid', 'Private', 'Self Pay'])
    language = fake.language_name()
    marital_status = random.choice(['Single', 'Married', 'Divorced', 'Widowed'])
    ethnicity = random.choice(['Caucasian', 'African American', 'Asian', 'Hispanic', 'Other'])
    admissions_data.append([hadm_id, subject_id, admittime, dischtime, deathtime, admission_type, admission_location, discharge_location, insurance, language, marital_status, ethnicity])

admissions_df = pd.DataFrame(admissions_data, columns=['HADM_ID', 'SUBJECT_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'LANGUAGE', 'MARITAL_STATUS', 'ETHNICITY'])

# Generate DIAGNOSES_ICD data
diagnoses_icd_data = []
for _ in range(5000):
    hadm_id = random.choice(admissions_df['HADM_ID'])
    subject_id = admissions_df[admissions_df['HADM_ID'] == hadm_id]['SUBJECT_ID'].values[0]
    icd_code = fake.bothify(text='?##')
    seq_num = random.randint(1, 10)
    charttime = fake.date_time_this_decade()
    diagnoses_icd_data.append([hadm_id, subject_id, icd_code, seq_num, charttime])

diagnoses_icd_df = pd.DataFrame(diagnoses_icd_data, columns=['HADM_ID', 'SUBJECT_ID', 'ICD_CODE', 'SEQ_NUM', 'CHARTTIME'])

# Generate PROCEDURES_ICD data
procedures_icd_data = []
for _ in range(3000):
    hadm_id = random.choice(admissions_df['HADM_ID'])
    subject_id = admissions_df[admissions_df['HADM_ID'] == hadm_id]['SUBJECT_ID'].values[0]
    icd_code = fake.bothify(text='?##')
    seq_num = random.randint(1, 10)
    charttime = fake.date_time_this_decade()
    procedures_icd_data.append([hadm_id, subject_id, icd_code, seq_num, charttime])

procedures_icd_df = pd.DataFrame(procedures_icd_data, columns=['HADM_ID', 'SUBJECT_ID', 'ICD_CODE', 'SEQ_NUM', 'CHARTTIME'])

# Generate PRESCRIPTIONS data
prescriptions_data = []
for _ in range(4000):
    hadm_id = random.choice(admissions_df['HADM_ID'])
    subject_id = admissions_df[admissions_df['HADM_ID'] == hadm_id]['SUBJECT_ID'].values[0]
    startdate = fake.date_between(start_date='-2y', end_date='today')
    enddate = fake.date_between(start_date=startdate, end_date='today')
    drug = fake.word()
    dose_val_rx = random.randint(1, 1000)
    dose_unit_rx = random.choice(['mg', 'ml', 'g'])
    route = random.choice(['oral', 'intravenous', 'topical', 'inhalation'])
    prescriptions_data.append([hadm_id, subject_id, startdate, enddate, drug, dose_val_rx, dose_unit_rx, route])

prescriptions_df = pd.DataFrame(prescriptions_data, columns=['HADM_ID', 'SUBJECT_ID', 'STARTDATE', 'ENDDATE', 'DRUG', 'DOSE_VAL_RX', 'DOSE_UNIT_RX', 'ROUTE'])
# Generate LABEVENTS data 
labevents_data = []
for _ in range(5000):
    hadm_id = random.choice(admissions_df['HADM_ID'])
    subject_id = admissions_df[admissions_df['HADM_ID'] == hadm_id]['SUBJECT_ID'].values[0]
    charttime = fake.date_time_this_decade()
    itemid = fake.bothify(text='???###')
    value = random.uniform(0, 100)
    valuenum = random.uniform(0, 100)
    valueuom = random.choice(['mg/dL', 'mmol/L', 'g/L'])
    flag = random.choice(['normal', 'abnormal'])
    labevents_data.append([hadm_id, subject_id, charttime, itemid, value, valuenum, valueuom, flag])

labevents_df = pd.DataFrame(labevents_data, columns=['HADM_ID', 'SUBJECT_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUENUM', 'VALUEUOM', 'FLAG'])

# Generate NOTEEVENTS data
noteevents_data = []
for _ in range(3000):
    hadm_id = random.choice(admissions_df['HADM_ID'])
    subject_id = admissions_df[admissions_df['HADM_ID'] == hadm_id]['SUBJECT_ID'].values[0]
    chartdate = fake.date_this_decade()
    charttime = fake.date_time_this_decade()
    category = random.choice(['discharge summary', 'nursing note', 'physician note'])
    text = fake.paragraph(nb_sentences=5)
    noteevents_data.append([hadm_id, subject_id, chartdate, charttime, category, text])

noteevents_df = pd.DataFrame(noteevents_data, columns=['HADM_ID', 'SUBJECT_ID', 'CHARTDATE', 'CHARTTIME', 'CATEGORY', 'TEXT'])

# Save dataframes to CSV
data_path = "data/"

if not os.path.exists(data_path):
    os.makedirs(data_path)

patients_df.to_csv(os.path.join(data_path, "patients.csv"), index=False)
admissions_df.to_csv(os.path.join(data_path, "admissions.csv"), index=False)
diagnoses_icd_df.to_csv(os.path.join(data_path, "diagnoses.csv"), index=False)
procedures_icd_df.to_csv(os.path.join(data_path, "procedures.csv"), index=False)
prescriptions_df.to_csv(os.path.join(data_path, "prescriptions.csv"), index=False)
labevents_df.to_csv(os.path.join(data_path, "labevents.csv"), index=False)
noteevents_df.to_csv(os.path.join(data_path, "noteevents.csv"), index=False)
# Displaying the dataframes for user
# patients_df, admissions_df, diagnoses_df, procedures_df, prescriptions_df, labevents_df
