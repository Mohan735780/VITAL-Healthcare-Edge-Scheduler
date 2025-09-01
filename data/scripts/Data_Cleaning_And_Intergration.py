import pandas as pd
import numpy as np
import glob
import re
from faker import Faker
import random
import os

def standardize_columns(df):
    """
    Converts all column names to a standardized snake_case format.
    """
    new_columns = {}
    for col in df.columns:
        clean_col = col.lower()
        clean_col = re.sub(r'[^a-zA-Z0-9]+', '_', clean_col)
        clean_col = clean_col.strip('_')
        new_columns[col] = clean_col
    
    df.rename(columns=new_columns, inplace=True)
    return df

def clean_and_merge_data():
    """
    Main function to load, clean, augment, and integrate all datasets,
    now including the healthcare offloading data.
    """
    print("--- Starting Enhanced Data Cleaning and Integration Process ---")
    
    # --- 1. LOAD ALL DATASETS ---
    print("Attempting to load all CSV files...")
    all_files = glob.glob("*.csv")
    dataframes = {}

    for f in all_files:
        # Use os.path.basename to handle different path structures
        filename = os.path.basename(f).split('.')[0].replace(' ', '_')
        try:
            dataframes[filename] = pd.read_csv(f, engine='python', on_bad_lines='skip')
        except Exception as e:
            print(f"FATAL: An error occurred while reading {f}. Error: {e}")
            return
    
    if not dataframes:
        print("No CSV files found in the directory. Exiting.")
        return

    print(f"\nSuccessfully loaded {len(dataframes)} datasets: {list(dataframes.keys())}")

    # --- NEW: VALIDATE REQUIRED FILES ---
    required_files = [
        'HCP', 'medical_resource_allocation_binary_dataset',
        'DPTARA_dataset', 'DPTARA_Synthetic_Healthcare_Tasks',
        'healthcare_offloading_dataset'
    ]
    
    missing_files = [rf for rf in required_files if rf not in dataframes]
    
    if missing_files:
        print("\n--- FATAL ERROR: Missing Required Data Files ---")
        print("The script cannot proceed because the following essential CSV files were not found:")
        for mf in missing_files:
            print(f"  - {mf}.csv")
        print("\nPlease ensure all required source datasets are in the same directory as this script.")
        return
        
    # --- 2. STANDARDIZE & CLEAN INDIVIDUAL DATAFRAMES ---
    for name, df in dataframes.items():
        dataframes[name] = standardize_columns(df)

    # Clinical Datasets
    df_hcp = dataframes['HCP']
    df_hcp['admissiondate'] = pd.to_datetime(df_hcp['admissiondate'], errors='coerce')
    df_hcp['dischargedate'] = pd.to_datetime(df_hcp['dischargedate'], errors='coerce')
    df_hcp.rename(columns={'patientid': 'patient_id'}, inplace=True)
    
    df_medical_resource = dataframes['medical_resource_allocation_binary_dataset']

    # Technical Datasets
    df_dptara = dataframes['DPTARA_dataset']
    df_synthetic_tasks = dataframes['DPTARA_Synthetic_Healthcare_Tasks']
    df_dptara['task_id'] = 'TASK_' + df_dptara['task_id'].astype(str)
    
    # NEW: Offloading Dataset
    df_offloading = dataframes['healthcare_offloading_dataset']
    # Keep only the most valuable and unique columns to avoid confusion
    offloading_cols_to_keep = [
        'heart_rate', 'dwt_coefficients', 'energy_consumption_j', 
        'response_time_ms', 'sla_violation'
    ]
    df_offloading = df_offloading[[col for col in offloading_cols_to_keep if col in df_offloading.columns]]
    print("Cleaned and selected key columns from healthcare_offloading_dataset.")

    # --- 3. CORE DATA INTEGRATION & AUGMENTATION ---
    print("\nStarting core data integration...")
    
    # Merge clinical data on 'patient_id'
    clinical_merged = pd.merge(df_hcp, df_medical_resource, on='patient_id', how='left', suffixes=('', '_res'))
    
    # Merge technical data
    technical_merged = pd.merge(df_dptara, df_synthetic_tasks, on='task_id', how='inner')
    
    # Augment technical data if needed
    if len(technical_merged) > 0 and len(technical_merged) < 1000:
        factor = int(1000 / len(technical_merged)) + 1
        augmented_tech = pd.concat([technical_merged] * factor, ignore_index=True)
        augmented_tech['task_id'] = augmented_tech['task_id'] + '_aug_' + augmented_tech.index.astype(str)
        technical_merged = augmented_tech.sample(frac=1).reset_index(drop=True)
        print(f"Augmented technical data to {len(technical_merged)} rows.")

    # --- 4. SYNTHETIC LINKING (ALL DATASTREAMS) ---
    # Link base clinical and technical data
    num_clinical = len(clinical_merged)
    num_technical = len(technical_merged)
    link_index_clinic = [i % num_clinical for i in range(num_technical)]
    clinical_to_link = clinical_merged.iloc[link_index_clinic].reset_index(drop=True)
    base_df = pd.concat([technical_merged.reset_index(drop=True), clinical_to_link], axis=1)
    
    # NEW: Link the offloading data to the base dataset
    num_base = len(base_df)
    num_offloading = len(df_offloading)
    link_index_offload = [i % num_offloading for i in range(num_base)]
    offloading_to_link = df_offloading.iloc[link_index_offload].reset_index(drop=True)
    final_df = pd.concat([base_df, offloading_to_link], axis=1)
    print("Synthetically linked all data streams: Clinical, Technical, and Offloading.")

    # --- 5. CONSOLIDATE & REMOVE REDUNDANT COLUMNS ---
    print("\nConsolidating features and removing redundant columns...")
    
    # Define columns to drop to avoid confusion for the model
    cols_to_drop = [
        'task_size_mb_y', 'deadline_s', 'priority_level', # Redundant technical info
        'age_res', 'gender_res', 'duration_of_stay', # Redundant clinical info
        'latency_ms' # Less specific than edge/cloud latency
    ]
    
    # Drop columns if they exist
    existing_cols_to_drop = [col for col in cols_to_drop if col in final_df.columns]
    final_df.drop(columns=existing_cols_to_drop, inplace=True)
    print(f"Dropped {len(existing_cols_to_drop)} redundant columns: {existing_cols_to_drop}")

    # --- 6. ADVANCED LOGICAL IMPUTATION ---
    print("\nImputing missing values with meaningful, rule-based random data...")
    fake = Faker()
    
    # Impute doctor names
    if 'doctorname' in final_df.columns and final_df['doctorname'].isna().any():
        final_df['doctorname'].fillna(f"Dr. {fake.last_name()}", inplace=True)

    # Create synthetic disease severity score (1-5) as a basis for logical imputation
    if 'disease_severity' not in final_df.columns or final_df['disease_severity'].isnull().any():
        final_df['disease_severity'] = np.random.randint(1, 6, size=len(final_df))
    
    # Define mappings for logical imputation
    health_status_choices = ['Stable', 'Unstable', 'Emergency']
    disease_type_map = {1: 'Allergy', 2: 'Influenza', 3: 'Diabetes', 4: 'Pneumonia', 5: 'Stroke'}
    resource_needed_map = {1: 'Medicine', 2: 'Staff', 3: 'Bed', 4: 'Specialist', 5: 'ICU'}

    def impute_by_severity(row, col, mapping):
        default_value = list(mapping.values())[0]
        return mapping.get(row['disease_severity'], default_value) if pd.isna(row[col]) else row[col]

    if 'health_status' in final_df.columns:
        final_df['health_status'] = final_df['health_status'].apply(lambda x: random.choice(health_status_choices) if pd.isna(x) else x)
    if 'disease_type' in final_df.columns:
        final_df['disease_type'] = final_df.apply(impute_by_severity, args=('disease_type', disease_type_map), axis=1)
    if 'resource_needed' in final_df.columns:
        final_df['resource_needed'] = final_df.apply(impute_by_severity, args=('resource_needed', resource_needed_map), axis=1)

    # --- NEW: Logically impute resource and network columns ---
    print("Logically imputing resource and network monitoring columns...")
    
    # Generate Sensor and Resource IDs
    final_df['sensor_id'] = [f'S{str(random.randint(100, 999))}' for _ in range(len(final_df))]
    final_df['resource_id'] = [f'R{str(random.randint(100, 999))}' for _ in range(len(final_df))]
    
    # Utilization status based on disease severity
    final_df['utilization_status'] = final_df['disease_severity'].apply(lambda s: 1 if random.random() < s / 5.5 else 0)
    
    # Network speed based on latency
    if 'network_latency_edge_ms' in final_df.columns:
        final_df['network_speed'] = 100 - (final_df['network_latency_edge_ms'] * np.random.uniform(0.5, 1.5))
        final_df['network_speed'] = final_df['network_speed'].clip(lower=5, upper=100)
    else:
        final_df['network_speed'] = np.random.uniform(10, 100, size=len(final_df))
        
    # Overlapping interference based on SLA violation
    if 'sla_violation' in final_df.columns and final_df['sla_violation'].notna().any():
         final_df['overlapping_interference'] = final_df['sla_violation'].apply(lambda s: 1 if (s == 1 and random.random() < 0.7) or (s == 0 and random.random() < 0.1) else 0)
    else:
        final_df['overlapping_interference'] = np.random.randint(0, 2, size=len(final_df))

    # Resource utilization efficiency based on treatment cost and duration
    final_df['length_of_stay'] = pd.to_numeric(final_df['length_of_stay'], errors='coerce').fillna(1).replace(0, 1)
    final_df['treatmentcost'] = pd.to_numeric(final_df['treatmentcost'], errors='coerce').fillna(final_df['treatmentcost'].median())
    cost_intensity = (final_df['treatmentcost'] / final_df['length_of_stay'])
    scaled_intensity = (cost_intensity - cost_intensity.min()) / (cost_intensity.max() - cost_intensity.min())
    final_df['resource_utilization_efficiency'] = 1 - scaled_intensity + np.random.uniform(-0.1, 0.1, size=len(final_df))
    final_df['resource_utilization_efficiency'] = final_df['resource_utilization_efficiency'].clip(0, 1)

    # Fill any remaining numerical NaNs with the column median
    for col in final_df.select_dtypes(include=np.number).columns:
        if final_df[col].isnull().any():
            final_df[col].fillna(final_df[col].median(), inplace=True)
            
    print("Imputation complete.")

    # --- 7. SAVE THE FINAL DATASET ---
    final_df.to_csv('integrated_dptara_plus_dataset.csv', index=False)
    
    print("\n--- Process Complete! ---")
    print("Successfully created 'integrated_dptara_plus_dataset.csv'")
    print("This file now includes offloading data and is ready for Feature Engineering.")
    print("\nFinal Dataset Preview:")
    print(final_df.head())
    print("\nFinal Dataset Shape:", final_df.shape)
    print("\nMissing values check after imputation:")
    print(final_df.isnull().sum().to_string())

if __name__ == '__main__':
    clean_and_merge_data()

