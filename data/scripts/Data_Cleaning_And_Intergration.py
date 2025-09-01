import pandas as pd
import numpy as np
import glob
import re
from faker import Faker
import random

def standardize_columns(df):
    """
    Converts all column names to a standardized snake_case format.
    Example: 'Patient_ID' becomes 'patient_id'.
    This version is simpler and more reliable.
    """
    new_columns = {}
    for col in df.columns:
        # Convert to lowercase
        clean_col = col.lower()
        # Replace all non-alphanumeric characters with an underscore
        clean_col = re.sub(r'[^a-zA-Z0-9]+', '_', clean_col)
        # Remove any leading or trailing underscores that might result
        clean_col = clean_col.strip('_')
        new_columns[col] = clean_col
    
    df.rename(columns=new_columns, inplace=True)
    return df

def clean_and_merge_data():
    """
    Main function to load, clean, augment, and integrate all datasets.
    """
    print("Starting data cleaning and integration process...")
    
    # --- 1. LOAD ALL DATASETS ---
    print("Attempting to load all CSV files...")
    all_files = glob.glob("*.csv")
    problem_file = 'healthcare offloading_dataset.csv'
    dataframes = {}

    for f in all_files:
        filename = f.split('.')[0].replace(' ', '_') # Sanitize filename
        try:
            if f == problem_file:
                dataframes[filename] = pd.read_csv(f, encoding='utf-8')
            else:
                 dataframes[filename] = pd.read_csv(f)
        except pd.errors.ParserError:
            print(f"Warning: ParserError for {f}. Retrying with robust Python engine.")
            try:
                dataframes[filename] = pd.read_csv(f, engine='python', on_bad_lines='skip')
                print(f"Successfully loaded {f} with Python engine (some rows may have been skipped).")
            except Exception as inner_e:
                print(f"FATAL: Could not read {f} even with robust Python engine. Error: {inner_e}")
                return
        except Exception as e:
            print(f"FATAL: An unexpected error occurred while reading {f}. Error: {e}")
            return

    if not dataframes:
        print("No dataframes were loaded. Exiting.")
        return
    
    print(f"\nSuccessfully loaded {len(dataframes)} datasets: {list(dataframes.keys())}")


    # --- 2. INDIVIDUAL DATAFRAME CLEANING ---
    for name, df in dataframes.items():
        dataframes[name] = standardize_columns(df)
    
    # Clinical Datasets
    df_hcp = dataframes['HCP']
    df_hcp['admissiondate'] = pd.to_datetime(df_hcp['admissiondate'])
    df_hcp['dischargedate'] = pd.to_datetime(df_hcp['dischargedate'])
    df_hcp['treatmentcost'] = df_hcp['treatmentcost'].astype(float)
    df_hcp.rename(columns={'patientid': 'patient_id'}, inplace=True)
    print("Cleaned HCP dataset.")

    df_medical_resource = dataframes['medical_resource_allocation_binary_dataset']
    print("Cleaned medical_resource_allocation dataset.")

    # Technical Datasets
    df_dptara = dataframes['DPTARA_dataset']
    df_synthetic_tasks = dataframes['DPTARA_Synthetic_Healthcare_Tasks']
    print("Cleaned technical datasets.")


    # --- 3. DATA INTEGRATION & AUGMENTATION ---
    print("\nStarting data integration and augmentation...")
    
    # Merge clinical data on 'patient_id'
    clinical_merged = pd.merge(df_hcp, df_medical_resource, on='patient_id', how='left', suffixes=('', '_res'))
    print("Merged clinical datasets.")
    
    # Merge technical data
    df_dptara['task_id'] = 'TASK_' + df_dptara['task_id'].astype(str)
    technical_merged = pd.merge(df_dptara, df_synthetic_tasks, on='task_id', how='inner')
    
    # Augment technical data to be > 1000 rows
    if len(technical_merged) > 0 and len(technical_merged) < 1000:
        factor = int(1000 / len(technical_merged)) + 1
        augmented_tech = pd.concat([technical_merged] * factor, ignore_index=True)
        # Create new unique task IDs for augmented data
        augmented_tech['task_id'] = augmented_tech['task_id'] + '_aug_' + augmented_tech.index.astype(str)
        technical_merged = augmented_tech.sample(frac=1).reset_index(drop=True) # Shuffle
        print(f"Augmented technical data to {len(technical_merged)} rows.")


    # --- 4. SYNTHETIC LINKING (CRITICAL STEP) ---
    num_clinical_records = len(clinical_merged)
    num_technical_records = len(technical_merged)
    
    if num_clinical_records == 0:
        print("Warning: No clinical records to merge. Final dataset will only contain technical data.")
        final_df = technical_merged
    else:
        link_index = [i % num_clinical_records for i in range(num_technical_records)]
        clinical_to_link = clinical_merged.iloc[link_index].reset_index(drop=True)
        final_df = pd.concat([technical_merged.reset_index(drop=True), clinical_to_link], axis=1)
        print("Synthetically linked clinical and technical data.")
        
    # --- 5. IMPUTE MISSING VALUES ---
    print("\nImputing missing values with meaningful, rule-based random data...")
    fake = Faker()
    
    # Impute doctor names
    if 'doctorname' in final_df.columns:
        # Use a single fake name for all NaNs for performance
        missing_doctor_mask = final_df['doctorname'].isna()
        final_df.loc[missing_doctor_mask, 'doctorname'] = f"Dr. {fake.last_name()}"

    # --- RULE-BASED IMPUTATION FOR KEY CLINICAL COLUMNS ---
    
    # Rule 1: Impute gender_res from the primary gender column
    if 'gender_res' in final_df.columns and 'gender' in final_df.columns:
        final_df['gender_res'].fillna(final_df['gender'], inplace=True)
        # If any still remain (e.g., if primary gender was also NaN), fill with random choice
        final_df['gender_res'].fillna(random.choice(['Male', 'Female']), inplace=True)
        print("Imputed 'gender_res' based on primary gender column.")

    # Rule 2: Create a synthetic disease severity score for all rows (1-5)
    final_df['disease_severity'] = np.random.randint(1, 6, size=len(final_df))
    print("Created synthetic 'disease_severity' score (1-5).")

    # Define mappings for imputation
    health_status_choices = ['Stable', 'Unstable', 'Emergency']
    disease_type_map = {
        1: ['Allergy', 'Common Cold'],
        2: ['Migraine', 'Influenza'],
        3: ['Diabetes', 'Asthma', 'Hypertension'],
        4: ['Pneumonia', 'Severe Fracture', 'Kidney Stone'],
        5: ['Cancer', 'Stroke', 'Heart Attack']
    }
    resource_needed_map = {
        1: ['Medicine'],
        2: ['Medicine', 'Staff'],
        3: ['Staff', 'Bed'],
        4: ['Bed', 'Specialist Consultation'],
        5: ['ICU', 'Surgical Procedure']
    }

    # Define a function to apply imputation based on severity
    def impute_based_on_severity(row, col_to_impute, mapping):
        if pd.isna(row[col_to_impute]):
            severity = row['disease_severity']
            return random.choice(mapping.get(severity, ['Medicine'])) # Default for safety
        return row[col_to_impute]

    # Rule 3: Impute health_status randomly
    if 'health_status' in final_df.columns:
        final_df['health_status'] = final_df['health_status'].apply(
            lambda x: random.choice(health_status_choices) if pd.isna(x) else x
        )
        print("Imputed 'health_status' with random values.")

    # Rule 4: Impute disease_type based on severity
    if 'disease_type' in final_df.columns:
        final_df['disease_type'] = final_df.apply(
            lambda row: impute_based_on_severity(row, 'disease_type', disease_type_map), axis=1
        )
        print("Imputed 'disease_type' based on severity score.")

    # Rule 5: Impute resource_needed based on severity
    if 'resource_needed' in final_df.columns:
         final_df['resource_needed'] = final_df.apply(
            lambda row: impute_based_on_severity(row, 'resource_needed', resource_needed_map), axis=1
        )
         print("Imputed 'resource_needed' based on severity score.")

    # --- Standard imputation for other remaining columns ---
    # Numerical columns
    if 'age_res' in final_df.columns:
         final_df['age_res'].fillna(pd.Series(np.random.randint(20, 85, size=len(final_df))), inplace=True)
    if 'duration_of_stay' in final_df.columns:
         final_df['duration_of_stay'].fillna(pd.Series(np.random.randint(1, 25, size=len(final_df))), inplace=True)
    if 'network_speed' in final_df.columns:
         final_df['network_speed'].fillna(pd.Series(np.random.uniform(10, 100, size=len(final_df))), inplace=True)
         
    # Binary/ID columns
    if 'utilization_status' in final_df.columns:
        final_df['utilization_status'].fillna(pd.Series(np.random.randint(0, 2, size=len(final_df))), inplace=True)
    if 'overlapping_interference' in final_df.columns:
        final_df['overlapping_interference'].fillna(pd.Series(np.random.randint(0, 2, size=len(final_df))), inplace=True)
    if 'resource_utilization_efficiency' in final_df.columns:
        final_df['resource_utilization_efficiency'].fillna(pd.Series(np.random.randint(0, 2, size=len(final_df))), inplace=True)
    if 'sensor_id' in final_df.columns:
        final_df['sensor_id'] = final_df['sensor_id'].apply(lambda x: f"S{random.randint(100, 999)}" if pd.isna(x) else x)
    if 'resource_id' in final_df.columns:
        final_df['resource_id'] = final_df['resource_id'].apply(lambda x: f"R{random.randint(100, 999)}" if pd.isna(x) else x)
        
    print("Imputation complete.")


    # --- 6. SAVE THE FINAL DATASET ---
    final_df.to_csv('integrated_dptara_plus_dataset.csv', index=False)
    
    print("\n--- Process Complete! ---")
    print("Successfully created 'integrated_dptara_plus_dataset.csv'")
    print("This file is now ready for the final Feature Engineering stage.")
    print("\nFinal Dataset Preview:")
    print(final_df.head())
    print("\nFinal Dataset Shape:", final_df.shape)
    print("\nMissing values check after imputation:")
    print(final_df.isnull().sum())


if __name__ == '__main__':
    clean_and_merge_data()

