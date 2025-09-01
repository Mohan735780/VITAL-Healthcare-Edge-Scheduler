import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MinMaxScaler

def engineer_features(input_path='integrated_dptara_plus_dataset.csv', output_path='final_analytics_base_table.csv'):
    """
    Loads the integrated dataset, engineers advanced features, and saves the
    final model-ready analytics base table.
    """
    print("--- Starting Advanced Feature Engineering Process ---")
    
    # --- 1. LOAD THE INTEGRATED DATASET ---
    try:
        df = pd.read_csv(input_path)
        print(f"Successfully loaded '{input_path}'. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"FATAL ERROR: The input file '{input_path}' was not found.")
        print("Please ensure the integrated dataset exists in the same directory.")
        return

    # --- 2. FEATURE EXTRACTION FROM DWT COEFFICIENTS ---
    print("\nExtracting features from DWT coefficients...")
    
    if 'dwt_coefficients' in df.columns:
        def parse_dwt(coeffs_str):
            try:
                # Safely evaluate the string representation of the list
                return ast.literal_eval(str(coeffs_str))
            except (ValueError, SyntaxError):
                # Return an empty list for malformed or non-string inputs
                return []

        # Apply the parsing function
        df['dwt_coefficients_list'] = df['dwt_coefficients'].apply(parse_dwt)

        # Create statistical features from the lists of coefficients
        df['dwt_mean'] = df['dwt_coefficients_list'].apply(lambda x: np.mean(x) if x else 0)
        df['dwt_std'] = df['dwt_coefficients_list'].apply(lambda x: np.std(x) if x else 0)
        df['dwt_max'] = df['dwt_coefficients_list'].apply(lambda x: np.max(x) if x else 0)
        df['dwt_min'] = df['dwt_coefficients_list'].apply(lambda x: np.min(x) if x else 0)
        df['dwt_range'] = df['dwt_max'] - df['dwt_min']
        print("Successfully created statistical features from DWT coefficients.")
    else:
        print("WARNING: 'dwt_coefficients' column not found. Skipping DWT feature extraction.")

    # --- 3. CREATE TIME-BASED & EFFICIENCY FEATURES ---
    print("\nCreating time-based and efficiency metrics...")
    
    # Time-based feature
    if 'deadline_sec' in df.columns and 'arrival_time' in df.columns:
        df['processing_time_allowed_sec'] = df['deadline_sec'] - df['arrival_time']
        df['processing_time_allowed_sec'] = df['processing_time_allowed_sec'].clip(lower=0)
    
    # Efficiency ratios
    # Adding a small epsilon to avoid division by zero
    if 'edge_cpu_mips' in df.columns and 'task_size_mb_x' in df.columns:
        df['compute_density_mips_per_mb'] = df['edge_cpu_mips'] / (df['task_size_mb_x'] + 1e-6)
    if 'treatmentcost' in df.columns and 'length_of_stay' in df.columns:
        df['cost_per_day'] = df['treatmentcost'] / (df['length_of_stay'].replace(0, 1) + 1e-6)

    # --- 4. CREATE HYBRID CLINICAL-TECHNICAL FEATURES ---
    print("\nEngineering hybrid clinical-technical priority scores...")
    scaler = MinMaxScaler()
    
    # Patient Acuity Score (Clinical Urgency)
    # A higher score means a more critical patient
    df['patient_acuity_score'] = (
        df['disease_severity'].fillna(3) + 
        (df['heart_rate'].fillna(df['heart_rate'].median()) / 50) + 
        (df['age'] / 100)
    )
    df['patient_acuity_score'] = scaler.fit_transform(df[['patient_acuity_score']])

    # Task Urgency Score (Technical Demand)
    # A higher score means a more demanding task
    df['task_urgency_score'] = (
        (1 / (df['deadline_sec'].fillna(df['deadline_sec'].median()) + 1)) * 10 + 
        df['task_size_mb_x'].fillna(df['task_size_mb_x'].median()) / 10
    )
    df['task_urgency_score'] = scaler.fit_transform(df[['task_urgency_score']])

    # Final Priority Score (Core DPTARA-Plus Logic)
    # Weighted sum to balance clinical needs and technical constraints
    clinical_weight = 0.6
    technical_weight = 0.4
    df['final_task_priority_score'] = (
        clinical_weight * df['patient_acuity_score'] + 
        technical_weight * df['task_urgency_score']
    )
    print("Successfully created patient acuity, task urgency, and final priority scores.")

    # --- 5. CATEGORICAL ENCODING ---
    print("\nApplying one-hot encoding to categorical features...")
    
    # Identify categorical columns with a reasonable number of unique values for encoding
    cols_to_encode = [
        'priority', 'assigned_resource', 'gender', 'severity', 
        'treatmenttype', 'bedtype', 'paymentmethod', 'health_status', 
        'disease_type', 'resource_needed'
    ]
    
    # Ensure columns exist before trying to encode them
    existing_cols_to_encode = [col for col in cols_to_encode if col in df.columns]
    
    df_encoded = pd.get_dummies(df, columns=existing_cols_to_encode, drop_first=True, dtype=float)
    print(f"Encoded {len(existing_cols_to_encode)} categorical columns.")
    
    # --- 6. FINAL CLEANUP & COLUMN SELECTION ---
    print("\nPerforming final cleanup and column selection for the model...")
    
    # Drop original objects, IDs, and redundant columns
    cols_to_drop = [
        'dwt_coefficients', 'dwt_coefficients_list', 'task_id', 'patient_id',
        'device_id', 'admissiondate', 'dischargedate', 'doctorname',
        'city', 'insuranceprovider', 'age_group', 'sensor_id', 'resource_id'
    ]
    
    # Add original categorical columns that were just encoded
    cols_to_drop.extend(existing_cols_to_encode)
    
    # Drop columns that exist in the dataframe
    final_cols_to_drop = [col for col in cols_to_drop if col in df_encoded.columns]
    df_final = df_encoded.drop(columns=final_cols_to_drop)
    
    print(f"Dropped {len(final_cols_to_drop)} unnecessary columns.")
    
    # --- 7. SAVE FINAL ANALYTICS BASE TABLE ---
    df_final.to_csv(output_path, index=False)

    print("\n--- Feature Engineering Complete! ---")
    print(f"Successfully created '{output_path}'")
    print("This file is the final, model-ready dataset.")
    print("\nFinal Dataset Preview:")
    print(df_final.head())
    print("\nFinal Dataset Shape:", df_final.shape)
    print("\nColumns in final dataset:", list(df_final.columns))

if __name__ == '__main__':
    engineer_features()

