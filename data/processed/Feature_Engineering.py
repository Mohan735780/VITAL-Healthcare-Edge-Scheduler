import pandas as pd
import numpy as np

def perform_feature_engineering(input_filepath='integrated_dptara_plus_dataset.csv'):
    """
    Loads the integrated dataset and engineers advanced features by synthesizing
    clinical and technical data to produce a model-ready analytics base table.
    """
    print("--- Starting Advanced Feature Engineering ---")

    try:
        df = pd.read_csv(input_filepath)
        print(f"Successfully loaded '{input_filepath}'. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"FATAL: The input file '{input_filepath}' was not found.")
        print("Please ensure the integrated dataset from the previous step is in the same directory.")
        return
        
    # --- 1. Data Integrity and Pre-computation Cleanup ---
    # Ensure key numerical columns are treated as floats
    for col in ['treatmentcost', 'length_of_stay', 'task_size_mb_x', 'deadline_sec']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill any remaining NaNs with median to ensure calculations don't fail
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
            print(f"Filled remaining NaNs in '{col}' with median.")

    # --- 2. Engineer Advanced Clinical Features ---

    # Map health_status to a numerical scale for calculations
    health_status_map = {'Stable': 1, 'Recovering': 2, 'Unstable': 3, 'Critical': 4, 'Emergency': 5}
    df['health_status_score'] = df['health_status'].map(health_status_map).fillna(1) # Default to stable
    print("Created 'health_status_score' by mapping clinical status to a numerical scale.")

    # Create a composite Patient Acuity Score. This is a critical feature.
    # It combines age (normalized), disease severity, and health status score.
    # A higher score indicates a more critical patient requiring higher priority attention.
    df['patient_acuity_score'] = (df['age'] / df['age'].max()) + \
                                (df['disease_severity'] / 5.0) + \
                                (df['health_status_score'] / 5.0)
    print("Created 'patient_acuity_score' to quantify patient criticality.")

    # Calculate Treatment Intensity as cost per day.
    # This reflects the level of resource consumption for the patient.
    df['treatment_intensity'] = df['treatmentcost'] / (df['length_of_stay'].replace(0, 1))
    print("Created 'treatment_intensity' (cost per day).")


    # --- 3. Engineer Advanced Technical Features ---

    # Calculate Task Density: How much data must be processed per unit of time.
    # This is a better measure of computational pressure than size or deadline alone.
    df['task_density'] = df['task_size_mb_x'] / (df['deadline_sec'] + 1e-6) # Epsilon to avoid zero division
    print("Created 'task_density' (MB/sec).")

    # Calculate Network Overhead: The additional latency penalty for offloading to the cloud.
    # This helps the model decide if the cloud's power is worth the network delay.
    df['network_overhead'] = df['network_latency_cloud_ms'] - df['network_latency_edge_ms']
    print("Created 'network_overhead' (cloud vs edge latency).")


    # --- 4. Engineer Hybrid Features (The Core of DPTARA-Plus) ---

    # **Create the Final Task Priority Score.**
    # This is the most important feature. It multiplies the technical demand (task_density)
    # by the clinical urgency (patient_acuity_score). This directly connects the patient's
    # condition to the task's scheduling priority.
    df['final_task_priority_score'] = df['task_density'] * (1 + df['patient_acuity_score'])
    print("Created 'final_task_priority_score' - a key hybrid feature.")
    

    # --- 5. Encode and Finalize the Dataset ---
    print("Applying one-hot encoding to categorical features...")
    
    categorical_cols = [
        'priority', 'gender', 'department', 'severity', 'health_status', 
        'disease_type', 'resource_needed', 'treatmenttype', 'bedtype', 
        'paymentmethod', 'readmitted'
    ]
    
    cols_to_encode = [col for col in categorical_cols if col in df.columns]
    
    df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True, prefix=cols_to_encode)
    print(f"Encoded {len(cols_to_encode)} columns.")

    # Drop original, intermediate, and high-cardinality columns
    columns_to_drop = [
        # Identifiers
        'task_id', 'iot_device_id', 'device_id', 'patient_id', 'doctorname',
        'sensor_id', 'resource_id',
        # Redundant or high cardinality text
        'admissiondate', 'dischargedate', 'city', 'insuranceprovider', 'age_group',
        # Redundant technical columns
        'task_size_mb_y', 'deadline_s', 'priority_level',
        # Intermediate clinical columns
        'gender_res', 'health_status_score'
    ]
    
    existing_cols_to_drop = [col for col in columns_to_drop if col in df_encoded.columns]
    df_final = df_encoded.drop(columns=existing_cols_to_drop)
    print(f"Dropped {len(existing_cols_to_drop)} unnecessary columns.")

    # --- 6. Save the Final Analytics Base Table ---
    output_filepath = 'final_analytics_base_table.csv'
    df_final.to_csv(output_filepath, index=False)
    
    print("\n--- Feature Engineering Complete! ---")
    print(f"Successfully created '{output_filepath}'")
    print("This dataset is now fully preprocessed and ready for model training.")
    print("\nFinal Dataset Preview:")
    print(df_final.head())
    print("\nFinal Dataset Shape:", df_final.shape)

if __name__ == '__main__':
    perform_feature_engineering()

