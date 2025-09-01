import pandas as pd
import glob

def audit_dataset(file_path):
    """
    Performs a basic data audit on a single CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        str: A formatted string containing the audit summary.
    """
    try:
        df = pd.read_csv(file_path)
        report = []
        report.append(f"--- Data Audit Report for: {file_path.split('/')[-1]} ---")
        report.append(f"\nShape (Rows, Columns): {df.shape}")
        
        report.append("\nFirst 5 Rows:")
        report.append(df.head().to_string())
        
        report.append("\nData Types and Non-Null Counts:")
        info_str = df.info(verbose=True, buf=None)
        # The info() method prints directly, so we capture it.
        # This is a bit of a workaround to get the string representation.
        import io
        buf = io.StringIO()
        df.info(buf=buf)
        report.append(buf.getvalue())

        report.append("\nMissing Values Count:")
        report.append(df.isnull().sum().to_string())
        
        report.append(f"\nNumber of Duplicate Rows: {df.duplicated().sum()}")
        
        report.append("\nStatistical Summary (Numerical Columns):")
        report.append(df.describe().to_string())
        
        report.append("\nUnique Values Count (Categorical Columns):")
        # Select object columns for nunique count
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            report.append(df[categorical_cols].nunique().to_string())
        else:
            report.append("No categorical columns found.")
            
        report.append("\n" + "="*50 + "\n")
        
        return "\n".join(report)

    except Exception as e:
        return f"Could not process {file_path}. Error: {e}\n" + "="*50 + "\n"

def run_full_audit():
    """
    Runs the data audit on all CSV files in the current directory.
    """
    # Find all csv files in the current directory
    csv_files = glob.glob('*.csv')
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        print("Please ensure your datasets are in the same folder as this script.")
        return

    print(f"Found {len(csv_files)} CSV files to audit: {', '.join(csv_files)}")
    
    # Open a file to write the full report
    with open("full_data_audit_report.txt", "w") as f:
        for file in csv_files:
            print(f"Auditing {file}...")
            report_str = audit_dataset(file)
            f.write(report_str)
    
    print("\nAudit complete! The full report has been saved to 'full_data_audit_report.txt'.")
    print("Please review this report carefully to understand the state of your data.")


if __name__ == '__main__':
    run_full_audit()
