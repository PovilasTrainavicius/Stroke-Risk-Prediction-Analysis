import pandas as pd

def preprrocess_data(data):
    """
    Cleans and preprocesses the input DataFrame by normalizing column names and string values,
    reporting missing values and duplicates in the 'id' column, and dropping 'id' if no duplicates are found.
    Args:
        data (pd.DataFrame): Input data to preprocess.
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """

    data = data.copy()
    missing_values = data.isnull().sum()
    duplicates = data["id"].duplicated().sum()    
    
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].str.lower().str.replace(' ', '_').str.replace('-', '_')
            
    if missing_values.any():
        for col, count in missing_values.items():
            if count > 0:
                percentage = (count / len(data)) * 100
                print(f"Column '{col}' has {count} missing values ({percentage:.2f}%)")
    else:
        print("No missing values found in the dataset.")

    if duplicates > 0:
        print(f"There are {duplicates} duplicate entries in the 'id' column.")
    else:
        print("No duplicate entries found in the 'id' column. Column 'id' will be dropped.")
        data = data.drop(columns=["id"])
    
    return data