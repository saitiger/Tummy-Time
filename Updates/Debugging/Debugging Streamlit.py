if isinstance(df, str):  # Checking if df is an error message
  st.error(df)
  
if len(df.columns) == 4:
    df.columns = ['A', 'B', 'C', 'D']
else:
    raise ValueError(f"Expected 4 columns, but got {len(df.columns)}")
 
expected_columns = 4
    if df.shape[1] == expected_columns:
        df.columns = ['A', 'B', 'C', 'D']
    else:
        raise ValueError(f"Expected {expected_columns} columns, but got {df.shape[1]}")

try:
        # Attempt to read the file directly with pandas
        df = pd.read_csv(file, encoding='utf-8', skiprows=99, on_bad_lines='skip', header=None)
except UnicodeDecodeError:
        try:
            # If UTF-8 fails, try 'latin-1'
            df = pd.read_csv(file, encoding='latin-1', skiprows=99, on_bad_lines='skip', header=None)
        except UnicodeDecodeError:
            return "Error: Unable to decode the file. Please ensure it's a valid CSV."
