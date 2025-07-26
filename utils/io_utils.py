import pandas as pd

def read_csv_file(path, encoding='utf-8', verbose=True):
    """
    Reads a CSV file with default encoding and basic error handling.

    Args:
        path (str): File path.
        encoding (str): File encoding. Default is 'utf-8'.
        verbose (bool): If True, prints basic info.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(path, encoding=encoding)
        if verbose:
            print(f"Loaded '{path}' — shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")
        return pd.DataFrame()