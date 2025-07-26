from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd

def clean_application_data(df, drop_cols=None, cat_cols=None, num_cols=None, use_knn=True, n_neighbors=5):
    df = df.copy()

    # === 1. Drop high-missing or low-value columns ===
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # === 2. Add missing flags for numeric columns (vectorized to avoid fragmentation) ===
    if num_cols:
        missing_flags = {
            f"{col}_MISSING": df[col].isnull().astype(int)
            for col in num_cols
        }
        df = pd.concat([df, pd.DataFrame(missing_flags, index=df.index)], axis=1)

    # === 3. Handle categorical imputation ===
    if cat_cols:
        for col in cat_cols:
            df[col] = df[col].fillna("Missing")

    # === 4. Handle numeric imputation ===
    if use_knn and num_cols:
        # Standardize before KNN
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df[num_cols])

        # KNN Imputer
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_array = imputer.fit_transform(scaled_array)

        # Replace original columns with imputed values (back in original scale)
        df[num_cols] = pd.DataFrame(imputed_array, columns=num_cols, index=df.index)
    elif num_cols:
        # Median fallback
        median_imputer = SimpleImputer(strategy='median')
        df[num_cols] = median_imputer.fit_transform(df[num_cols])

    return df
