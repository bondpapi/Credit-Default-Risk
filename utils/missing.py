# utils/missing_utils.py

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def profile_missing(df, name=""):
    total = df.isnull().sum()
    percent = total / len(df)
    dtype = df.dtypes
    missing = pd.concat([total, percent, dtype], axis=1)
    missing.columns = ['MissingCount', 'MissingRatio', 'Dtype']
    missing = missing[missing.MissingCount > 0].sort_values(by='MissingRatio', ascending=False)
    print(f"\n{name} — {missing.shape[0]} columns with missing values")
    return missing


def knn_imputation_pipeline(df, categorical_cols, numeric_cols, n_neighbors=5):
    df = df.copy()
   
    # Track where imputation occurred
    for col in numeric_cols:
        df[f"{col}_MISSING"] = df[col].isnull().astype(int)

    transformer = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ])

    X_encoded = transformer.fit_transform(df)
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_imputed = imputer.fit_transform(X_encoded)

    return pd.DataFrame(X_imputed)
