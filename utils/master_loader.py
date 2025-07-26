from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import pandas as pd
from utils.io_utils import read_csv_file
from utils.cleaning import clean_application_data


def clean_all_home_credit_tables():
    # === 1. Define tables to load ===
    tables = {
        'application': read_csv_file('application_train.csv'),
        'application_test': read_csv_file('application_test.csv'),
        'bureau': read_csv_file('bureau.csv'),
        'bureau_balance': read_csv_file('bureau_balance.csv'),
        'previous_application': read_csv_file('previous_application.csv'),
        'credit_card_balance': read_csv_file('credit_card_balance.csv'),
        'pos_cash': read_csv_file('POS_CASH_balance.csv'),
        'installments': read_csv_file('installments_payments.csv'),
    }

    # === 2. Define output container ===
    cleaned = {}

    # === 3. Settings ===
    use_knn = False           # Turn off KNN to avoid slowness
    n_neighbors = 5           # Only used if KNN is enabled
    skip_cols = ['SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'TARGET']

    # === 4. Clean each table ===
    for name, df in tables.items():
        print(f"\n Cleaning {name}.csv")

        # Identify drop candidates
        drop_cols = df.columns[df.isnull().mean() > 0.7].tolist()

        # Identify categorical and numeric columns
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        num_cols = df.select_dtypes(
            include=['int64', 'float64']).columns.tolist()

        # Exclude ID/TARGET cols from numeric processing
        num_cols = [col for col in num_cols if col not in skip_cols]

        # Remove dropped cols from cat/num to avoid KeyErrors
        cat_cols = [
            col for col in cat_cols if col in df.columns and col not in drop_cols]
        num_cols = [
            col for col in num_cols if col in df.columns and col not in drop_cols]

        # Clean with current configuration
        df_clean = clean_application_data(
            df=df,
            drop_cols=drop_cols,
            cat_cols=cat_cols,
            num_cols=num_cols,
            use_knn=use_knn,
            n_neighbors=n_neighbors
        )

        cleaned[name] = df_clean

    return cleaned


def knn_impute_selected(df, columns, n_neighbors=5):
    df = df.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[columns])

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed = imputer.fit_transform(scaled)

    df[columns] = pd.DataFrame(imputed, columns=columns, index=df.index)
    return df
