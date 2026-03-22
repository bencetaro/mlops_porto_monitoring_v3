import pandas as pd
import numpy as np
import argparse, os, joblib, logging, zipfile
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Start data preparation script.")

    train_pth = os.path.join(input_dir, "train.csv.zip")
    test_pth = os.path.join(input_dir, "test.csv.zip")
    assert os.path.exists(train_pth) or os.path.exists(test_pth), "Both 'train.csv.zip' and 'test.csv.zip' must exist in the input directory."

    with zipfile.ZipFile(train_pth, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    with zipfile.ZipFile(test_pth, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    logging.info("Loading datasets...")
    train = pd.read_csv(os.path.join(output_dir, "train.csv"))
    test = pd.read_csv(os.path.join(output_dir, "test.csv"))
    
    # Separate ID
    test_id = test["id"]
    train.drop("id", axis=1, inplace=True)
    test.drop("id", axis=1, inplace=True)

    # Rename numeric columns
    num_cols = [c for c in train.columns if not (c.endswith("cat") or c.endswith("bin") or c.endswith("target"))]
    train.rename(columns={c: c + "_num" for c in num_cols}, inplace=True)
    test.rename(columns={c: c + "_num" for c in num_cols}, inplace=True)

    # Replace -1 with NaN
    train.replace(-1, np.nan, inplace=True)
    test.replace(-1, np.nan, inplace=True)
    logging.info(f"Missing values before imputation:\n{train.isna().sum().sort_values(ascending=False).head()}")

    # Drop features with >20% missing
    missing_pct = train.isna().sum() / len(train)
    cols_to_drop = missing_pct[missing_pct > 0.2].index
    train.drop(columns=cols_to_drop, inplace=True)
    test.drop(columns=cols_to_drop, inplace=True)
    logging.info(f"Dropped {len(cols_to_drop)} cols with >20% missing.")

    # Imputation
    cat_imputer = SimpleImputer(strategy="most_frequent")
    num_imputer = SimpleImputer(strategy="median")
    bin_imputer = SimpleImputer(strategy="most_frequent") 

    bin_cols = [c for c in train.columns if c.endswith("bin")]
    cat_cols = [c for c in train.columns if c.endswith("cat")]
    num_cols = [c for c in train.columns if c.endswith("num")]

    train[num_cols] = num_imputer.fit_transform(train[num_cols])
    test[num_cols] = num_imputer.transform(test[num_cols])
    train[cat_cols] = cat_imputer.fit_transform(train[cat_cols])
    test[cat_cols] = cat_imputer.transform(test[cat_cols])
    train[bin_cols] = np.round(bin_imputer.fit_transform(train[bin_cols]))
    test[bin_cols] = np.round(bin_imputer.transform(test[bin_cols]))
    logging.info(f"Missing values after imputation:\n{train.isna().sum().sort_values(ascending=False).head()}")

    # Scale numerical features
    logging.info("Scale and encode features...")
    scaler = StandardScaler()
    train[num_cols] = scaler.fit_transform(train[num_cols])
    test[num_cols] = scaler.transform(test[num_cols])

    # Label encode categoricals
    le_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train[cat_cols] = le_encoder.fit_transform(train[cat_cols])
    test[cat_cols] = le_encoder.transform(test[cat_cols])

    # Save processed data
    logging.info(f"Cleaned and prepocessed data will be saved in {output_dir}")
    train.to_csv(os.path.join(output_dir, "train_preprocessed.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test_preprocessed.csv"), index=False)
    test_id.to_csv(os.path.join(output_dir, "test_ids.csv"), index=False)
    joblib.dump((num_imputer, cat_imputer, bin_imputer, scaler, le_encoder, cols_to_drop, bin_cols, cat_cols, num_cols), os.path.join(output_dir, "preprocessors.pkl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)