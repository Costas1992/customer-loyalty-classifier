import pandas as pd
import joblib
import os
import mlflow
from monitor import check_drift, compute_rfm

REFERENCE_PATH = "data/customers.csv"
NEW_BATCH_PATH = "data/new_batch_drifted.csv"
N_CLUSTERS = 3


def merge_by_client_id(reference_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Combine old and new data. For any client_id present in both,
    the new observation wins. Client_ids only in reference are kept
    as-is. This is an upsert, not a plain concatenation -- both
    datasets share the same client_ids here since new_df is resampled
    from the same source."""
    combined = pd.concat([reference_df, new_df])
    # Keep the LAST occurrence of each client_id -- since new_df
    # is concatenated after reference_df, "last" means "new data wins"
    deduped = combined.drop_duplicates(subset="client_id", keep="last")
    return deduped.reset_index(drop=True)


def retrain_if_drifted(reference_path: str = REFERENCE_PATH, new_batch_path: str = NEW_BATCH_PATH):
    reference_df = pd.read_csv(reference_path, sep=";", skiprows=1)
    new_df = pd.read_csv(new_batch_path, sep=";")

    print("Checking for drift...")
    drift_result = check_drift(reference_df, new_df)
    for feat, info in drift_result["features"].items():
        print(f"  {feat}: PSI={info['psi']} ({info['status']})")

    if not drift_result["overall_drift"]:
        print("No significant drift detected. Skipping retrain.")
        return None

    print("Drift detected -- retraining on combined old + new data.")
    combined_df = merge_by_client_id(reference_df, new_df)
    print(f"Combined dataset: {len(combined_df)} customers ({len(new_df)} new/updated).")

    return combined_df


import shutil
from datetime import datetime


def write_combined_data(combined_df: pd.DataFrame, target_path: str = "data/customers_after_retrain.csv"):
# NOTE: writes to a separate file rather than overwriting customers.csv directly,
# so retrain runs are safe to repeat without risking the canonical dataset.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = target_path.replace(".csv", f"_backup_{timestamp}.csv")
    shutil.copy(target_path, backup_path)
    print(f"Backed up original data to {backup_path}")

    # customers.csv has a title row above the real header (that's why every
    # read in this project uses skiprows=1). Preserve that same structure
    # when writing back, or train.py's skiprows=1 will skip real data instead.
    with open(target_path, "w") as f:
        f.write("clients_month_attendance\n")
    combined_df.to_csv(target_path, sep=";", mode="a", index=False)
    print(f"Wrote {len(combined_df)} customers back to {target_path}")


if __name__ == "__main__":
    combined = retrain_if_drifted()

    if combined is not None:
        write_combined_data(combined)

        mlflow.set_experiment("customer-segmentation")
        with mlflow.start_run(run_name="retrain_trigger"):
            mlflow.log_param("retrain_triggered", True)
            mlflow.log_metric("n_customers_after_merge", len(combined))

        print("Retraining model on updated data...")
        os.system("python model/train.py")
    else:
        mlflow.set_experiment("customer-segmentation")
        with mlflow.start_run(run_name="retrain_trigger"):
            mlflow.log_param("retrain_triggered", False)