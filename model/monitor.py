import numpy as np
import pandas as pd
import joblib
import os

FEATURES = ["recency", "frequency", "monetary"]
N_BUCKETS = 5  # quintiles

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    month_cols = [c for c in df.columns if "2025" in c]

    def get_recency(row):
        for i, col in reversed(list(enumerate(month_cols))):
            if row[col] > 0:
                return len(month_cols) - i
        return len(month_cols) + 1

    out = df.copy()
    out["recency"] = out.apply(get_recency, axis=1)
    out["frequency"] = (out[month_cols] > 0).sum(axis=1)
    out["monetary"] = out[month_cols].sum(axis=1)
    return out

def _psi_for_feature(reference: np.ndarray, current: np.ndarray, buckets: int = N_BUCKETS) -> float:
    ref_range = reference.max() - reference.min()

    if ref_range <= 20:
        # Narrow integer range -> equal-width bins across observed range
        edges = np.linspace(reference.min(), reference.max(), buckets + 1)
        edges = np.unique(edges)
    else:
        quantiles = np.linspace(0, 1, buckets + 1)
        edges = np.unique(np.quantile(reference, quantiles))

    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(len(current), 1), 1e-6, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)

def check_drift(reference_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    ref_rfm = compute_rfm(reference_df)
    new_rfm = compute_rfm(new_df)

    result = {"features": {}, "overall_drift": False}
    for feat in FEATURES:
        psi = _psi_for_feature(ref_rfm[feat].values, new_rfm[feat].values)
        flagged = psi > 0.25
        result["features"][feat] = {
            "psi": round(psi, 4),
            "status": "significant" if psi > 0.25 else ("moderate" if psi > 0.10 else "stable"),
        }
        if flagged:
            result["overall_drift"] = True

    return result


if __name__ == "__main__":
    import mlflow

    mlflow.set_experiment("customer-segmentation")
    with mlflow.start_run(run_name="drift_check_self_test"):
        ref = pd.read_csv("data/customers.csv", sep=";", skiprows=1)
        result = check_drift(ref, ref)
        print("Self-comparison sanity check (expect ~0 PSI on all features):")
        for feat, info in result["features"].items():
            print(f"  {feat}: PSI={info['psi']} ({info['status']})")
            mlflow.log_metric(f"psi_{feat}", info["psi"])
        mlflow.log_param("overall_drift", result["overall_drift"])
        print(f"Overall drift detected: {result['overall_drift']}")