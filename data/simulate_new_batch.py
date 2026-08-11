import pandas as pd
import numpy as np

np.random.seed(42)

SOURCE = "data/customers.csv"
MONTH_COLS_HINT = "2025"
CHURN_FRACTION = 0.40       # 40% of clients simulate churn
CHURN_RECENT_MONTHS = 6     # zero out the last N months for churned clients
NOISE_STD = 0.15            # ~15% relative noise for the clean batch

def load_source():
    return pd.read_csv(SOURCE, sep=";", skiprows=1)


def make_clean_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Resample rows with small random noise on visit counts.
    Distribution shape should stay close to the original."""
    month_cols = [c for c in df.columns if MONTH_COLS_HINT in c]
    out = df.copy()

    for col in month_cols:
        noise = np.random.normal(loc=0, scale=NOISE_STD, size=len(out))
        # Multiplicative noise, rounded back to non-negative integers
        out[col] = np.clip(np.round(out[col] * (1 + noise)), 0, None).astype(int)

    return out

def make_drifted_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Zero out the most recent months for a random subset of clients,
    simulating churn. Leaves monetary's historical total mostly intact
    relative to a full resample, but tanks recency/frequency for that
    subset -- the PSI check should catch this on those two features."""
    month_cols = [c for c in df.columns if MONTH_COLS_HINT in c]
    out = df.copy()

    n_churned = int(len(out) * CHURN_FRACTION)
    churned_idx = np.random.choice(out.index, size=n_churned, replace=False)

    recent_cols = month_cols[-CHURN_RECENT_MONTHS:]
    out.loc[churned_idx, recent_cols] = 0

    return out

if __name__ == "__main__":
    source = load_source()

    clean = make_clean_batch(source)
    drifted = make_drifted_batch(source)

    clean.to_csv("data/new_batch_clean.csv", sep=";", index=False)
    drifted.to_csv("data/new_batch_drifted.csv", sep=";", index=False)

    n_churned = int(len(source) * CHURN_FRACTION)
    print(f"Wrote data/new_batch_clean.csv   ({len(clean)} rows, small noise, expect NO drift)")
    print(f"Wrote data/new_batch_drifted.csv ({len(drifted)} rows, {n_churned} clients churned, expect drift on recency/frequency)")