import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

# load the dataset
df = pd.read_csv("data/customers.csv", sep=";", skiprows=1)

print(f"Loaded {len(df)} customers.")

# Calculate RFM features
month_cols = [c for c in df.columns if "2025" in c]

# recency: how many months since the last visit (lower = better)
def get_recency(row):
    for i, col in reversed(list(enumerate(month_cols))):
        if row[col] > 0:
            return len(month_cols) - i
    return len(month_cols) + 1  # never visited

# FREQUENCY: number of months with at least 1 visit
# MONETARY: total visits across all months

df["recency"] = df.apply(get_recency, axis=1)
df["frequency"] = (df[month_cols] > 0).sum(axis=1)
df["monetary"] = df[month_cols].sum(axis=1)

print("\n RFM Summary:")
print(df[["recency", "frequency", "monetary"]].describe().round(2))

# SCALE THE FEATURES 
# KMeans works with distances, so we need to scale
# so recency, frequency, monetary are on same range
rfm = df[["recency", "frequency", "monetary"]]
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# TRAIN KMEANS WITH 3 CLUSTERS 
# 3 clusters = Loyal, At Risk, Lost
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(rfm_scaled)

# LABEL THE CLUSTERS 
# Figure out which cluster number = which label
# by looking at the average frequency per cluster
cluster_summary = df.groupby('cluster')[['recency','frequency','monetary']].mean()
print("\n Cluster centers (before labeling):")
print(cluster_summary.round(2))

# Sort by frequency descending to assign labels
sorted_clusters = cluster_summary['frequency'].sort_values(ascending=False).index.tolist()
label_map = {
    sorted_clusters[0]: 'Loyal',
    sorted_clusters[1]: 'At Risk',
    sorted_clusters[2]: 'Lost'
}
df['segment'] = df['cluster'].map(label_map)

print("\n Segment distribution:")
print(df['segment'].value_counts())

print("\n Sample customers per segment:")
for seg in ['Loyal', 'At Risk', 'Lost']:
    sample = df[df['segment'] == seg][['client_name','frequency','monetary']].head(3)
    print(f"\n  {seg}:")
    print(sample.to_string(index=False))

# MODEL & SCALER 
os.makedirs('model', exist_ok=True)
joblib.dump(kmeans,  'model/kmeans_model.pkl')
joblib.dump(scaler,  'model/scaler.pkl')
joblib.dump(label_map, 'model/label_map.pkl')

# Save results
df[['client_id','client_name','recency','frequency','monetary','segment']]\
    .to_csv('data/customers_segmented.csv', index=False)

print("\n Model saved to model/kmeans_model.pkl")
print(" Results saved to data/customers_segmented.csv")
print("\n Training complete!")