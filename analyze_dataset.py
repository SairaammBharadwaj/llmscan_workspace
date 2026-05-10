import pandas as pd
import numpy as np

df=pd.read_csv(
    "data/advanced_causal_dataset_v6.csv"
)

print("\nDataset Shape:")
print(df.shape)

print("\nClass Distribution:")
print(df.iloc[:, -1].value_counts())

print("\nNaN Count:")
print(df.isnull().sum().sum())

features=df.iloc[:, :-1]

print("\nFeature Variance:")
print(features.var().describe())

safe=df[
    df.iloc[:, -1]==0
]

unsafe=df[
    df.iloc[:, -1]==1
]

safe_mean=safe.iloc[:, :-1].mean()

unsafe_mean=unsafe.iloc[:, :-1].mean()

difference=(
    unsafe_mean -
    safe_mean
).abs()

print("\nTop 20 Most Different Features:")

print(
    difference.sort_values(
        ascending=False
    ).head(20)
)