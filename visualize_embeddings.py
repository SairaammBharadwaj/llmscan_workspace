import pandas as pd
import matplotlib.pyplot as plt
import umap

df=pd.read_csv(
    "data/advanced_causal_dataset_v4.csv"
)

X=df.iloc[:, :-1].values
y=df.iloc[:, -1].values

embedding=umap.UMAP(
    n_components=2,
    random_state=42
).fit_transform(X)

plt.figure(figsize=(10,8))

scatter=plt.scatter(
    embedding[:,0],
    embedding[:,1],
    c=y
)

plt.title(
    "Causal Feature Space"
)

plt.savefig(
    "feature_space.png"
)

print(
    "Saved feature_space.png"
)