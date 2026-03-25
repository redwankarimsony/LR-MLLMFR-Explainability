#%% Load libraries
import os
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import numpy as np
import pandas as pd
import json
import glob
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import euclidean_distances

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


#%% ===============================
# Helper Functions
#=================================

def load_jsonl_embeddings(jsonl_path):
    embeddings = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            pid = item["pair_id"]
            emb = np.array(item["embedding"], dtype=float)
            embeddings[pid] = emb
    return embeddings


def project_embeddings(X, method="tsne", random_state=42):
    """
    Projection ONLY for visualization.
    """
    if method == "tsne":
        proj = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate=200,
            init="random",
            random_state=random_state
        )
        return proj.fit_transform(X)

    elif method == "pca":
        proj = PCA(n_components=2, random_state=random_state)
        return proj.fit_transform(X)

    elif method == "umap":
        if not HAS_UMAP:
            raise ImportError("UMAP is not installed.")
        proj = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=random_state
        )
        return proj.fit_transform(X)

    else:
        raise ValueError("Unknown projection method")


#%% ===============================
# Separation Metrics (CORE PART)
#=================================

def inter_intra_distance_ratio(X, y):
    """
    Inter / intra class distance ratio.
    Higher is better.
    """
    D = euclidean_distances(X)

    same = []
    diff = []

    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if y[i] == y[j]:
                same.append(D[i, j])
            else:
                diff.append(D[i, j])

    return np.mean(diff) / np.mean(same)


def fisher_discriminant_ratio(X, y):
    """
    Fisher ratio for binary classes.
    """
    classes = np.unique(y)
    assert len(classes) == 2, "Fisher ratio defined for binary labels."

    X0 = X[y == classes[0]]
    X1 = X[y == classes[1]]

    mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
    var0 = np.mean(np.sum((X0 - mu0) ** 2, axis=1))
    var1 = np.mean(np.sum((X1 - mu1) ** 2, axis=1))

    return np.sum((mu0 - mu1) ** 2) / (var0 + var1)


def compute_separation_metrics(X, y):
    """
    Compute all separation metrics on ORIGINAL embeddings.
    """
    metrics = {}
    metrics["silhouette"] = silhouette_score(X, y)
    metrics["davies_bouldin"] = davies_bouldin_score(X, y)
    metrics["calinski_harabasz"] = calinski_harabasz_score(X, y)
    metrics["inter_intra_ratio"] = inter_intra_distance_ratio(X, y)
    metrics["fisher_ratio"] = fisher_discriminant_ratio(X, y)
    return metrics


#%% ===============================
# Load Dataset
#=================================

embedding_paths = glob.glob(".data/IJBS/Explanations-with-*/gpt-5.2*.jsonl")
METADATA_PATH = ".data/IJBS/ijbs_still_benchmark_scores_with_roc.csv"
# embedding_paths = glob.glob(".data/BUPT-CBFace/Explanations-with-*/*_embeddings.jsonl")
# METADATA_PATH = ".data/BUPT-CBFace/cbface_top100_pairs_scores_filtered.csv"

df = pd.read_csv(METADATA_PATH)
# df = df.dropna(subset=['ROC'])
print(f"Number of pairs with ROC scores: {len(df)}")
# pair_id_list = df["pair_id"].tolist()
# df.set_index("pair_id", inplace=True)

EXP_NAMES = [p.split("/")[-2].replace("Explanations-", "") for p in embedding_paths]
print("Found experiments:", *EXP_NAMES, sep="\n")
print(*embedding_paths, sep="\n")


#%% ===============================
# Main Loop
#=================================

results = []

for jsonl_path, exp_name in zip(embedding_paths, EXP_NAMES):
    print(f"\nProcessing: {exp_name}")

    emb_dict = load_jsonl_embeddings(jsonl_path)
    jsonl_filename = os.path.basename(jsonl_path)
    gen_model_name = jsonl_filename.split("_")[0]
    embedding_model_name = jsonl_filename.split("_")[1]

    X = []
    y = []
    valid_rows = []
    # for pid, emb in emb_dict.items():
    #     try:
    #         y.append(df.loc[pid, "label"])
    #         X.append(emb)
    #     except KeyError:
    #         # print(f"  Warning: pair_id {pid} not found in metadata, skipping.")
    #         continue
    
    for idx, row in df.iterrows():
        if row['pair_id'] not in emb_dict:
            print(f"  Warning: pair_id {row['pair_id']} in metadata not found in embeddings, skipping.")
        else:
            X.append(emb_dict[row['pair_id']])
            y.append(row['label'])
            valid_rows.append(idx)

    X = np.array(X)
    y = np.array(y)

    # ---- Compute metrics on ORIGINAL embeddings ----
    metrics = compute_separation_metrics(X, y)
    metrics["experiment"] = exp_name
    metrics['gen_model'] = gen_model_name
    metrics['embedding_model'] = embedding_model_name
    results.append(metrics)

    print("Separation metrics (original space):")
    for k, v in metrics.items():
        if k not in ["experiment", "gen_model", "embedding_model"]:
            print(f"  {k:25s}: {v:.4f}")
    

    # ---- Optional visualization ----
    plot_output_dir = f"results/openai/tSNEs/{exp_name}"
    plot_path = os.path.join(plot_output_dir, f"{gen_model_name}_{embedding_model_name}_tsne.png")
    plot_data_path = os.path.join(plot_output_dir, f"{gen_model_name}_{embedding_model_name}_tsne.csv")
    os.makedirs(plot_output_dir, exist_ok=True)

    coords = project_embeddings(X, method="tsne")  # change to pca / umap if desired
    df_valid = df.loc[valid_rows].copy()
    df_valid['tSNE1'] = coords[:, 0]
    df_valid['tSNE2'] = coords[:, 1]
    df_valid.to_csv(plot_data_path, index=False)
    print(f"Saved tSNE coordinates to \n{plot_data_path}\n\n")

    # Dataframe with the plot 

    plt.figure(figsize=(6, 5), dpi=150)
    plt.scatter(coords[:, 0], coords[:, 1], c=y, cmap="bwr", s=15, alpha=0.6)
    plt.title(f"{exp_name}_{gen_model_name}_{embedding_model_name}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.colorbar(label="0 = Impostor, 1 = Genuine")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    # plt.show()
    plt.close()




#%% ===============================
# Save metrics
#=================================

results_df = pd.DataFrame(results)
save_path = "results/openai/tSNEs/ijbs_cluster_separation_metrics.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
results_df.to_csv(save_path, index=False)
print(f"\nSaved separation metrics at {save_path}")