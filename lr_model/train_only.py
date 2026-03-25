# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File : train_only.py
# @Author : Redwan Sony

import os
import argparse
import sys
import random
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit
from lr_model.utils.eval_utils import plot_llr_density
from lr_model.gmms import fit_single_gmm


def llr_to_probability(llr_scores, prior_genuine=0.5, temperature=100.0):

    prior_impostor = 1.0 - prior_genuine

    log_prior_odds = np.log(prior_genuine / prior_impostor)

    scaled_llr = llr_scores / temperature

    prob_genuine = expit(scaled_llr + log_prior_odds)

    return prob_genuine


def load_jsonl_embeddings(jsonl_path):
    """
    Load embeddings from a JSONL file.
    Args:
        jsonl_path (str): Path to the JSONL file.
    Returns:
        dict: A dictionary mapping pair_id to embedding numpy arrays.
    """

    embeddings = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            pid = item["pair_id"]
            emb = np.array(item["embedding"], dtype=float)
            embeddings[pid] = emb
    return embeddings



def load_data(dataframe_path, jsonl_path):

    """method to load the training data from a jsonl file

    Args:
        jsonl_path (str): Path of the training jsonl file
    Returns:
        np.ndarray, np.ndarray, pd.DataFrame: genuine and impostor features and the training dataframe
    """

    # Load the training dataframe
    df_train = pd.read_csv(dataframe_path)

    # Load the training embeddings
    embeddings_dict = load_jsonl_embeddings(jsonl_path)

    emb_genuine, emb_impostor, valid_rows = [], [], []
    for idx, row in df_train.iterrows():
        pid = row['pair_id']
        label = row['label']
        if pid in embeddings_dict:
            emb = embeddings_dict[pid]
            valid_rows.append(idx)
            if label == 1:
                emb_genuine.append(emb)
            else:
                emb_impostor.append(emb)
        else:
            print(f"Warning: pair_id {pid} in training dataframe not found in training embeddings, skipping.")

    emb_genuine = np.array(emb_genuine)
    emb_impostor = np.array(emb_impostor)

    return emb_genuine, emb_impostor, df_train.loc[valid_rows].copy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a likelihood ratio model using GMMs on the provided dataset.")
    parser.add_argument("--train_dataframe", type=str, default='.data/BUPT-CBFace/cbface_top100_pairs_scores_filtered.csv', help="Path to the training dataframe CSV file.")
    parser.add_argument("--train_embeddings", type=str, default='.data/BUPT-CBFace/Explanations-with-training-ground-truth/gpt-4o_text-embedding-3-small_embeddings.jsonl', help="Path to the training embeddings JSONL file.")
    parser.add_argument("--test_dataframe", type=str, default=".data/IJBS/ijbs_still_benchmark_scores_with_roc.csv", help="Path to the test dataframe CSV file.")
    # parser.add_argument("--test_embeddings", type=str, default="", help="Path to the test embeddings JSONL file.")
    parser.add_argument("--gen_model_name", type=str, default="gpt-4o", help="Name of the generative model used for explanations (e.g. gpt-4o, gemini-2.5-flash).")
    parser.add_argument("--embedding_model_name", type=str, default="text-embedding-3-small", help="Name of the embedding model used (e.g. text-embedding-3-small, text-embedding-3-large).")
    parser.add_argument("--experiment_name", type=str, default="with-kprpe-score-decision", help="Name of the experiment (e.g. with-gt, with-no-info, with-scores, with-scores-gt, with-kprpe-score-decision).")
    args = parser.parse_args()
    args.test_embeddings = f".data/IJBS/Explanations-{args.experiment_name}/{args.gen_model_name}_{args.embedding_model_name}_embeddings.jsonl"
    
    gen_model_name = args.gen_model_name
    embedding_model_name = args.embedding_model_name
    experiment_name = args.experiment_name

    # Output directory for results
    results_dir = f"results/lr-eval/{experiment_name}/{gen_model_name}_{embedding_model_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the dataset and preprocess the embeddings
    train_df_path = args.train_dataframe
    train_embeddings_path = args.train_embeddings
    train_genuine, train_impostor, train_df = load_data(train_df_path, train_embeddings_path)
    print(f"Training data loaded. \nGenuine: {train_genuine.shape}, Impostor: {train_impostor.shape}")

    # Test data loading
    test_dataframe_path = args.test_dataframe
    test_jsonl_path = args.test_embeddings
    test_genuine, test_impostor, test_df = load_data(test_dataframe_path, test_jsonl_path)
    print(f"Test data loaded. \nGenuine: {test_genuine.shape}, Impostor: {test_impostor.shape}")



    # ============================== PCA for dimensionality reduction before GMM fitting ==============================
    # Vstack the features to apply the PCA and then separate them again after the transformation
    all_features = np.vstack((train_genuine, train_impostor))
    pca = PCA(n_components=0.95, svd_solver='full')
    all_features = pca.fit_transform(all_features)
    print(f"All features shape after PCA: {all_features.shape}")
    pca_model_path = os.path.join(results_dir, "pca_model.pkl")
    joblib.dump(pca, pca_model_path)

    print("Saved PCA model to pca_model.pkl")

    labels = np.array([1]*train_genuine.shape[0] + [0]*train_impostor.shape[0])

    genuine_features_compressed = all_features[labels == 1]
    impostor_features_compressed = all_features[labels == 0]

    print(f"Genuine features after PCA shape: {genuine_features_compressed.shape}")
    print(f"Impostor features after PCA shape: {impostor_features_compressed.shape}")

    # Fit GMMs to the genuine and impostor features
    gmm_genuine = fit_single_gmm(genuine_features_compressed, n_components=4, covariance_type='full')
    gmm_impostor = fit_single_gmm(impostor_features_compressed, n_components=4, covariance_type='full')

    # Save the GMM models
    joblib.dump(gmm_genuine, os.path.join(results_dir, "gmm_genuine_model.pkl"))
    joblib.dump(gmm_impostor, os.path.join(results_dir,"gmm_impostor_model.pkl"))

    test_features = np.vstack((test_genuine, test_impostor))
    print(f"Test all features shape before PCA: {test_features.shape}")

    # Apply the saved PCA transformation
    pca = joblib.load(pca_model_path)
    test_features = pca.transform(test_features)
    print(f"Test all features shape after PCA: {test_features.shape}")

    test_features= test_features

    # Compute the log-likelihood ratios for the test dataset
    llr_genuine = gmm_genuine.score_samples(test_features)
    llr_impostor = gmm_impostor.score_samples(test_features)

    llr = llr_genuine - llr_impostor
    print(f"LLR scores computed for test data. Shape: {llr.shape}")
    # Print the minimum and maximum LLR scores
    print(f"LLR Score - Min: {np.min(llr)}, Max: {np.max(llr)}")
    # sys.exit(0)

    # MinMax normalize the LLR scores to [0, 1]
    scaler = MinMaxScaler()
    llr_scores_norm = scaler.fit_transform(llr.reshape(-1, 1)).flatten()


    prob_gen = llr_to_probability(llr, prior_genuine=0.5, temperature=500.0)
    llr_scores_norm = prob_gen  # Use the posterior probability of genuine as the final score

    # Create the test labels
    test_labels = np.array([1]*test_genuine.shape[0] + [0]*test_impostor.shape[0])     


    # Save the normalized LLR scores and labels to a CSV file for later analysis
    test_df['LLR_Score'] = llr
    test_df['LLR_Score_Normalized'] = llr_scores_norm
    test_df.to_csv(os.path.join(results_dir, "test_scores.csv"), index=False)
    print(f"Saved test scores and labels to {os.path.join(results_dir, 'test_scores.csv')}")

    # llr_scores_norm = llr
    y_val = test_labels

    # df[model_name] = llr_scores_norm
    # llr_scores_norm = (df[model_name] + df['KPRPE'])/2
    # df.to_csv(test_dataframe_path, index=False)

    print(len(llr_scores_norm), len(y_val), "Test Scores stat")
    plot_llr_density(llr_scores_norm, y_val,
                     save_name=os.path.join(results_dir, "llr_density.png"),
                     title='Normalized LLR Score Density (Covariance:)',
                     dpi=200, show_plot=False)

    print("sample radom scores:")

    # Sample 15 random indices
    random_indices = random.sample(range(len(llr_scores_norm)), 15)

    # Header
    print(f"{'Index':<8} {'LLR Score':<15} {'Label':^7}")
    print("-" * 36)

    # Rows
    for idx in random_indices:
        print(f"{idx:<8} {llr_scores_norm[idx]:<15.4f} {str(y_val[idx]):^7}")

    # Plot the ROC curve with log-scaled x-axis
    fpr, tpr, thresholds = roc_curve(y_val, llr_scores_norm)
    print("ROC curve data computed.", len(fpr), len(tpr))


    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # log scale for x-axis
    plt.xscale('log')
    plt.xlim([0.00001, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig( os.path.join(results_dir, f"roc_curve.png"), dpi=200)
    plt.close()

    # Save the ROC data to a CSV file
    roc_data = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr,
        'Thresholds': thresholds})
    roc_data.to_csv(os.path.join(results_dir, f"roc_data.csv"), index=False)
    print(f"ROC data saved to roc_data.csv")


    # Take llr_scores_norm and y_val and then plot the distributions of genuine and impostor scores
    llr_genuine = llr_scores_norm[y_val == 1]
    llr_impostor = llr_scores_norm[y_val == 0]

    plt.figure(figsize=(8, 6), dpi=150)
    plt.hist(llr_genuine, bins=100, alpha=0.6, color='blue', label='Genuine')
    plt.hist(llr_impostor, bins=100, alpha=0.6, color='red', label='Impostor')
    plt.title('LLR Score Distributions')
    plt.xlabel('LLR Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"llr_score_distributions.png"), dpi=200) 
    plt.close()


    # dff = pd.read_csv(".data/IJBS/ijbs_still_benchmark_scores.csv")
    # # Add a new column with the model name first with empty values first
    # dff[experiment_name] = np.nan

    # # Now update the scores based on pair ids because some pair ids might be missing
    # for pid, score in zip(pair_ids, llr_scores_norm):
    #     dff.loc[dff['pair_id'] == pid, model_name+"_"+experiment_name ] = score

    # dff.to_csv(f".data/IJBS/ijbs_still_benchmark_scores.csv", index=False)


