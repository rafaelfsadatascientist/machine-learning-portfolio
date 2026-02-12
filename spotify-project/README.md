# Spotify Tracks Clustering Project

## Author
**Rafael Severiano**

---

## Overview

This project aims to cluster Spotify tracks based on audio features extracted from the dataset. The goal is to create relatively coherent groups of songs that share musical characteristics. These clusters can then be used as a **recommendation system** or for **musical analysis**.

Music is complex and subjective, and the features available in the dataset may not perfectly separate tracks into distinct groups. Therefore, the objective is not to build a perfect model, but to demonstrate reasonable success in grouping tracks and analyzing patterns.

---

## Dataset

- **Source:** Spotify Tracks Dataset (Kaggle)  
- **Link:** https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset  

---

## Problem Framing

This is an **unsupervised learning** problem where we try to discover natural groupings in the data.

- **Task:** Clustering
- **Approach:** KMeans clustering with PCA for dimensionality reduction
- **Features used:** Audio features such as danceability, energy, valence, instrumentalness, etc.
- **Exclusions:** Metadata features such as popularity and recording context features (e.g., liveness)

---

## Methodology

### 1. Data Cleaning
- Remove duplicates
- Handle missing values
- Encode cyclical features (`key` transformed using sine/cosine)

### 2. Feature Selection
Keep only intrinsic audio features and remove metadata.

### 3. Preprocessing
- **Categorical features:** One-Hot Encoding
- **Numerical features:** Standard Scaling
- **Dimensionality reduction:** PCA (5 components)

### 4. Clustering
- KMeans clustering
- Evaluate K using **inertia**
- Final clustering with **k = 20**

---

## Outputs and Visualizations

The project generates several figures stored in the `figures/` directory:

| Figure | Description |
|--------|-------------|
| `feature_correlation_heatmap.png` | Correlation between audio features |
| `variance_analysis_total_dataset.png` | Variance of selected features |
| `k_selection.png` | Inertia plot for choosing K |
| `mean_per_cluster_bar.png` | Mean feature values per cluster (bar plot) |
| `mean_per_cluster_matrix.png` | Mean feature values per cluster (heatmap) |
| `variance_per_cluster.png` | Variance per cluster |
| `group_1_analysis.png` | Cluster distribution for heavy genres |
| `group_2_analysis.png` | Cluster distribution for classical/ambient |
| `group_3_analysis.png` | Cluster distribution for dance/disco |

---

## Key Findings

- Some clusters successfully capture clear patterns such as:
  - **High energy + low valence** → metal/rock clusters
  - **High instrumentalness + low speechiness** → classical/ambient clusters
  - **High danceability** → dance/disco clusters

---

## Limitations

- Music is complex and subjective; features may not fully represent style.
- Alternative approaches like **DBSCAN** could improve results.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
