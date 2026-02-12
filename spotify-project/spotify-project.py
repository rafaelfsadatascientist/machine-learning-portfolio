# ============================================================
# Spotify Tracks Clustering Project
# Author: Rafael Severiano
# ============================================================

# In this project, we group Spotify audio tracks based on a set of their audio features.
# These groups can be used as a recommendation system or for musical analysis.
#
# Music is extremely complex, and individual taste is highly subjective. In addition,
# the features used in this project may not be the most effective for achieving clear
# separation between groups. Therefore, the goal is not to build a perfect model, but
# rather to create one that demonstrates reasonable success in some aspects.
#
# After building the model, we analyze the results and discuss its limitations.
# Our analysis will focus on a specific subset of features.

# ------------------------------
# Import libraries
# ------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
# ------------------------------
# Data Loading & Cleaning
# ------------------------------

df = pd.read_csv("data/dataset-Spotify.csv")
df = df.drop_duplicates()
print(df.describe())
print(df.info())
print(df.isna().sum())
# We remove the 'Unnamed: 0' column from the dataset. In addition, since musical keys
# are cyclical in nature, we apply a cyclical transformation to the 'key' feature
# to avoid distorted distance relationships between notes.
#
# Undefined keys (represented by -1) are first replaced with the most
# common key in the dataset. Then, sine and cosine transformations are applied
# to properly encode the cyclical structure of the musical scale.

most_common_key = df.loc[df['key'] != -1, 'key'].mode()[0]
df['key'] = df['key'].replace(-1, most_common_key)
df['key_sin'] = np.sin(2 * np.pi * df['key']/12)
df['key_cos'] = np.cos(2 * np.pi * df['key']/12)
df = df.drop(columns=['key','Unnamed: 0'])

# Select columns for clustering. We keep only features that describe the intrinsic
# musical characteristics of the track, excluding metadata such as popularity
# and features related to recording context. In particular, 'liveness' estimates
# the presence of an audience in the recording, where higher values indicate a
# greater probability that the track was performed live (values above 0.8 strongly
# suggest a live performance).

X = df.drop(['track_id','artists','album_name','track_name','popularity','liveness'], axis=1)

# ------------------------------
# Correlation heatmap
# ------------------------------

plt.figure(figsize=(10,10))
sns.heatmap(X.corr(numeric_only=True), annot=True, cmap='Spectral', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_correlation_heatmap.png", dpi=200, bbox_inches="tight")
plt.show()

# As expected, loudness is strongly positively correlated with energy and
# negatively correlated with acousticness and instrumentalness .
#
# Valence shows a reasonably positive correlation with danceability, which
# is also plausible, as more danceable tracks tend to convey a happier mood.
#
# Instrumentalness presents a slightly negative correlation with valence,
# which also makes sense: highly instrumental tracks can be calmer (e.g.,
# classical music) and may convey a more somber or less euphoric mood.

# ------------------------------
# Variance analysis 
# ------------------------------

features = ['danceability', 'energy', 'valence', 'instrumentalness', 'speechiness']

variances_total = X[features].var()
colors = plt.cm.viridis(np.linspace(0, 1, len(variances_total)))

plt.figure(figsize=(10,5))
for i, feature in enumerate(features):
    plt.bar(feature, variances_total[feature], color=colors[i], label=feature)

plt.ylabel('Variance')
plt.title('Variance of Selected Audio Features (Total Dataset)')
plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(FIG_DIR / "variance_analysis_total_dataset.png", dpi=200, bbox_inches="tight")
plt.show()

# ------------------------------
# Feature preprocessing
# ------------------------------

# We use PCA due to the large number of columns that will be created after
# applying one-hot encoding. Given the high number of distinct genres,
# dimensionality reduction is required to make the problem more tractable.

cat = ['explicit','track_genre']  
num = ['duration_ms','danceability','energy','loudness',
       'speechiness','acousticness','instrumentalness','valence','tempo']  
passthrough = ['key_sin','key_cos','time_signature','mode']  
t1 = ('cat', OneHotEncoder(), cat)
t2 = ('num', StandardScaler(), num)
t3 = ('passthrough', 'passthrough', passthrough)
preprocessor = ColumnTransformer(transformers=[t1,t2,t3])

# ----------------------------------------------------------
# Determine optimal K 
# ----------------------------------------------------------

inertia = []
K_range = range(2, 41)

for k in K_range:
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=5)),
        ('kmeans', KMeans(n_clusters=k, random_state=42))
    ])
    pipeline.fit(X)
    inertia.append(pipeline['kmeans'].inertia_)
 
plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, 'o-', color='blue')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('K Selection')
plt.tight_layout()
plt.savefig(FIG_DIR / "k_selection.png", dpi=200, bbox_inches="tight")
plt.show()

# We choose k = 20 because at this point inertia has significantly decreased,
# and further increases in k lead to less significant improvements.
# This value also allows for more detailed musical segmentation.

# ------------------------------
# Training
# ------------------------------

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=5)),
    ('kmeans', KMeans(n_clusters=20, random_state=42))
])
pipeline.fit(X)
df['Cluster'] = pipeline['kmeans'].labels_

# ------------------------------
# Mean of features per cluster
# ------------------------------

cluster_mean = df.groupby('Cluster')[features].mean()

plt.figure(figsize=(20,15))
colors = plt.cm.viridis(np.linspace(0,1,len(features)))

for i, feature in enumerate(features):
    plt.bar(cluster_mean.index + i*0.15, cluster_mean[feature], width=0.15, color=colors[i], label=feature)

plt.xlabel('Cluster')
plt.ylabel('Mean Feature Value')
plt.title('Mean Selected Audio Features per Cluster')
plt.xticks(cluster_mean.index + 0.15, cluster_mean.index) 
plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(FIG_DIR / "mean_per_cluster_bar.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(20, 20))
sns.heatmap(
    cluster_mean,
    annot=True,
    fmt=".2f",
    cmap='viridis',
    linewidths=0.5,
    linecolor='white',
    annot_kws={"size": 12},
    cbar_kws={"shrink": 0.8}
)
plt.title("Mean Selected Audio Features per Cluster", fontsize=20)
plt.xlabel("Features", fontsize=16)
plt.ylabel("Cluster", fontsize=16)
plt.tight_layout()
plt.savefig(FIG_DIR / "mean_per_cluster_matrix.png", dpi=200, bbox_inches="tight")
plt.show()

# We can observe that some clusters tend to have higher or lower average values
# for certain features. We will later assess how informative these patterns are.

# ------------------------------
# Variance within each cluster 
# ------------------------------

cluster_var = df.groupby('Cluster')[features].var()

plt.figure(figsize=(20,15))
for i, feature in enumerate(features):
    plt.bar(cluster_var.index + i*0.15, cluster_var[feature], width=0.15, color=colors[i], label=feature)

plt.xlabel('Cluster')
plt.ylabel('Variance')
plt.title('Variance of Selected Audio Features per Cluster')
plt.xticks(cluster_var.index + 0.15, cluster_var.index)
plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(FIG_DIR / "variance_per_cluster.png", dpi=200, bbox_inches="tight")
plt.show()

# We observe that some clusters exhibit low variance for certain features,
# which suggests that they may have captured well-defined patterns in those
# dimensions. Conversely, other clusters show higher variance in some features,
# which may indicate the presence of noise or the capture of more complex patterns.
#
# Comparing cluster-level variance with the variance of the entire dataset
# can be instructive for understanding the separations produced by the model.
#
# As discussed earlier, musical tracks can be highly sophisticated and often
# combine multiple nuances, which can naturally lead to greater variability
# within certain clusters.


# --------------------------------------------------------
# Heatmaps of track counts by cluster for selected genres
# --------------------------------------------------------

# From this point onward, we will analyze the clusters formed by the model.
# The plan is to select a few genre groups and check whether similar genres
# tend to appear in the same clusters, and whether the clusters themselves
# have coherent average characteristics that reflect the nature of these sounds.


# Define groups to analyze

group1 = ['death-metal', 'hard-rock', 'heavy-metal']
group2 = ['classical', 'ambient']
group3 = ['dance', 'disco']
groups = [group1, group2, group3]

for i, group in enumerate(groups):
    fig, ax = plt.subplots(1, len(group), figsize=(20, 6))
    matrices = {}
    for genre in group:
        genre_df = df[df['track_genre'] == genre]
        cluster_counts = genre_df.groupby('Cluster')['track_id'].count().reset_index()
        cluster_counts.rename(columns={'track_id': 'Number of Tracks'}, inplace=True)
        cluster_counts = cluster_counts.sort_values(by='Number of Tracks', ascending=False)
        matrices[genre] = cluster_counts
    for j, genre in enumerate(matrices):
        sns.heatmap(
            matrices[genre].set_index('Cluster'),
            cmap='viridis',
            annot=True,
            fmt="d",
            ax=ax[j]
        )
        ax[j].set_title(f"Number of {genre} Tracks per Cluster")
        ax[j].set_ylabel("Cluster")
    fig.suptitle(f"Group {i+1} Analysis", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(FIG_DIR / f"group_{i+1}_analysis.png", dpi=200, bbox_inches="tight")
    plt.show()


# ------------------------------
# Cluster Analysis
# ------------------------------

# Group 1 includes genres associated with high energy and lower valence.
# Clusters 0 and 12 capture these characteristics well on average.
# Cluster 6, although also high in energy, has a slightly higher valence,
# suggesting it may have captured heavier tracks with a more harmonious
# or less aggressive mood.

# Notably, these clusters were very successful at capturing heavy metal
# and hard rock tracks, and they also captured a large portion of death metal tracks.

# In particular, cluster 0, with extremely high energy and the lowest valence
# among the three, captured the largest number of death metal and heavy metal
# tracks compared to other clusters. These are the heaviest genres in the group.


# Cluster 4 also captured many death metal tracks; it has high energy and
# relatively low valence. Due to its high instrumentalness, it likely includes
# tracks that stand out for guitar solos, drums, and other instrumental elements.

# Group 2 includes genres characterized by high instrumentalness, low energy,
# and low speechiness. Clusters 10 and 2 are well-suited to these characteristics
# and captured a large portion of these tracks.

# Group 3 consists of tracks with high danceability. Cluster 19, which exhibits
# very high danceability, was the most effective at capturing tracks from this group.

# ------------------------------
# Conclusion
# ------------------------------

# The model was able to capture certain patterns and created clusters that
# group tracks in a meaningful way.
#
# However, the complexity of music, the limited separability of the features,
# and the highly subjective nature of individual taste pose challenges for
# separating tracks clearly.
#
# While the model can be useful in some contexts, it may face difficulties
# due to overlapping or ambiguous clusters. Although it can successfully group
# tracks that follow certain patterns, it is not immune to including tracks
# that may feel out of place to some listeners.
#
# A more effective model could be trained using features with stronger
# discriminative power, or by exploring alternative clustering approaches,
# such as DBSCAN.


    
        
    
    