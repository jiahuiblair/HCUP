import numpy as np;
import pandas as pd;
import matplotlib as mpl;
import sklearn as skl;
from scipy.stats import f_oneway;
import matplotlib.pyplot as plt;
import seaborn as sns;
from collections import Counter;
from sklearn.preprocessing import OneHotEncoder;
from mlxtend.frequent_patterns import apriori;
from mlxtend.frequent_patterns import association_rules;
from mlxtend.preprocessing import TransactionEncoder;
from sklearn.cluster import DBSCAN;
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

## Specify the column dtype based on the error message
dtypes = {'I10_DX36': 'str', 'I10_DX37': 'str', 'I10_DX38': 'str', 'I10_DX39': 'str', 'I10_DX40': 'str',
          'I10_DX30': 'str', 'I10_DX31': 'str', 'I10_DX32': 'str', 'I10_DX33': 'str', 'I10_DX34': 'str',
          'I10_DX35': 'str', 'I10_PR20': 'str', 'I10_PR21': 'str', 'I10_PR22': 'str', 'I10_PR23': 'str',
          'I10_PR24': 'str', 'I10_PR25': 'str'}
## Show max rows & columns if print
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

## Read the data
NIS_All_Clean_Merge_2 = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_All_Clean_Merge_2.csv", dtype=dtypes)

##################### Define the list of column names #############################################
Response_column = ['PLOS','TOTCHG','DIED']
numeric_columns = ['AGE']
cmr_columns = ["CMR_AIDS","CMR_ALCOHOL","CMR_DEMENTIA","CMR_DEPRESS","CMR_DIAB_CX",
               "CMR_DIAB_UNCX","CMR_DRUG_ABUSE","CMR_HTN_CX","CMR_HTN_UNCX","CMR_LUNG_CHRONIC","CMR_OBESE","CMR_PERIVASC",
               "CMR_THYROID_HYPO","CMR_THYROID_OTH"]
binary_columns = ['AWEEKEND', 'ELECTIVE', 'FEMALE', 'HCUP_ED','I10_INJURY','I10_MULTINJURY','PCLASS_ORPROC', 'KID',]
categorical_columns = ['YEAR','I10_SERVICELINE','PAY1','RACE', 'TRAN_IN', 'ZIPINC_QRTL','APRDRG_Risk_Mortality',
                       'APRDRG_Severity','HOSP_BEDSIZE','HOSP_LOCTEACH', 'HOSP_REGION','H_CONTRL', 'Area']

## Find the DX and PR list
NIS_ALL_DX = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_ALL_DX_list.csv", dtype=dtypes)
NIS_ALL_PR = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_ALL_PR_list.csv", dtype=dtypes)
DX_list = NIS_ALL_DX['ICD'].tolist()
PR_list = NIS_ALL_PR['ICD'].tolist()

#### One-hot encoder
## Perform one-hot encoding on the categorical columns
encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
encoded_features = encoder.fit_transform(NIS_All_Clean_Merge_2[categorical_columns])

## Get the feature names
encoded_feature_names = encoder.get_feature_names_out(categorical_columns)

## Convert the encoded features back to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

## Drop the least frequent categories
# Sum up the `1`s in each column
column_sums = encoded_df.sum(axis=0)
# Create a dictionary to store the minimum frequency column for each original feature
least_frequent_cols = {}
# Loop through each original categorical feature
for feature in categorical_columns:
    # Filter out the columns that belong to the current categorical feature
    feature_cols = [col for col in encoded_df.columns if col.startswith(feature + '_')]
    # Get the column with the minimum frequency of `1`s
    min_freq_col = column_sums[feature_cols].idxmin()
    # Store the column to drop
    least_frequent_cols[feature] = min_freq_col

# Drop the least frequent columns from the encoded dataframe
encoded_df_cleaned = encoded_df.drop(columns=list(least_frequent_cols.values()))

# Concatenate the encoded features with the original DataFrame (excluding the original categorical columns)
NIS_All_Clean_final = pd.concat([NIS_All_Clean_Merge_2.drop(columns=categorical_columns), encoded_df_cleaned], axis=1)
onehot_columns_0 = encoded_df_cleaned.columns.tolist()


# # Transpose the data
clustering_set = DX_list + cmr_columns #+ binary_columns + encoded_df_cleaned.columns.tolist()

# ###### Isolation Forest ######
# from sklearn.ensemble import IsolationForest
#
# # Transpose the data so that each ICD code is treated as a "sample"
# X_transposed = NIS_All_Clean_final[clustering_set].T  # ICD codes become rows, patients become columns
#
# # Initialize Isolation Forest for feature (ICD code) outlier detection
# iso_forest = IsolationForest(contamination=0.1, random_state=42) # contamination parameter represents the proportion of outliers in the dataset.
# outlier_labels = iso_forest.fit_predict(X_transposed)
#
# # Outlier labels: -1 for outliers, 1 for inliers
# X_transposed['Outlier'] = outlier_labels
#
# # Calculate anomaly scores for each ICD code
# anomaly_scores = iso_forest.decision_function(X_transposed.iloc[:, :-1])  # Exclude the Outlier column itself
# X_transposed['Anomaly_Score'] = anomaly_scores
# #
# # # Visualize the distribution of anomaly scores for ICD features
# # plt.figure(figsize=(10, 6))
# # sns.histplot(anomaly_scores, bins=30, kde=True)
# # plt.xlabel('Anomaly Score')
# # plt.ylabel('Frequency')
# # plt.title('Distribution of Anomaly Scores for ICD Features')
# # plt.show()
#
# # Identify outlier ICD codes
# outlier_ICD_codes = X_transposed[X_transposed['Outlier'] == -1]
#
# # Print only the names of the outlier ICD codes
# outlier_feature_names = outlier_ICD_codes.index.tolist()
# print("Outlier ICD feature names:", outlier_feature_names)
# kept_cluster_features = [feature for feature in clustering_set if feature not in outlier_feature_names]
# print("Keep ICD feature names:", kept_cluster_features)
#


############################################################################################
# # Calculate the distance matrix
metric_type = "hamming"
method_type = ("complete")

data = NIS_All_Clean_final[clustering_set].copy()
dist_matrix = pdist(data.T, metric= metric_type)  # Transpose to cluster features
dist_matrix = squareform(dist_matrix)


#### Outlier detector ####
# Convert distances to a square matrix form to interpret similarities
similarity_matrix = 1 - dist_matrix
sim_df = pd.DataFrame(similarity_matrix, index= data.columns, columns=data.columns)
mean_similarities = sim_df.mean(axis=1)
print(mean_similarities)

########################## Hierarchical clustering ################################

# Perform hierarchical clustering using complete linkage
linked = linkage(dist_matrix, method= method_type)

# Visualize the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', labels=data.columns)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('ICD Code')
plt.ylabel('Distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Silhouette analysis
silhouette_scores = [] #measures how similar a data point is to its own cluster (cohesion) compared to other clusters (separation)
# High Score (close to 1): Indicates that the data point is well-matched to its own cluster and poorly matched to neighboring clusters. This suggests a good clustering result
# Low Score (close to 0): Indicates that the data point is close to the decision boundary between two clusters, suggesting overlapping clusters or a less distinct clustering

# Define maximum number of clusters as len(dist_matrix) - 1 to stay within valid range
max_clusters = min(len(dist_matrix) - 1, 37)
silhouette_scores = []

for n_clusters in range(2, max_clusters):
    # Perform clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=method_type, compute_full_tree=True)
    cluster_labels = clustering.fit_predict(dist_matrix)

    # Check if the number of unique labels is within valid range for silhouette score
    if 2 <= len(set(cluster_labels)) <= len(dist_matrix) - 1:
        # Calculate silhouette score with precomputed distance matrix
        silhouette_avg = silhouette_score(dist_matrix, cluster_labels, metric='precomputed')
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {n_clusters}, the silhouette score is: {silhouette_avg}")
    else:
        print(f"Skipping n_clusters = {n_clusters} due to insufficient number of labels.")

# Plotting silhouette scores
plt.figure()
plt.plot(range(2, max_clusters), silhouette_scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis for Optimal Cluster Number")
plt.xticks(range(2, max_clusters))
plt.grid()
plt.show()


# Cut the dendrogram
num_clusters = 8
labels = fcluster(linked, num_clusters, criterion='maxclust')

# Create a DataFrame with diagnosis and cluster labels
clustered_data = pd.DataFrame({'ICD': data.T.index, 'Cluster': labels})
clustered_data.to_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\Cluster_DX_PR.csv", index=False)

# Explore the clusters
grouping_df = pd.read_excel(r"C:\Users\Jiahui\PycharmProjects\NIS\ICD_regrouping.xlsx", sheet_name='Mapping_long')
Cluster_result = pd.merge(clustered_data, grouping_df, on='ICD', how='left')

# Calculate the frequency of each ICD code in df1
df1_freq = NIS_All_Clean_Merge_2[clustering_set].sum(axis=0)
df1_freq = df1_freq.reset_index()
df1_freq.columns = ['ICD', 'Frequency']

print(df1_freq)
# Merge the frequency information into df2
print(clustered_data['ICD'].dtype)
print(df1_freq['ICD'].dtype)

Cluster_result = Cluster_result.merge(df1_freq, on='ICD', how= 'left')

Cluster_result.to_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\Cluster_DX_PR_descrip.csv", index=False)

