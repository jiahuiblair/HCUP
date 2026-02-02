import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re

# Show max rows & columns if print
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Read the data
Prediction_summary = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\Outcome\PLOSD\Prediction_summary_removaLOS_7.0.csv")

list_feature = ["Outcome", "Day", "Selected_feature_rf", "Feature_importance_rf"]
Prediction_summary_2 = Prediction_summary[list_feature].copy()

# Replace Day 99 with 8
Prediction_summary_2['Day'] = Prediction_summary_2['Day'].replace(99, 8)

# Function to convert string representation of lists to actual lists
def convert_float_list(value):
    try:
        if isinstance(value, str):
            cleaned_value = re.sub(r'np\.float64\((.*?)\)', r'\1', value)
            return ast.literal_eval(cleaned_value)
        return value  # If already a list, return as is
    except (ValueError, SyntaxError):
        return np.nan  # Return NaN for any invalid values

# Apply conversion
Prediction_summary_2['Feature_importance_rf'] = Prediction_summary_2['Feature_importance_rf'].astype(str).apply(convert_float_list)

# Drop NaN values if needed
Prediction_summary_2.dropna(subset=['Feature_importance_rf'], inplace=True)

# Convert stringified lists to actual lists
Prediction_summary_2['Selected_feature_rf'] = Prediction_summary_2['Selected_feature_rf'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
Prediction_summary_2['Feature_importance_rf'] = Prediction_summary_2['Feature_importance_rf'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Explode both 'Feature' and 'Importance' columns simultaneously
df_exploded = Prediction_summary_2.explode(['Selected_feature_rf', 'Feature_importance_rf'], ignore_index=True)

# Sort values
df_sorted = df_exploded.sort_values(by=['Outcome', 'Day', 'Feature_importance_rf'], ascending=[True, True, False])

# Save sorted data
df_sorted.to_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\Outcome\Sorted_Feature_outcome.csv", index=False)

# Ensure 'Feature_importance_rf' is numeric
df_sorted['Feature_importance_rf'] = pd.to_numeric(df_sorted['Feature_importance_rf'], errors='coerce')

# Get the top 20 features for each day
top_features = df_sorted.groupby("Day").apply(lambda x: x.nlargest(10, "Feature_importance_rf")).reset_index(drop=True)

# Prepare the data for the heatmap
heatmap_data = top_features.pivot_table(index="Selected_feature_rf", columns="Day", values="Feature_importance_rf", aggfunc="mean").fillna(0)

# Rename the column for Day 8 to ">7"
heatmap_data.rename(columns={8: ">7"}, inplace=True)

# Plot the heatmap
plt.figure(figsize=(18, 14))
sns.heatmap(
    heatmap_data,
    cmap="Blues",
    annot=False,
    linewidths=0.5,
    cbar_kws={"label": "Feature Importance"},
    vmin=0,
    vmax=heatmap_data.max().max()
)

# Adjust x-axis labels
xticks = [">7" if x == ">7" else int(x) for x in heatmap_data.columns]
plt.xticks(ticks=range(len(xticks)), labels=xticks, fontsize=10)

# Adjust y-axis labels
plt.yticks(fontsize=10, rotation=0, ha="right")

# Add labels and title
plt.title("Feature Importance Heatmap by Day")
plt.xlabel("Day")
plt.ylabel("Feature")
plt.tight_layout(pad=2.0)

# Show the plot
plt.show()

###################################PR###################################
NIS_ALL_PR = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_ALL_PR_list.csv")
PR_list = NIS_ALL_PR['ICD'].tolist()
print(PR_list)
print(df_sorted.head())
# Filter the dataset to include only features in PR_list
df_filtered = df_sorted[df_sorted["Selected_feature_rf"].isin(PR_list)]
print(df_filtered.head())

# Ensure 'Feature_importance_rf' is numeric
df_filtered['Feature_importance_rf'] = pd.to_numeric(df_sorted['Feature_importance_rf'], errors='coerce')

# Get the top 20 features for each day
top_features_f = df_filtered.groupby("Day").apply(lambda x: x.nlargest(10, "Feature_importance_rf")).reset_index(drop=True)

# Prepare the data for the heatmap
heatmap_data = top_features_f.pivot_table(index="Selected_feature_rf", columns="Day", values="Feature_importance_rf", aggfunc="mean").fillna(0)

# Rename the column for Day 8 to ">7"
heatmap_data.rename(columns={8: ">7"}, inplace=True)

# Plot the heatmap
plt.figure(figsize=(18, 14))
sns.heatmap(
    heatmap_data,
    cmap="Blues",
    annot=False,
    linewidths=0.5,
    cbar_kws={"label": "Feature Importance"},
    vmin=0,
    vmax=heatmap_data.max().max()
)

# Adjust x-axis labels
xticks = [">7" if x == ">7" else int(x) for x in heatmap_data.columns]
plt.xticks(ticks=range(len(xticks)), labels=xticks, fontsize=10)

# Adjust y-axis labels
plt.yticks(fontsize=10, rotation=0, ha="right")

# Add labels and title
plt.title("Feature Importance Heatmap by Day")
plt.xlabel("Day")
plt.ylabel("Feature")
plt.tight_layout(pad=2.0)

# Show the plot
plt.show()

######################## Bar chart of number of important features identfied #################
# Count features per day
feature_counts = df_sorted["Day"].value_counts().sort_index()



# Plot bar chart
plt.figure(figsize=(8, 5))
feature_counts.plot(kind="bar", color="coral", edgecolor="black")

# Set the y-axis limits and ticks:
plt.ylim(0, 150)  # Set the y-axis limits from 0 to 150
plt.yticks(range(0, 151, 10)) # Set the y-axis ticks from 0 to 150, in steps of 10

plt.xlabel("Day")
plt.ylabel("Number of Features")
plt.title("Number of Features Selected per Day")
plt.xticks(rotation=0)
plt.show()