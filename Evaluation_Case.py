import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# List of file paths
file_paths = [
    r"C:\Users\Jiahui\PycharmProjects\NIS\PLOSM_case\Prediction_data_output_Z006_1.0.csv",
    r"C:\Users\Jiahui\PycharmProjects\NIS\PLOSM_case\Prediction_data_output_Z006_2.0.csv",
    r"C:\Users\Jiahui\PycharmProjects\NIS\PLOSM_case\Prediction_data_output_Z006_3.0.csv",
    r"C:\Users\Jiahui\PycharmProjects\NIS\PLOSM_case\Prediction_data_output_Z006_4.0.csv",
    r"C:\Users\Jiahui\PycharmProjects\NIS\PLOSM_case\Prediction_data_output_Z006_5.0.csv",
    r"C:\Users\Jiahui\PycharmProjects\NIS\PLOSM_case\Prediction_data_output_Z006_6.0.csv",
    r"C:\Users\Jiahui\PycharmProjects\NIS\PLOSM_case\Prediction_data_output_Z006_7.0.csv",
]

# Read the first file and keep PLOSD
final_df = pd.read_csv(
    file_paths[0],
    usecols=["KEY_NIS", "PLOSD", "RF_Prediction_PLOSD"],
    dtype={"KEY_NIS": "int32", "PLOSD": "int8", "RF_Prediction_PLOSD": "float32"},
)

# Drop duplicates in KEY_NIS, keeping the last occurrence
final_df = final_df.drop_duplicates(subset=["KEY_NIS"], keep="last")

# Rename prediction column
final_df.rename(columns={"RF_Prediction_PLOSD": "Day1"}, inplace=True)

# Iteratively merge each file, dropping PLOSD
for i, file in enumerate(file_paths[1:], start=2):
    df = pd.read_csv(
        file,
        usecols=["KEY_NIS", "RF_Prediction_PLOSD"],  # Drop PLOSD to avoid duplication
        dtype={"KEY_NIS": "int32", "RF_Prediction_PLOSD": "float32"},
    )

    # Drop duplicates in KEY_NIS
    df = df.drop_duplicates(subset=["KEY_NIS"], keep="last")

    # Rename prediction column
    df.rename(columns={"RF_Prediction_PLOSD": f"Day{i}"}, inplace=True)

    # Merge
    final_df = final_df.merge(df, on="KEY_NIS", how="left")

    print(f"Merged file {i}: Shape = {final_df.shape}")

# Save to a new CSV file
output_path = r"C:\Users\Jiahui\PycharmProjects\NIS\PLOSM_case\merged_predictions.csv"
final_df.to_csv(output_path, index=False)
####################################################################################################################

# Convert numeric values to string, replacing NaNs with '-'

df_filled = final_df.fillna('9').astype('int64').astype('str').replace('9', '-')
print(df_filled.head())


# Create a readable pattern label (PLOSD_Day1_Day2_..._Day7)
df_filled["Pattern"] = df_filled[['PLOSD', 'Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7']].agg("".join, axis=1)

# Aggregate counts of unique patterns
pattern_counts = df_filled["Pattern"].value_counts()

print(pattern_counts)

# Select top patterns for readability
top_patterns = pattern_counts.head(50)  # Adjust as needed

# Plot bar chart with correct counts
plt.figure(figsize=(12, 8))
sns.barplot(y=top_patterns.index, x=top_patterns.values, palette="viridis") #palette="viridis")

# Formatting the plot
plt.xlabel("Number of Patients")
plt.ylabel("Prediction Trend (True Label & Day1-Day7 Predictions)")
plt.title("Top Prediction Trends Across Patients")

plt.show()