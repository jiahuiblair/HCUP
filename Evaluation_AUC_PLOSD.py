# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc


## Show max rows & columns if print
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# List of file paths
file_paths = [
    r"C:\Users\Jiahui\PycharmProjects\NIS\Outcome\PLOSD\Prediction_data_output_removaLOS_1.0.csv",
    r"C:\Users\Jiahui\PycharmProjects\NIS\Outcome\PLOSD\Prediction_data_output_removaLOS_2.0.csv",
    r"C:\Users\Jiahui\PycharmProjects\NIS\Outcome\PLOSD\Prediction_data_output_removaLOS_3.0.csv",
    r"C:\Users\Jiahui\PycharmProjects\NIS\Outcome\PLOSD\Prediction_data_output_removaLOS_4.0.csv",
    r"C:\Users\Jiahui\PycharmProjects\NIS\Outcome\PLOSD\Prediction_data_output_removaLOS_5.0.csv",
    r"C:\Users\Jiahui\PycharmProjects\NIS\Outcome\PLOSD\Prediction_data_output_removaLOS_6.0.csv",
    r"C:\Users\Jiahui\PycharmProjects\NIS\Outcome\PLOSD\Prediction_data_output_removaLOS_7.0.csv"
]

# Model numbers corresponding to the file order
model_numbers = [1, 2, 3, 4, 5, 6, 7]

# Read and combine DataFrames
df_list = []
for file_path, model_num in zip(file_paths, model_numbers):
    df = pd.read_csv(file_path)
    df['Day'] = model_num
    df_list.append(df)

# Concatenate DataFrames
combined_df = pd.concat(df_list, ignore_index=True)

n_classes = 4  # Number of classes

# Plotting
plt.figure(figsize=(12, 8))
colors = ["blue", "green", "red", "orange"]

# Iterate through each class
for i in range(n_classes):
    plt.figure(figsize=(8, 6))  # Create a new figure for each class

    # Iterate through each day
    for day in model_numbers:
        day_data = combined_df[combined_df["Day"] == day]
        true_classes = pd.get_dummies(day_data["PLOSD"], columns=range(n_classes))

        # Compute ROC and AUC for the current class
        fpr, tpr, _ = roc_curve(true_classes.iloc[:, i], day_data[f"RF_Prob_Class_{i}_PLOSD"])#### change here
        roc_auc = auc(fpr, tpr)

        # Set the legend label, replacing 99 with >7
        day_label = ">7" if day == 99 else f"{day}"
        plt.plot(fpr, tpr, label=f"Day {day_label} (AUC = {roc_auc:.4f})", linestyle="--" if day > 7 else "-")

    # Plot settings for the current class
    plt.plot([0, 1], [0, 1], 'k--', label="Random guess")  # Random guess line
    plt.title(f"ROC Curves for Class {i}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.show()
