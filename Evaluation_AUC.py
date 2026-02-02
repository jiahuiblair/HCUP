# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import make_scorer, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score


# Show max rows & columns if print
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# List of file paths
file_paths = [
    r"C:\Users\Jiahui\PycharmProjects\NIS\Outcome\PLOSDIED\Prediction_data_output_removeLOS_1.0.csv",
    r"C:\Users\Jiahui\PycharmProjects\NIS\Outcome\PLOSDIED\Prediction_data_output_removeLOS_2.0.csv",
    r"C:\Users\Jiahui\PycharmProjects\NIS\Outcome\PLOSDIED\Prediction_data_output_removeLOS_3.0.csv",
    r"C:\Users\Jiahui\PycharmProjects\NIS\Outcome\PLOSDIED\Prediction_data_output_removeLOS_4.0.csv",
    r"C:\Users\Jiahui\PycharmProjects\NIS\Outcome\PLOSDIED\Prediction_data_output_removeLOS_5.0.csv",
    r"C:\Users\Jiahui\PycharmProjects\NIS\Outcome\PLOSDIED\Prediction_data_output_removeLOS_6.0.csv",
    r"C:\Users\Jiahui\PycharmProjects\NIS\Outcome\PLOSDIED\Prediction_data_output_removeLOS_7.0.csv"
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

# Initialize the plot
plt.figure(figsize=(10, 7))

# Loop through each day to calculate and plot the ROC curve
for day in model_numbers:
    # Filter data for the specific day
    day_data = combined_df[combined_df['Day'] == day]

    # True labels and predicted probabilities
    y_true = day_data['DIED']
    y_prob = day_data['GB_Prob_DIED']

    # Optional: Define sample weights if needed
    # sample_weight = day_data.get('Sample_Weight', None)  # Adjust this based on your dataset

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)  # Add sample_weight=sample_weight if weights are used

    # Calculate AUC using weighted average
    auc_score = roc_auc_score(y_true, y_prob, average='weighted')  # Automatically weighted
    roc_auc = auc(fpr, tpr)

    # # Set the legend label, replacing 99 with >7
    day_label = ">7" if day == 99 else f"{day}"
    plt.plot(fpr, tpr, label=f"Day {day_label} (AUC = {auc_score:.4f})", linestyle="--" if day > 7 else "-")

# Plot diagonal line (random classifier)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

# Customize the plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Day')
plt.legend(loc='lower right')
plt.grid()
plt.show()

