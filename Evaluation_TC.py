# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


## Show max rows & columns if print
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

## Read the data
Prediction_summary = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\TCprediction_comb\Prediction_summary_PLOSD_removeLos_7.0.csv")

Prediction_summary_2 = Prediction_summary.copy()
list_feature = ["Selected_feature_gb", "Selected_feature_rf", "Feature_importance_gb", "Feature_importance_rf"]
Prediction_summary_2.drop(columns=list_feature, inplace=True)
# print(Prediction_summary_2.columns)

# Assuming your DataFrame is named df
Prediction_summary_2['Day'] = Prediction_summary_2['Day'].replace(99, 8)

# calculate sqrt of the MSE
Prediction_summary_2['RMSE_rf_cv'] = np.sqrt(Prediction_summary_2['MSE_rf_cv'])
Prediction_summary_2['RMSE_gb_cv'] = np.sqrt(Prediction_summary_2['MSE_gb_cv'])

Prediction_summary_2['RMSE_rf'] = np.sqrt(Prediction_summary_2['MSE_rf'])
Prediction_summary_2['RMSE_gb'] = np.sqrt(Prediction_summary_2['MSE_gb'])

print(Prediction_summary_2.head())

# Melt the DataFrame
Prediction_summary_long = pd.melt(
    Prediction_summary_2,
    id_vars=["Day"],  # Columns to keep
    var_name="Metric",  # Name for the new column containing melted column names
    value_name="Value"  # Name for the new column containing values
)

# Add a 'Model' column based on the presence of '_rf' or '_gb' in the column names
Prediction_summary_long["Model"] = Prediction_summary_long["Metric"].apply(lambda x: "rf" if "_rf" in x else "gb" if "_gb" in x else None)

# Remove '_rf' and '_gb' from the Metric column
Prediction_summary_long["Metric"] = Prediction_summary_long["Metric"].str.replace("_rf", "").str.replace("_gb", "")

# Filter rows where 'Model' is not None (ignores rows with irrelevant columns)
Prediction_summary_long = Prediction_summary_long[Prediction_summary_long["Model"].notnull()]

# Sort the DataFrame for better readability
Prediction_summary_long = Prediction_summary_long.sort_values(by=["Day", "Model", "Metric"]).reset_index(drop=True)

# Filter the cross validation averaged metrics
filt_list = ['MAE_cv', 'RMSE_cv', 'R2_cv', 'MAE', 'RMSE', 'R2']
Prediction_summary_long_plot = Prediction_summary_long[Prediction_summary_long['Metric'].isin(filt_list)]

# Display the transformed DataFrame
print(Prediction_summary_long_plot)


# common_ylim = (0.4, 1.0)
for item in Prediction_summary_long_plot['Metric'].unique():
    # Initialize a plot
    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=Prediction_summary_long_plot[Prediction_summary_long_plot['Metric'] == item],
        x="Day",
        y="Value",
        hue="Model",
        style="Model")

    # Adjust x-axis ticks to only include specific values
    plt.xticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8], labels=[1, 2, 3, 4, 5, 6, 7, ">7"])

    # Annotate y-values on each point
    for model in Prediction_summary_long_plot['Model'].unique():
        data = Prediction_summary_long_plot[
            (Prediction_summary_long_plot['Metric'] == item) &
            (Prediction_summary_long_plot['Model'] == model)
            ]
        for x, y in zip(data['Day'], data['Value']):
            plt.text(
                x, y, f'{y:.2f}',  # Format the value with 2 decimal places
                fontsize=10,
                ha='center',
                va='bottom',
                color='black'
            )

    # Set plot title and labels
    plt.xlabel("Day", fontsize=15)
    plt.ylabel(f"{item}", fontsize=15)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    # plt.xticks(range(1, 100, 10))  # Adjust x-ticks for better readability
    # y_ticks = np.arange(0.4, 1.1, 0.1)
    # plt.ylim(common_ylim)
    plt.yticks(fontsize = 18)
    plt.xticks(fontsize=18)
    plt.grid(visible=True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.tight_layout()
    plt.show()