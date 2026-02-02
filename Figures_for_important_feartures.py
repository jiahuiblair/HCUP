# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import cross_val_score
from collections import Counter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define data types to address specific data loading errors
dtypes = {f'I10_DX{i}': 'str' for i in range(30, 41)}
dtypes.update({f'I10_PR{i}': 'str' for i in range(20, 26)})

## Show max rows & columns if print
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load the datasets
NIS_All_Clean_Merge = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_All_Clean_Merge.csv", dtype=dtypes)
NIS_All_Clean_Merge_2 = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_All_Clean_Merge_2.csv", dtype=dtypes)


# Define function to map categories to binary
def map_category_0_1(value):
    return 1 if value in [1, 2, 3, 4] else 0


NIS_All_Clean_Merge['HCUP_ED'] = NIS_All_Clean_Merge['HCUP_ED'].apply(map_category_0_1).astype(int)
NIS_All_Clean_Merge['I10_INJURY'] = NIS_All_Clean_Merge['I10_INJURY'].apply(map_category_0_1).astype(int)

# Filter LOS outliers based on IQR for PLOS mapping
Q1 = NIS_All_Clean_Merge_2['LOS'].quantile(0.25)
Q3 = NIS_All_Clean_Merge_2['LOS'].quantile(0.75)
IQR = Q3 - Q1
lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

# Filter out LOS outliers and define PLOS
NIS_All_clean_no_outliers = NIS_All_Clean_Merge_2[
    (NIS_All_Clean_Merge_2['LOS'] >= lower_bound) & (NIS_All_Clean_Merge_2['LOS'] <= upper_bound)]
desc = NIS_All_clean_no_outliers['LOS'].describe()
q3 = desc['75%']

NIS_All_Clean_Merge['PLOS'] = NIS_All_Clean_Merge['LOS'].apply(lambda x: 1 if x > q3 else 0)
NIS_All_Clean_Merge['DIED'] = NIS_All_Clean_Merge['DIED'].fillna(9).astype(int)

NIS_All_Clean_Merge_2['PLOS'] = NIS_All_Clean_Merge_2['LOS'].apply(lambda x: 1 if x > q3 else 0)
NIS_All_Clean_Merge_2['DIED'] = NIS_All_Clean_Merge_2['DIED'].fillna(9).astype(int)


# Combine PLOS and DIED into single categorical outcome
def map_PLOSD(value):
    return {"00": "0", "10": "1", "01": "2", "11": "3"}.get(value, "")


NIS_All_Clean_Merge['PLOSDc'] = NIS_All_Clean_Merge['PLOS'].astype(str) + NIS_All_Clean_Merge['DIED'].astype(str)
NIS_All_Clean_Merge['PLOSD'] = NIS_All_Clean_Merge['PLOSDc'].apply(map_PLOSD)

NIS_All_Clean_Merge['DIED'] = NIS_All_Clean_Merge['DIED'].replace(9, np.nan)

## for merge_2
NIS_All_Clean_Merge_2['PLOSDc'] = NIS_All_Clean_Merge_2['PLOS'].astype(str) + NIS_All_Clean_Merge_2['DIED'].astype(str)
NIS_All_Clean_Merge_2['PLOSD'] = NIS_All_Clean_Merge_2['PLOSDc'].apply(map_PLOSD)

NIS_All_Clean_Merge_2['DIED'] = NIS_All_Clean_Merge_2['DIED'].replace(9, np.nan)

# Handle PRDay data to create PRDay_model
def map_PRDay(value):
    if value <= 0:
        return 0
    elif value > q3:
        return 99
    else:
        return value


NIS_All_Clean_Merge['PRDay_model'] = NIS_All_Clean_Merge['PRDay'].apply(map_PRDay)


# Define categorical columns to one-hot encode
categorical_columns = ['YEAR', 'I10_SERVICELINE', 'PAY1', 'RACE', 'TRAN_IN', 'ZIPINC_QRTL',
                       'APRDRG_Risk_Mortality', 'APRDRG_Severity', 'HOSP_BEDSIZE', 'HOSP_LOCTEACH',
                       'HOSP_REGION', 'H_CONTRL', 'Area']

# One-hot encode and drop least frequent categories
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(NIS_All_Clean_Merge[categorical_columns])
encoded_feature_names = encoder.get_feature_names_out(categorical_columns)
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Drop the categories < 1% of the sample size for each original categorical column
encoded_df_cleaned = encoded_df.loc[:, encoded_df.sum(axis=0) > len(NIS_All_Clean_Merge)*0.01]
list_to_delete = ['YEAR_2021', 'TRAN_IN_0.0',  'ZIPINC_QRTL_4.0', 'APRDRG_Risk_Mortality_1', 'APRDRG_Severity_1', 'HOSP_BEDSIZE_3',
                  'Area_nan', 'H_CONTRL_3', 'HOSP_REGION_1']

encoded_df_cleaned_1 = encoded_df_cleaned.drop(columns = list_to_delete)
# Combine encoded columns back with original data
NIS_All_Clean_Merge_final = pd.concat([NIS_All_Clean_Merge.drop(columns=categorical_columns), encoded_df_cleaned],
                                      axis=1)

### Diagnosis and procedure codes
NIS_ALL_DX = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_ALL_DX_list.csv", dtype=dtypes)
NIS_ALL_PR = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_ALL_PR_list.csv", dtype=dtypes)
NIS_ALL_DX_R = pd.read_excel(r"C:\Users\Jiahui\PycharmProjects\NIS\ICD_regrouping.xlsx", sheet_name='Mapping_long_R')
DX_list = NIS_ALL_DX_R[NIS_ALL_DX_R['RELEVANT'] == 1]['ICD'].tolist()
PR_list = NIS_ALL_PR['ICD'].tolist()

cmr_columns = ["CMR_AIDS","CMR_ALCOHOL","CMR_DEMENTIA","CMR_DEPRESS","CMR_DIAB_CX",
               "CMR_DIAB_UNCX","CMR_DRUG_ABUSE","CMR_HTN_CX","CMR_HTN_UNCX","CMR_LUNG_CHRONIC","CMR_OBESE","CMR_PERIVASC",
               "CMR_THYROID_HYPO","CMR_THYROID_OTH"]
binary_columns = ['KID','C9100','C9102','Z006','AWEEKEND', 'ELECTIVE', 'FEMALE', 'HCUP_ED',
                       'I10_INJURY','PCLASS_ORPROC'] # Exclude the C9101 (Remission)

# Define feature sets and outcome columns
Response_column = ['PLOSD'] # ['PLOS','DIED']

Total_x_list = ['AGE'] + [col for col in encoded_df_cleaned_1.columns] + DX_list + PR_list + cmr_columns + binary_columns
Total_list = ['AGE'] + [col for col in encoded_df_cleaned_1.columns] + DX_list + PR_list + cmr_columns + binary_columns + Response_column
PRDay_list =  [1,2,3,4,5,6,7]

###################### KID ##########################################
Kid_sum = len(NIS_All_Clean_Merge_2[NIS_All_Clean_Merge_2['KID']==1])
print(Kid_sum)

list_kid = ['KID']
Kid_data = NIS_All_Clean_Merge_2[NIS_All_Clean_Merge_2['KID']==0]
print(Kid_data.columns)
kid_summary = {}
for column in list_kid:
    kid_summary[column] = Kid_data.groupby('PLOSD')[column].apply(pd.value_counts) #only get counts
    counts = Kid_data.groupby('PLOSD')[column].value_counts().unstack(fill_value=0)
    relative_frequency = counts.div(len(Kid_data[column]))  # Normalize counts to get relative frequency
    relative_frequency = relative_frequency.round(4)  # Round to 2 decimal places
    kid_summary[column] = pd.concat([counts, relative_frequency], axis=1, keys=['Count', 'Relative Frequency'])

# Print the summary statistics for each value variable
for column, summary in kid_summary.items():
    print(f"Summary for {column}:")
    print(summary)
    print()

################### Age vs PLOS #########################################


print(NIS_All_Clean_Merge["AGE"].describe())  # Check for min/max values
print(NIS_All_Clean_Merge[NIS_All_Clean_Merge["AGE"] < 0])

# Set up figure size and style
sns.set_style("whitegrid")

# Define category labels
category_labels = {0: "Normal", 1: "PLOS only", 2: "Mortality only", 3: "Mortality & PLOS"}
category_order = [0, 1, 2, 3]

# Ensure PLOSD is numeric and categorical
NIS_All_Clean_Merge["PLOSD"] = pd.to_numeric(NIS_All_Clean_Merge["PLOSD"], errors='coerce')
NIS_All_Clean_Merge["PLOSD"] = pd.Categorical(NIS_All_Clean_Merge["PLOSD"], categories=category_order, ordered=True)

# Determine grid size (two columns)
n_plots = len(PRDay_list)
n_rows = int(np.ceil(n_plots / 2))  # Ensure enough rows

# Create figure with subplots in a grid (rows x 2 columns)
fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows), sharex=True)

# Flatten axes for easy indexing (works even if n_rows == 1)
axes = axes.flatten()

NIS_All_Clean_Merge_final['PRDay_model'] = NIS_All_Clean_Merge_final['PRDay_model'].mask(NIS_All_Clean_Merge_final['PRDay_model'] == 0, 1)
print(NIS_All_Clean_Merge_final['PRDay_model'].unique())

# Loop through PRDay_list to create violin plots in subplots
for i, day in enumerate(PRDay_list):
    # Filter data for the given PRDay
    model_data = NIS_All_Clean_Merge_final[NIS_All_Clean_Merge_final['LOS'] >= day]
    print("day",day)
    print("Length model_data", len(model_data))
    # Drop missing values in PLOSD

    model_data["PLOSD"] = model_data["PLOSD"].replace("", np.nan)  # Replace empty strings with NaN
    model_data = model_data.dropna(subset=["PLOSD"])
    model_data["PLOSD"] = model_data["PLOSD"].astype(int)
    
    subset = model_data
    print("Length subset", len(subset))
    # Skip plotting if no valid data remains
    if subset.empty:
        print(f"Skipping PRDay {day}: No valid data after filtering.")
        continue

    # Map PLOSD to labels for better readability
    print(subset["PLOSD"].unique())
    subset["PLOSD_Label"] = subset["PLOSD"].map(category_labels)


    # Create the violin plot
    sns.violinplot(
        x="PLOSD_Label", y="AGE", data=subset,
        palette="pastel", inner="quartile", order=category_labels.values(),
        ax=axes[i]#, bw_adjust=0.5
    )

    # Add a horizontal line at y=18
    axes[i].axhline(y=18, color='red', linewidth=0.8, linestyle='dotted', label="18")

    # Fix the y-ticks
    axes[i].set_yticks([0, 20, 40, 60, 80, 100])

    # Formatting
    axes[i].set_title(f"Day {day}", fontsize=5, pad=5, fontweight="bold")
    axes[i].set_xlabel("")  # Remove individual x-axis labels
    axes[i].set_ylabel("Age")

# Hide any unused subplots (if PRDay_list has an odd number of items)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the full figure
plt.show()

###############################Age vs TC ##################################
# # Set up the figure with multiple subplots (one column of scatter plots)
# fig, axes = plt.subplots(len(PRDay_list), 1, figsize=(10, 4 * len(PRDay_list)), constrained_layout=True)
#
# # Loop through PRDay_list to create scatter plots for Age vs. Total Charges
# for i, day in enumerate(PRDay_list):
#     model_data = NIS_All_Clean_Merge[NIS_All_Clean_Merge['PRDay_model'] == day]
#     subset = model_data.dropna(subset=['AGE', 'TOTCHG'])
#
#     # Scatter Plot (Dot plot)
#     sns.scatterplot(x="AGE", y="TOTCHG", data=subset, ax=axes[i], alpha=0.5, color="blue", edgecolor=None)
#     sns.regplot(x="AGE", y="TOTCHG", data=subset, ax=axes[i], scatter=False, color="red", line_kws={"linewidth":1})  # Trend line
#     axes[i].set_title(f"Day {day}", fontsize=10, fontweight="bold", pad=5)
#     axes[i].set_xlabel(None)
#     axes[i].set_ylabel(None)
#
# # Show the figure
# plt.show()

############################### PLOS vs C9100 #####################################
#
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
#
# # Define the labels for the PLOS categories
# category_labels = {0: "Normal", 1: "PLOS only", 2: "Mortality only", 3: "Mortality & PLOS"}
#
# # Set up the figure
# fig, ax = plt.subplots(figsize=(14, 6))
#
# # Set the colors
# colors = sns.color_palette("Set2", n_colors=4)  # Color palette for the 4 PLOS categories
# c9100_colors = sns.color_palette("Set1", n_colors=2)  # Color palette for C9100 (2 categories)
#
# # Prepare the data for plotting
# count_data = []
#
# # Loop through each day to calculate the total count for PLOS and C9100
# for day in PRDay_list:
#     model_data = NIS_All_Clean_Merge[NIS_All_Clean_Merge['PRDay_model'] == day]
#
#     # Calculate the counts for PLOS and C9100
#     plos_count = model_data['PLOS'].value_counts().to_dict()
#     var_count = model_data['C9100'].value_counts().to_dict()
#
#     count_data.append({
#         'Day': day,
#         'PLOS_0': plos_count.get(0, 0),  # Assuming 0 represents "Normal"
#         'PLOS_1': plos_count.get(1, 0),  # Assuming 1 represents "PLOS only"
#         'PLOS_2': plos_count.get(2, 0),  # Assuming 2 represents "Mortality only"
#         'PLOS_3': plos_count.get(3, 0),  # Assuming 3 represents "Mortality & PLOS"
#         'var_0': var_count.get(0, 0),  # Assuming 0 represents "No" for C9100
#         'var_1': var_count.get(1, 0)   # Assuming 1 represents "Yes" for C9100
#     })
#
# # Convert to DataFrame
# count_df = pd.DataFrame(count_data)
#
# # Reshape the data for the bar plot
# count_df_melted = count_df.melt(id_vars="Day", value_vars=["PLOS_0", "PLOS_1", "PLOS_2", "PLOS_3"],
#                                  var_name="PLOS Category", value_name="Count")
#
# # Map PLOS category values to labels
# count_df_melted['PLOS Category'] = count_df_melted['PLOS Category'].map({
#     'PLOS_0': category_labels[0],
#     'PLOS_1': category_labels[1],
#     'PLOS_2': category_labels[2],
#     'PLOS_3': category_labels[3]
# })
#
# # Plot the bar plot with log scale
# sns.barplot(x="Day", y="Count", hue="PLOS Category", data=count_df_melted, palette=colors, ax=ax, dodge=True)
# ax.set_yscale('log')  # Apply log scale to make small values more visible
#
# # Overlay the trend lines for C9100 counts (two lines: C9100_0 and C9100_1)
# ax.plot(count_df['Day'], count_df['var_0'], marker="o", color=c9100_colors[0], label='C9100 (No)', linestyle='-', linewidth=2)
# ax.plot(count_df['Day'], count_df['var_1'], marker="^", color=c9100_colors[1], label='C9100 (Yes)', linestyle='-', linewidth=2)
#
# # Add title and labels
# ax.set_title('Total Count of PLOS Categories and C9100 Across 7 Days', fontsize=14, fontweight='bold')
# ax.set_xlabel('PRDay')
# ax.set_ylabel('Total Count (Log Scale)')
#
# # Add the legend
# ax.legend(title="Categories", loc="upper left")
#
# # Show the plot
# plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
# plt.tight_layout()  # Adjust layout to prevent overlap
# plt.show()
#









########################################################
#ALL_list = ['C9100','C9101','C9102']
# summary_per_variable_PLOSD = {}
# for column in ALL_list:
#     summary_per_variable_PLOSD[column] = NIS_All_Clean_Merge.groupby('PLOSD')[column].apply(pd.value_counts) #only get counts
#     counts = NIS_All_Clean_Merge.groupby('PLOSD')[column].value_counts().unstack(fill_value=0)
#     relative_frequency = counts.div(len(NIS_All_Clean_Merge[column]))  # Normalize counts to get relative frequency
#     relative_frequency = relative_frequency.round(4)*100  # Round to 2 decimal places
#     summary_per_variable_PLOSD[column] = pd.concat([counts, relative_frequency], axis=1, keys=['Count', 'Relative Frequency'])
#
# ## Print the summary statistics for each value variable
# for column, summary in summary_per_variable_PLOSD.items():
#     print(f"Summary for {column}:")
#     print(summary)
#     print()




