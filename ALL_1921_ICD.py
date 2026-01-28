import numpy as np;
import pandas as pd;
import matplotlib as mpl;
import sklearn as skl;
from scipy.stats import f_oneway;
import matplotlib.pyplot as plt;
import seaborn as sns;
from collections import Counter;
from mlxtend.frequent_patterns import apriori;
from mlxtend.frequent_patterns import association_rules;
from mlxtend.preprocessing import TransactionEncoder;
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

## Specify the column dtype based on the error message
dtypes = {'I10_DX36': 'str', 'I10_DX37': 'str', 'I10_DX38': 'str', 'I10_DX39': 'str', 'I10_DX40': 'str',
          'I10_DX30': 'str', 'I10_DX31': 'str', 'I10_DX32': 'str', 'I10_DX33': 'str', 'I10_DX34': 'str',
          'I10_DX35': 'str', 'I10_PR20': 'str', 'I10_PR21': 'str', 'I10_PR22': 'str', 'I10_PR23': 'str',
          'I10_PR24': 'str', 'I10_PR25': 'str'}

## Read the data
NIS_ALL = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_ALL_Clean.csv", dtype=dtypes)
# Load ICD grouping structure from Excel
grouping_df = pd.read_excel(r"C:\Users\Jiahui\PycharmProjects\NIS\ICD_regrouping.xlsx", sheet_name='Mapping_long')

######################## Check the missing PRday value and delete the missing rows ###############################
# Define the list of procedure and corresponding day columns
procedure_cols = [f'I10_PR{i}' for i in range(1, 26)]
day_cols = [f'PRDAY{i}' for i in range(1, 26)]

# Step 1: Replace spaces (' ') with NaN in both procedure and day columns (HCUP stores " " for missing values)
NIS_ALL[procedure_cols + day_cols] = NIS_ALL[procedure_cols + day_cols].replace(' ', np.nan)

# Step 2: Identify rows where there is a procedure code but the corresponding day is missing (NaN)
# Step 1: Create a boolean DataFrame where True indicates a non-null procedure code
boolean_df_PR = NIS_ALL[procedure_cols].notna()
boolean_df_Day = NIS_ALL[day_cols].notna()

# Step 2: Count the number of True values for each row

NIS_ALL['Procedure_Count'] = boolean_df_PR.sum(axis=1)
NIS_ALL['Day_Count'] = boolean_df_Day.sum(axis=1)

# Create a new column based on the comparison
NIS_ALL['Missing'] = (NIS_ALL['Procedure_Count'] != NIS_ALL['Day_Count']).astype(int)

# delete the missing rows
NIS_All_DAY = NIS_ALL[NIS_ALL['Missing'] == 0]

######################### Work with the Procedure code ########################
# Reshape into a long dataframe
df_long = pd.melt(NIS_All_DAY, id_vars=['KEY_NIS','HOSP_NIS'], value_vars=procedure_cols, var_name='ICD_Code_Column', value_name='ICD_Code')
df_days = pd.melt(NIS_All_DAY, id_vars=['KEY_NIS','HOSP_NIS'], value_vars=day_cols, var_name='Day_Column', value_name='PRDay')

# Combine the ICD codes with the corresponding days
df_long['PRDay'] = df_days['PRDay']


# Find the PR code list which has Freq greater than 1%
icd_freq = df_long['ICD_Code'].value_counts()
frequent_codes = icd_freq[icd_freq > 0.01*len(NIS_All_DAY)].index

# Filter the long df to only include relevant ICD Codes
filtered_df = df_long[df_long['ICD_Code'].isin(frequent_codes)]
filtered_df.drop(columns=['ICD_Code_Column'], inplace=True)

# Convert 'PRDay' to numeric, setting errors='coerce' to convert non-numeric to NaN
filtered_df['PRDay'] = pd.to_numeric(filtered_df['PRDay'], errors='coerce')

# Pivot the DataFrame
pivot_df = filtered_df.pivot_table(index=['KEY_NIS','HOSP_NIS','PRDay'], columns='ICD_Code', aggfunc='size', fill_value=0) ## !! NEED to discuss this

# Ensure the values are binary (1 or 0)
NIS_ALL_PR = (pivot_df > 0).astype(int).reset_index()

# Exporting to CSV after resetting the index to include KEY_NIS and Day in the CSV file
#NIS_ALL_PR.to_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_ALL_PR.csv", index=False)

print("This is the columns in NIS_ALL_PR")
print(NIS_ALL_PR.columns.tolist())

# Get the description of the PR list
ICD_PR = pd.read_excel(r"C:\Users\Jiahui\PycharmProjects\NIS\ICDdescription.xlsx", sheet_name= "PR")
filtered_ICD_PR = ICD_PR[ICD_PR['ICD'].isin(frequent_codes)]
filtered_ICD_PR.reset_index().to_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_ALL_PR_list.csv", index=False)


############################## Work with the Diagnosis code ########################################
# Concatenate all DX columns for each patient
# Define the list of DX columns
dx_cols = [f'I10_DX{i}' for i in range(1, 41)]

# Create a new df for new DX columns with KEY_NIS
NIS_ALL_DX = NIS_ALL.copy()

# Values to replace: replace these value with Nan, so they can be created pon their own columns in prediction
values_to_replace = ['C9100', 'C9101', 'C9102','Z006']

# Replace values with NaN using np.where
NIS_ALL_DX = NIS_ALL_DX.replace(to_replace=values_to_replace, value=np.nan)

for col in dx_cols:
    NIS_ALL_DX[col] = NIS_ALL_DX[col].astype(str).str[:3]

# Convert grouping structure to a list of tuples
grouping = grouping_df[["ICD", "Start", "End"]].values.tolist()

# Function to map ICD codes to categories using the grouping list
def map_icd_to_category(code):
    for group, start, end in grouping:
        # Compare lexicographically
        if start <= code <= end:
            return group
    return np.nan  # If code doesn't fall in any range

# Apply the mapping function to each ICD column
for col in NIS_ALL_DX[dx_cols].columns:
    NIS_ALL_DX[f"{col}_Category"] = NIS_ALL_DX[col].apply(map_icd_to_category)

## Show max rows & columns if print
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

list_to_delete = ['Procedure_Count', 'Day_Count', 'Missing']
NIS_ALL_DX.drop(columns=list_to_delete, inplace=True)
NIS_ALL_DX.drop(columns=procedure_cols + day_cols + dx_cols, inplace=True)

# Replace spaces (' ') with NaN in both procedure and day columns (HCUP stores " " for missing values)
dx_cols_cate = [f'I10_DX{i}_Category' for i in range(1, 41)]
NIS_ALL_DX[dx_cols_cate] = NIS_ALL_DX[dx_cols_cate].replace(' ', np.nan)

# Drop any missing values
all_dx = NIS_ALL_DX[dx_cols_cate].apply(lambda row: row.dropna().tolist(), axis=1)


# Flatten the list to get all codes in the dataset
all_dx_codes = [code for sublist in all_dx for code in sublist]

# Count the frequency of each code
dx_freq = Counter(all_dx_codes)

# Convert to a DataFrame for easier readability
dx_freq_df = pd.DataFrame(dx_freq.items(), columns=['DX_Code', 'Frequency']).sort_values(by='Frequency', ascending=False)

# Get the list of ICD DX_Code which has the frequency greater than 1%
DX_list_1percent = dx_freq_df['DX_Code'][dx_freq_df['Frequency'] > (len(NIS_ALL_DX)*0.01)].tolist()

# Create a new column for each value in list
for value in DX_list_1percent:
    NIS_ALL_DX[value] = np.where(NIS_ALL_DX.filter(like='I10_DX').eq(value).any(axis=1), 1, 0)


# Get the DX ICD description
## ICD_DX = pd.read_excel(r"C:\Users\Jiahui\PycharmProjects\NIS\ICDdescription.xlsx", sheet_name= "DX")
## filtered_ICD_DX = ICD_DX[ICD_DX['ICD'].isin(DX_list_1percent)]
## filtered_ICD_DX.reset_index().to_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_ALL_DX_list.csv", index=False)

# Create a DataFrame from the list
#DX_list_1percent = [item for item in DX_list_1percent if item != 'n']
filter_ICD_DX = pd.DataFrame(DX_list_1percent, columns=['ICD'])
filter_ICD_DX.reset_index().to_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_ALL_DX_list.csv", index=False)

NIS_ALL_DX.drop(columns=dx_cols_cate, inplace=True)
#NIS_ALL_DX.drop('n', axis=1, inplace=True)


#NIS_ALL_DX.to_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_ALL_DX.csv", index=False)

## Merge the data

NIS_All_Clean_Merge = pd.merge(NIS_ALL_PR, NIS_ALL_DX, on=['KEY_NIS','HOSP_NIS'], how='left')

NIS_All_Clean_Merge.to_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_All_Clean_Merge.csv", index=False)

missing_values = NIS_All_Clean_Merge.isnull().sum()
print("Missing value 1")
print(missing_values)
############################## Work with the Procedure code without day ########################################
# Concatenate all DX columns for each patient
# Define the list of DX columns
pr_cols = [f'I10_PR{i}' for i in range(1, 26)]
key = ['KEY_NIS','HOSP_NIS']

# Create a new df for new DX columns with KEY_NIS
NIS_ALL_PR_2 = NIS_ALL[pr_cols + key]

print(NIS_ALL_PR_2.columns.tolist())

# Replace spaces (' ') with NaN in both procedure and day columns (HCUP stores " " for missing values)
NIS_ALL_PR_2[pr_cols] = NIS_ALL_PR_2[pr_cols].replace(' ', np.nan)

# Drop any missing values
all_pr = NIS_ALL_PR_2[pr_cols].apply(lambda row: row.dropna().tolist(), axis=1)

# Flatten the list to get all codes in the dataset
all_pr_codes = [code for sublist in all_pr for code in sublist]

# Count the frequency of each code
pr_freq = Counter(all_pr_codes)

# Convert to a DataFrame for easier readability
pr_freq_df = pd.DataFrame(pr_freq.items(), columns=['PR_Code', 'Frequency']).sort_values(by='Frequency', ascending=False)

# Get the list of ICD DX_Code which has the frequency greater than 1%
PR_list_2_percent = pr_freq_df['PR_Code'][pr_freq_df['Frequency'] > (len(NIS_ALL_PR_2)*0.01)].tolist()

# Create a new column for each value in list
for value in PR_list_2_percent:
    NIS_ALL_PR_2[value] = np.where(NIS_ALL_PR_2.filter(like='I10_PR').eq(value).any(axis=1), 1, 0)

# Get the DX ICD description
ICD_PR_2 = pd.read_excel(r"C:\Users\Jiahui\PycharmProjects\NIS\ICDdescription.xlsx", sheet_name= "PR")
filtered_ICD_PR_2 = ICD_PR_2[ICD_PR_2['ICD'].isin(PR_list_2_percent)]
filtered_ICD_PR_2.reset_index().to_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_ALL_PR_2_list.csv", index=False)

NIS_ALL_PR_2.drop(columns=pr_cols, inplace=True)

## Merge the data

NIS_All_Clean_Merge_2 = NIS_ALL_DX.merge( NIS_ALL_PR_2, on=['KEY_NIS','HOSP_NIS'], how='left')

missing_values = NIS_All_Clean_Merge_2.isnull().sum()
print("Missing value 2")
print(missing_values)

NIS_All_Clean_Merge_2.to_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_All_Clean_Merge_2.csv", index=False)

