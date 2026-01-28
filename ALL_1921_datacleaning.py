import numpy as np;
import pandas as pd;
import matplotlib as mpl;
import sklearn as skl;
from scipy.stats import f_oneway;
import matplotlib.pyplot as plt;
import seaborn as sns;
from collections import Counter;
from scipy.stats import gaussian_kde;
from mlxtend.frequent_patterns import apriori;
from mlxtend.frequent_patterns import association_rules;
from mlxtend.preprocessing import TransactionEncoder;
from sklearn.preprocessing import OneHotEncoder;
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

## Specify the column dtype based on the error message
dtypes = {'I10_DX36': 'str', 'I10_DX37': 'str', 'I10_DX38': 'str', 'I10_DX39': 'str', 'I10_DX40': 'str',
          'I10_DX30': 'str', 'I10_DX31': 'str', 'I10_DX32': 'str', 'I10_DX33': 'str', 'I10_DX34': 'str',
          'I10_DX35': 'str', 'I10_PR20': 'str', 'I10_PR21': 'str', 'I10_PR22': 'str', 'I10_PR23': 'str',
          'I10_PR24': 'str', 'I10_PR25': 'str'}

## Read the 2019 data
NIS_2019_Core = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_2019_Core.csv", dtype=dtypes)

## Read the 2021 data
NIS_2021_Core = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_2021_Core.csv", dtype=dtypes)

## Combine 2019 and 2021 data
NIS_All = pd.concat([NIS_2019_Core, NIS_2021_Core], axis=0, join='outer', ignore_index=True)

## Show max rows & columns if print
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

######################################## Clean the data ########################################################
#### Create a list to show the columns needs to be deleted from the dataset
list_to_delete = ['AMONTH',"AGE_NEONATE","DISCWT","DRG","DRGVER","DRG_NoPOA","I10_BIRTH","MDC_NoPOA","NIS_STRATUM","APRDRG",
                  "N_DISC_U","N_HOSP_U","S_DISC_U","S_HOSP_U","TOTAL_DISC","DISPUNIFORM", "TRAN_OUT", "DQTR", "CMR_ARTH",
                  "CMR_AUTOIMMUNE", "CMR_CANCER_LEUK", "CMR_CANCER_LYMPH",
                  "CMR_CANCER_METS", "CMR_CANCER_NSITU", "CMR_CANCER_SOLID"]
NIS_All.drop(columns=list_to_delete, inplace=True)

#### Create columns list
Response_column = ['LOS','TOTCHG','DIED']
pr_cols = [f'I10_PR{i}' for i in range(1, 26)]
day_cols = [f'PRDAY{i}' for i in range(1, 26)]
dx_cols = [f'I10_DX{i}' for i in range(1, 41)]

numeric_cols = ['AGE']

cmr_columns = ["CMR_AIDS","CMR_ALCOHOL","CMR_DEMENTIA","CMR_DEPRESS","CMR_DIAB_CX",
               "CMR_DIAB_UNCX","CMR_DRUG_ABUSE","CMR_HTN_CX","CMR_HTN_UNCX","CMR_LUNG_CHRONIC","CMR_OBESE","CMR_PERIVASC",
               "CMR_THYROID_HYPO","CMR_THYROID_OTH"]

#### Create new columns

## Create Z006 to indicate if the pt in clinical trial or not
NIS_All["Z006"] = 0
NIS_All["Z006"] = (NIS_All.loc[:, "I10_DX1":"I10_DX40"] == 'Z006').any(axis=1).astype(int)

## Create the reamission information
list = ['C9100','C9101','C9102'] # ICD code for NEO059
# Create a new column for each value in list
for value in list:
    NIS_All[value] = np.where(NIS_All.filter(like='I10_DX').eq(value).any(axis=1), 1, 0)

## Create Area for patient location
def map_category_Area(value):
    if value in [1, 2]:
        return "0"
    elif value in [3, 4]:
        return "1"
    elif value in [5, 6]:
        return "2"
    else:
        return np.nan
## Create the new variable based on the mapping
NIS_All.loc[:,'Area'] = NIS_All.loc[:,'PL_NCHS'].apply(map_category_Area)

categorical_columns = ['YEAR', 'AWEEKEND', 'ELECTIVE', 'FEMALE', 'HCUP_ED',
                       'I10_INJURY', 'I10_MULTINJURY','I10_SERVICELINE','PAY1', 'PCLASS_ORPROC',
                       'PL_NCHS','RACE', 'TRAN_IN', 'YEAR', 'ZIPINC_QRTL','APRDRG_Risk_Mortality', 'APRDRG_Severity',
                      'HOSP_BEDSIZE','HOSP_LOCTEACH', 'HOSP_REGION','H_CONTRL', 'Area','C9100','C9101','C9102']
MDC_col = ["MDC"]

#### Deal with the missing value
## Replace the missing value: Based on HCUP data description, convert some missing vlaue to nan
NIS_All.replace(-999999999, np.nan, inplace=True)
NIS_All.replace(-99, np.nan, inplace=True)
NIS_All.replace(-9, np.nan, inplace=True)
NIS_All.replace(-8, np.nan, inplace=True)
NIS_All.replace(-6666, np.nan, inplace=True)
NIS_All.replace(-6, np.nan, inplace=True)

## Replace spaces (' ') with NaN in both procedure and day columns (HCUP stores " " for missing values)
NIS_All[pr_cols + day_cols + dx_cols] = NIS_All[pr_cols + day_cols + dx_cols].replace(' ', np.nan)

## Remove the whitespace may exist in the data
int_cols = Response_column + numeric_cols + cmr_columns + categorical_columns + MDC_col

for col in int_cols:
    # Convert columns to strings to handle whitespace (including integer columns)
    NIS_All[col] = NIS_All[col].astype(str)
    # Remove leading/trailing whitespace
    NIS_All[col] = NIS_All[col].str.strip()
    # Convert them back to int (use 'errors=coerce' to handle cases where conversion may fail)
    NIS_All[col] = pd.to_numeric(NIS_All[col], errors='coerce')

## Compute the missing values in each column
missing_values = NIS_All.isnull().sum()
print(missing_values)

#### Impute the missing value
missing_continuous = ["AGE"]
missing_binary = ["FEMALE", "ELECTIVE"]
missing_categorical = ["PAY1", "TRAN_IN", "PL_NCHS", "ZIPINC_QRTL", "RACE"]

## Replace missing values with the Mean/Median
NIS_All["AGE"] = pd.to_numeric(NIS_All['AGE'], errors='coerce')

for col in missing_continuous:
    NIS_All[col].fillna(NIS_All[col].median(), inplace=True)

## Replace missing values with the most frequent value (mode)
for col in missing_binary:
    NIS_All[col].fillna(NIS_All[col].mode()[0], inplace=True)

## Replace missing values with the most frequent value (mode) and one hot encoder
for col in missing_categorical:
    NIS_All[col].fillna(NIS_All[col].mode()[0], inplace=True)

## Create KiD versus adults by AGE
def map_category_AGE(value):
    if value <= 18 :
        return 1
    elif value > 18:
        return 0
    else:
        return np.nan
## Create the new variable based on the mapping

NIS_All.loc[:,'KID'] = NIS_All.loc[:,'AGE'].apply(map_category_AGE)

missing_values = NIS_All.isnull().sum()
print("Missing value", missing_values)

## PLOT ##################################
# Create histogram plot for continuous variable
#for column in missing_continuous:
    # plot histogram
#    plt.hist(NIS_All[column], bins='auto',density=True, alpha=0.5, edgecolor='k') # alpha set transparency for better visual; 'k' black edge color
    # plot density curve
#    data = NIS_All[column].dropna()
#    kde = gaussian_kde(data)
#    x_vals = np.linspace(data.min(),data.max(),1000)
#    plt.plot(x_vals, kde(x_vals), color='red', linestyle='-',linewidth=2,label='Density')

#    plt.title(f'Distribution of {column}')
#    plt.xlabel(column)
#    plt.ylabel('Density')
#    plt.grid(True)
#    plt.show()

# Create bar plot for categorical variables
#num_plots = len(missing_categorical)
#fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))  # Adjust figure size as needed
#axes = axes.flatten()  # Flatten axes to easily iterate

# Iterate over columns and create bar plots
#for i, column in enumerate(missing_categorical):
    # Count the occurrences of each category
#    value_counts = NIS_All[column].value_counts()

    # Create a bar plot in the corresponding subplot
#    sns.barplot(x=value_counts.index, y=value_counts, ax=axes[i], width=0.4)  # Fixed width for bars

    # Set axis labels
#    axes[i].set_xlabel(column)
#    axes[i].set_ylabel("Count")

    # Set x-axis ticks to integer values
#    x_ticks = range(len(value_counts))
#    axes[i].set_xticks(x_ticks)
    # Set x-axis labels to the corresponding integer values
#    axes[i].set_xticklabels(x_ticks) # Rotate x-labels for better readability

# Remove any empty subplots if the number of columns is less than 9
#for j in range(i + 1, 9):
#    fig.delaxes(axes[j])

# Adjust spacing and display the plot
#plt.tight_layout()
#plt.show()
####################################################


NIS_All.to_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_ALL_Clean.csv", index=False)
