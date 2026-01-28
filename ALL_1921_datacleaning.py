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
#### Create a list to show the columns needs to be deleted from the dataset; These features were not selected to included in the prediction model
list_to_delete = ['AMONTH',"AGE_NEONATE","DISCWT","DRG","DRGVER","DRG_NoPOA","I10_BIRTH","MDC_NoPOA","NIS_STRATUM","APRDRG",
                  "N_DISC_U","N_HOSP_U","S_DISC_U","S_HOSP_U","TOTAL_DISC","DISPUNIFORM", "TRAN_OUT", "DQTR", "CMR_ARTH",
                  "CMR_AUTOIMMUNE", "CMR_CANCER_LEUK", "CMR_CANCER_LYMPH",
                  "CMR_CANCER_METS", "CMR_CANCER_NSITU", "CMR_CANCER_SOLID"]
NIS_All.drop(columns=list_to_delete, inplace=True)

#### Create columns list
Response_column = ['LOS','TOTCHG','DIED'] # All the patient outcomes (LOS: Length of stay, TOTCHG: Total charges, DIED: Mortality)

pr_cols = [f'I10_PR{i}' for i in range(1, 26)] # A list of ICD procedure feature names. I10_PR ranges from 1 to 25 to contain up to 25 procedure code for each record.
day_cols = [f'PRDAY{i}' for i in range(1, 26)] # A corresponding day feature names. PRDay ranges from 1 to 25 to show which day each procedure was performed to each I10_PR code.
dx_cols = [f'I10_DX{i}' for i in range(1, 41)] # A list of ICD diagnosis feature names. I10_DX ranges from 1 to 40 to contain up to 40 diagnosis code for each record.

numeric_cols = ['AGE'] # Continuous features 

cmr_columns = ["CMR_AIDS","CMR_ALCOHOL","CMR_DEMENTIA","CMR_DEPRESS","CMR_DIAB_CX",
               "CMR_DIAB_UNCX","CMR_DRUG_ABUSE","CMR_HTN_CX","CMR_HTN_UNCX","CMR_LUNG_CHRONIC","CMR_OBESE","CMR_PERIVASC",
               "CMR_THYROID_HYPO","CMR_THYROID_OTH"] # CMR commorbidity diagnosis. 

#### Create new columns

## Create Z006 to indicate if the pt in clinical trial or not. Z006 is the ICD diagnosis code to show if the visit involves clinical trial activities or not.
NIS_All["Z006"] = 0 # No clinical trial activity and assign 0
NIS_All["Z006"] = (NIS_All.loc[:, "I10_DX1":"I10_DX40"] == 'Z006').any(axis=1).astype(int) # Check all I10_DX columns to find clinical trial activity and assign 1

## Create the reamission information
ALL_list = ['C9100','C9101','C9102'] # ICD code for NEO059: patients diagnosis with Acute Lymphoblastic Leukemia (ALL). C9100, C9101 and C9102 show different disease progress with ALL.

# Create a new column for each value in ALL_list
for value in ALL_list:
    NIS_All[value] = np.where(NIS_All.filter(like='I10_DX').eq(value).any(axis=1), 1, 0)

## Create Area for patient location
def map_category_Area(value):
    if value in [1, 2]:
        return "0" # Urban
    elif value in [3, 4]:
        return "1" # Transfer from Urban to Rural: Metro counties
    elif value in [5, 6]:
        return "2" # Rural
    else:
        return np.nan
## Create the new variable based on the mapping
NIS_All.loc[:,'Area'] = NIS_All.loc[:,'PL_NCHS'].apply(map_category_Area)

categorical_columns = ['YEAR', 'AWEEKEND', 'ELECTIVE', 'FEMALE', 'HCUP_ED',
                       'I10_INJURY', 'I10_MULTINJURY','I10_SERVICELINE','PAY1', 'PCLASS_ORPROC',
                       'PL_NCHS','RACE', 'TRAN_IN', 'YEAR', 'ZIPINC_QRTL','APRDRG_Risk_Mortality', 'APRDRG_Severity',
                      'HOSP_BEDSIZE','HOSP_LOCTEACH', 'HOSP_REGION','H_CONTRL', 'Area','C9100','C9101','C9102']

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


NIS_All.to_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_ALL_Clean.csv", index=False)

