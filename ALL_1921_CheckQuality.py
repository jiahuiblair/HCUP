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

######################################## Clean the data ########################################################
## Show max rows & columns if print
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

## replace the missing value
NIS_All.replace(-999999999, np.nan, inplace=True)
NIS_All.replace(-99, np.nan, inplace=True)
NIS_All.replace(-9, np.nan, inplace=True)
NIS_All.replace(-8, np.nan, inplace=True)
NIS_All.replace(-6666, np.nan, inplace=True)
NIS_All.replace(-6, np.nan, inplace=True)

## Create the reamission information
list_ALL = ['C9100','C9101','C9102'] # ICD code for NEO059
# Create a new column for each value in list
for value in list_ALL:
    NIS_All[value] = np.where(NIS_All.filter(like='I10_DX').eq(value).any(axis=1), 1, 0)

## Create KiD versus adults by AGE
def map_category_AGE(value):
    if value <= 18 :
        return 1
    else:
        return 0
## Create the new variable based on the mapping
NIS_All["AGE"] = pd.to_numeric(NIS_All['AGE'], errors='coerce')
NIS_All.loc[:,'KID'] = NIS_All.loc[:,'AGE'].apply(map_category_AGE)

## Function to map values to categories
def map_category_Area(value):
    if value in [1, 2]:
        return "0"
    elif value in [3, 4]:
        return "1"
    elif value in [5, 6]:
        return "2"
    else:
        return ''
## Create the new variable based on the mapping
NIS_All.loc[:,'Area'] = NIS_All.loc[:,'PL_NCHS'].apply(map_category_Area)

## Create Z006 to indicate if the pt in clinical trial or not
NIS_All["Z006"] = 0
NIS_All["Z006"] = (NIS_All.loc[:, "I10_DX1":"I10_DX40"] == 'Z006').any(axis=1).astype(int)

#####################################################################################

# Define the list of procedure and corresponding day columns
procedure_cols = [f'I10_PR{i}' for i in range(1, 26)]
day_cols = [f'PRDAY{i}' for i in range(1, 26)]

# Step 1: Replace spaces (' ') with NaN in both procedure and day columns
NIS_All[procedure_cols + day_cols] = NIS_All[procedure_cols + day_cols].replace(' ', np.nan)

# Step 2: Identify rows where there is a procedure code but the corresponding day is missing (NaN)
# Step 1: Create a boolean DataFrame where True indicates a non-null procedure code
boolean_df_PR = NIS_All[procedure_cols].notna()
boolean_df_Day = NIS_All[day_cols].notna()






# Step 2: Count the number of True values for each row

NIS_All['Procedure_Count'] = boolean_df_PR.sum(axis=1)
NIS_All['Day_Count'] = boolean_df_Day.sum(axis=1)

# Create a new column based on the comparison
NIS_All['Missing'] = (NIS_All['Procedure_Count'] != NIS_All['Day_Count']).astype(int)

print(NIS_All['Missing'].sum())


