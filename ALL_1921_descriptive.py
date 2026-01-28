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

# ## Specify the column dtype based on the error message
# dtypes = {'I10_DX36': 'str', 'I10_DX37': 'str', 'I10_DX38': 'str', 'I10_DX39': 'str', 'I10_DX40': 'str',
#           'I10_DX30': 'str', 'I10_DX31': 'str', 'I10_DX32': 'str', 'I10_DX33': 'str', 'I10_DX34': 'str',
#           'I10_DX35': 'str', 'I10_PR20': 'str', 'I10_PR21': 'str', 'I10_PR22': 'str', 'I10_PR23': 'str',
#           'I10_PR24': 'str', 'I10_PR25': 'str'}
# ## Show max rows & columns if print
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

## Read the data
# NIS_All_Clean_Merge = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_All_Clean_Merge.csv", dtype=dtypes)
#
# NIS_All_Clean = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_All_Clean.csv", dtype=dtypes)
#
# ##################### Define the list of column names #############################################
# Response_column = ['LOS','TOTCHG','DIED']
# numeric_columns = ['AGE']
# cmr_columns = ["CMR_AIDS","CMR_ALCOHOL","CMR_DEMENTIA","CMR_DEPRESS","CMR_DIAB_CX",
#                "CMR_DIAB_UNCX","CMR_DRUG_ABUSE","CMR_HTN_CX","CMR_HTN_UNCX","CMR_LUNG_CHRONIC","CMR_OBESE","CMR_PERIVASC",
#                "CMR_THYROID_HYPO","CMR_THYROID_OTH"]
# categorical_columns = ['YEAR', 'AWEEKEND', 'ELECTIVE', 'FEMALE', 'HCUP_ED',
#                        'I10_INJURY', 'I10_MULTINJURY','I10_SERVICELINE','PAY1', 'PCLASS_ORPROC',
#                        'PL_NCHS','RACE', 'TRAN_IN', 'YEAR', 'ZIPINC_QRTL','APRDRG_Risk_Mortality', 'APRDRG_Severity',
#                       'HOSP_BEDSIZE','HOSP_LOCTEACH', 'HOSP_REGION','H_CONTRL', 'KID', 'Area','Z006']
#
# ## Convert more variable to : 0 or 1
# def map_category_0_1(value):
#     if value in [1, 2, 3, 4]:
#         return 1
#     else:
#         return 0
# NIS_All_Clean['HCUP_ED'] = NIS_All_Clean['HCUP_ED'].apply(map_category_0_1).astype(int)
# NIS_All_Clean['I10_INJURY'] = NIS_All_Clean['I10_INJURY'].apply(map_category_0_1).astype(int)

############################### For response variable ######################################################
#### LOS plot ####
# # Get the important statistics (quartiles, median, etc.)
# desc = NIS_All_Clean['LOS'].describe()
# median = desc['50%']
# q1 = desc['25%']
# q3 = desc['75%']

# Create a boxplot without outliers
# plt.figure(figsize=(12, 2))
# sns.boxplot(x=NIS_All_Clean['LOS'], color='skyblue')
#
# # Annotate the important values on the boxplot
# plt.text(-5, 0.5, f'Q1: {q1}', horizontalalignment='center', color='green')
# plt.text(10, -0.4, f'Median: {median}', horizontalalignment='center', color='green')
# plt.text(20, 0.5, f'Q3: {q3}', horizontalalignment='center', color='green')
#
# # Customize and show the plot
# plt.title('Boxplot of LOS')
# plt.xlabel('LOS')
# plt.show()
# ########################
# # Calculate Q1 (25th percentile) and Q3 (75th percentile)
# Q1 = NIS_All_Clean['LOS'].quantile(0.25)
# Q3 = NIS_All_Clean['LOS'].quantile(0.75)
# IQR = Q3 - Q1
#
# # # Define the lower and upper bounds for non-outliers
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
#
# # Filter out the outliers
# NIS_All_clean_no_outliers = NIS_All_Clean[(NIS_All_Clean['LOS'] >= lower_bound) & (NIS_All_Clean['LOS'] <= upper_bound)]
#
# # Get the important statistics (quartiles, median, etc.)
# desc = NIS_All_clean_no_outliers['LOS'].describe()
# median = desc['50%']
# q1 = desc['25%']
# q3 = desc['75%']

# # Create a boxplot without outliers
# plt.figure(figsize=(12, 2))
# sns.boxplot(x=NIS_All_clean_no_outliers['LOS'], color='skyblue')
#
# # Annotate the important values on the boxplot
# plt.text(2.5, 0.5, f'Q1: {q1}', horizontalalignment='center', color='green')
# plt.text(4.5, 0.5, f'Median: {median}', horizontalalignment='center', color='green')
# plt.text(7, 0.5, f'Q3: {q3}', horizontalalignment='center', color='green')
#
# # Customize and show the plot
# plt.title('Boxplot of LOS (Removing outliers)')
# plt.xlabel('LOS')
# plt.show()

#### Plot for PLOS ####
## Create PLOS: >=75th percentile
# def map_PLOS(value):
#     if value >= q3:
#         return 1
#     else:
#         return 0
#
# NIS_All_Clean.loc[:,'PLOS'] = NIS_All_Clean.loc[:,'LOS'].apply(map_PLOS)

## Plot PLOS##
# sns.barplot(x=NIS_All_Clean['PLOS'].value_counts(), y=NIS_All_Clean['PLOS'].value_counts().index, orient='h', width=0.4, color='skyblue')
# # Customize and show the plot
# # Annotate the bars with the count values
# for i in range(len(NIS_All_Clean['PLOS'].value_counts())):
#     plt.annotate(str(NIS_All_Clean['PLOS'].value_counts()[i]), xy=(NIS_All_Clean['PLOS'].value_counts()[i], i), ha='left', va='center')
# plt.title('Barplot of PLOS')
# plt.xlabel('PLOS')
# plt.show()

#### Plot Mortality ####
# DIED = NIS_All_Clean['DIED'].dropna()
# DIED = DIED.astype(int)
# sns.barplot(x=DIED.value_counts(), y=DIED.value_counts().index, width=0.4, orient='h', color='skyblue')
# # Customize and show the plot
# # Annotate the bars with the count values
# for i in range(len(DIED.value_counts())):
#     plt.annotate(str(DIED.value_counts()[i]), xy=(DIED.value_counts()[i], i), ha='left', va='center')
# plt.title('Barplot of Mortality')
# plt.xlabel('Mortality')
# plt.show()

#### Plot TOTCHG ####
# Create a boxplot without outliers
# plt.figure(figsize=(12, 2))
# sns.boxplot(x=NIS_All_Clean['TOTCHG'], color='skyblue')
# plt.title('Boxplot of Total Charges')
# plt.xlabel('Total Charges')
# plt.show()

################################################## Descriptive analysis ##############################################
## Group by categorical column and calculate summary statistics for each value variable
# categorical_list = cmr_columns + categorical_columns
#PLOS_DIED = NIS_All_Clean[(NIS_All_Clean['LOS'].notnull()) & (NIS_All_Clean['DIED'].notnull())]
# summary_per_variable_PLOS = {}
# for column in categorical_list:
#     summary_per_variable_PLOS[column] = NIS_All_Clean.groupby('PLOS')[column].apply(pd.value_counts) #only get counts
#     counts = NIS_All_Clean.groupby('PLOS')[column].value_counts().unstack(fill_value=0)
#     relative_frequency = counts.div(len(NIS_All_Clean[column]))  # Normalize counts to get relative frequency
#     relative_frequency = relative_frequency.round(4)*100  # Round to 2 decimal places
#     summary_per_variable_PLOS[column] = pd.concat([counts, relative_frequency], axis=1, keys=['Count', 'Relative Frequency'])
#
# ## Print the summary statistics for each value variable
# for column, summary in summary_per_variable_PLOS.items():
#     print(f"Summary for {column}:")
#     print(summary)
#     print()

# summary_per_variable_DIED = {}
# for column in categorical_list:
#     summary_per_variable_DIED[column] = NIS_All_Clean.groupby('DIED')[column].apply(pd.value_counts) #only get counts
#     counts = NIS_All_Clean.groupby('DIED')[column].value_counts().unstack(fill_value=0)
#     relative_frequency = counts.div(len(NIS_All_Clean[column]))  # Normalize counts to get relative frequency
#     relative_frequency = relative_frequency.round(4)*100  # Round to 2 decimal places
#     summary_per_variable_DIED[column] = pd.concat([counts, relative_frequency], axis=1, keys=['Count', 'Relative Frequency'])
#
# ## Print the summary statistics for each value variable
# for column, summary in summary_per_variable_DIED.items():
#     print(f"Summary for {column}:")
#     print(summary)
#     print()


# summary_per_variable_TOTCHG = {}
# for column in categorical_list:
#     summary_per_variable_TOTCHG[column] = NIS_All_Clean['TOTCHG'].groupby(NIS_All_Clean[column]).describe()
#
# ## Print the summary statistics for each value variable
# for column, summary in summary_per_variable_TOTCHG.items():
#     print(f"Summary for {column}:")
#     print(summary)
#     print()

#print(NIS_All_Clean['AGE'].groupby(NIS_All_Clean['PLOS']).describe())
#print(NIS_All_Clean['AGE'].groupby(NIS_All_Clean['DIED']).describe())

# print(NIS_All_Clean['AGE'].describe())
# print(NIS_All_Clean['LOS'].describe())
# print(NIS_All_Clean['TOTCHG'].describe())
#
# ### Find assiociation between outcomes with ANOVA and CHI square ###
# import scipy.stats as stats
# ## FIND the association between charges and PLOS
# PLOS_TOTCHG = NIS_All_Clean[(NIS_All_Clean['LOS'].notnull()) & (NIS_All_Clean['TOTCHG'].notnull())]
# # Get non-empty groups of TOTCHG for each unique category in PLOS
# groups = [PLOS_TOTCHG['TOTCHG'][PLOS_TOTCHG['PLOS'] == category]
#           for category in PLOS_TOTCHG['PLOS'].unique()
#           if len(PLOS_TOTCHG['TOTCHG'][PLOS_TOTCHG['PLOS'] == category]) > 0]
#
# # Perform the one-way ANOVA
# f_statistic, p_value = stats.f_oneway(*groups)
# formatted_p_value = format(p_value, '.10f')
# print("ANOVA test results for PLOS and Total charges")
# print("F-statistic:", f_statistic)
# print("p-value:", formatted_p_value)
#
# ## FIND the association between charges and DIED
# DIED_TOTCHG = NIS_All_Clean[(NIS_All_Clean['DIED'].notnull()) & (NIS_All_Clean['TOTCHG'].notnull())]
# # Get non-empty groups of TOTCHG for each unique category in PLOS
# groups = [DIED_TOTCHG['TOTCHG'][DIED_TOTCHG['DIED'] == category]
#           for category in DIED_TOTCHG['DIED'].unique()
#           if len(DIED_TOTCHG['TOTCHG'][DIED_TOTCHG['DIED'] == category]) > 0]
#
# # Perform the one-way ANOVA
# f_statistic, p_value = stats.f_oneway(*groups)
# formatted_p_value = format(p_value, '.10f')
# print("ANOVA test results for DIED and Total charges")
# print("F-statistic:", f_statistic)
# print("p-value:", formatted_p_value)
#
# ## FIND the association between DIED and PLOS
# from scipy.stats import chi2_contingency;
# DIED_PLOS = NIS_All_Clean[(NIS_All_Clean['DIED'].notnull()) & (NIS_All_Clean['LOS'].notnull())]
# contingency_table = pd.crosstab(DIED_PLOS['PLOS'], DIED_PLOS['DIED'])
# chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
# print("Chi-square results for PLOS and Mortality")
# print(p_val)
# print(contingency_table)

## Calcualate the Frequency of PR list
NIS_All_Clean_Merge_2 = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_All_Clean_Merge_2.csv")
NIS_ALL_PR = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_ALL_PR_list.csv")
PR_list = NIS_ALL_PR['ICD'].tolist()

NIS_ALL_DX_R = pd.read_excel(r"C:\Users\Jiahui\PycharmProjects\NIS\ICD_regrouping.xlsx", sheet_name='Mapping_long_R')
DX_list = NIS_ALL_DX_R[NIS_ALL_DX_R['RELEVANT'] == 1]['ICD'].tolist()


sums_pr = NIS_All_Clean_Merge_2[PR_list].sum()
print(" PR: ")
print(sums_pr)

sums_dx = NIS_All_Clean_Merge_2[DX_list].sum()
print(" DX: ")
print(sums_dx)

### Create figures for important fearures



