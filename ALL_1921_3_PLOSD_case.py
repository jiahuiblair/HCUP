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
NIS_All_Clean_Merge_case = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_ALL_Clean_Z006.csv", dtype=dtypes)

print("NIS_All_Clean_Merge_case", len(NIS_All_Clean_Merge_case))

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


# Combine PLOS and DIED into single categorical outcome
def map_PLOSD(value):
    return {"00": "0", "10": "1", "01": "2", "11": "3"}.get(value, "")


NIS_All_Clean_Merge['PLOSDc'] = NIS_All_Clean_Merge['PLOS'].astype(str) + NIS_All_Clean_Merge['DIED'].astype(str)
NIS_All_Clean_Merge['PLOSD'] = NIS_All_Clean_Merge['PLOSDc'].apply(map_PLOSD)

NIS_All_Clean_Merge['DIED'] = NIS_All_Clean_Merge['DIED'].replace(9, np.nan)

# Handle PRDay data to create PRDay_model
def map_PRDay(value):
    if value <= 0:
        return 0
    elif value > q3:
        return q3+1
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
#### Filter the NIS_ALL_Clean_Merge with KEY_NIS for case_set
case_set_final = pd.merge(
    NIS_All_Clean_Merge_final,
    NIS_All_Clean_Merge_case[['KEY_NIS']],  # Select only the 'KEY_NIS' column
    on='KEY_NIS',
    how='inner'  # Keep only matching rows
)

print("case_set_final",len(case_set_final))

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
PRDay_list =  sorted([day for day in NIS_All_Clean_Merge['PRDay_model'].unique() if day != 0])

print("Total number of features included in the model")
print(len(Total_x_list))

# Initialize the results DataFrame for evaluation storage
column_names = ['Outcome', 'Day', 'Train_accuracy_rf', 'Test_accuracy_rf', 'Train_accuracy_gb', 'Test_accuracy_gb']
outcome_df = pd.DataFrame(columns=column_names)

PRDay_list = [x for x in PRDay_list if x != np.float64(1.0)]
PRDay_list = [x for x in PRDay_list if x != np.float64(8.0)]

print(PRDay_list)

# Loop through outcomes and days

for day in PRDay_list:#PRDay_list[1.0]
    print("######################################################")
    print(f"Prediction on day: {day}")

    # Update the
    model_data = NIS_All_Clean_Merge_final[Total_list].copy()
    model_data = model_data[NIS_All_Clean_Merge_final['LOS'] >= day]
    model_data.loc[NIS_All_Clean_Merge_final['PRDay_model'] > day, PR_list] = 0
    model_data = model_data[~model_data[Response_column].apply(lambda row: row.eq('').any(), axis=1)]
    model_data = model_data.dropna(subset=Response_column)

    # the case data preparation
    case_set_final = case_set_final[~case_set_final[Response_column].apply(lambda row: row.eq('').any(), axis=1)]
    case_set_final = case_set_final.dropna(subset=Response_column)

    model_case = case_set_final[Total_list].copy()
    model_case = model_case[case_set_final['LOS'] >= day]
    case_set_final = case_set_final[case_set_final['LOS'] >= day]

    model_case.loc[case_set_final['PRDay_model'] > day, PR_list] = 0



    # Separate features and target
    X = model_data[Total_x_list]
    y = model_data[Response_column].astype(int)

    case_X = model_case[Total_x_list].copy()
    case_y_list = ['KEY_NIS', 'PLOSD']
    case_y = case_set_final[case_y_list].copy()

    # Split data into train and test
    X_train, X_test_0, y_train, y_test_0 = train_test_split(X, y, test_size=0.2, random_state=42)

    print(len(X_train[X_train['Z006']==1]))
    print(len(X_test_0[X_test_0['Z006'] == 1]))

    for outcome in Response_column:
        print("######################################################")
        print(f"Evaluating outcome: {outcome}")

        # SMOTE balance method
        smote = SMOTE(sampling_strategy='auto', random_state=42)

        # Fit and resample
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train[outcome])

        # Scale continuous features
        scaler = StandardScaler()
        X_train_resampled['AGE'] = scaler.fit_transform(X_train_resampled[['AGE']])
        case_X['AGE'] = scaler.transform(case_X[['AGE']])

        X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train_resampled.columns)

        # Initialize models and RFECV for feature selection
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rfecv_rf = RFECV(estimator=rf, step=20, cv=5, scoring='accuracy')

        # Fit RFECV and get selected features
        rfecv_rf.fit(X_train_resampled, y_train_resampled)

        selected_features_rf = list(X_train_resampled.columns[rfecv_rf.support_])
        print("Length of selected rf", len(selected_features_rf))


        # Remove duplicates while preserving the order
        selected_features_rf = list(dict.fromkeys(selected_features_rf))
        print("Unique selected features rf:", len(selected_features_rf))

        # Subset data for selected features && remove the duplicate features
        X_train_selected_rf = X_train_resampled.loc[:, ~X_train_resampled.columns.duplicated()][selected_features_rf]
        X_test_selected_rf = case_X.loc[:, ~case_X.columns.duplicated()][selected_features_rf]

        print("length columns in X_train_selected rf", len(X_test_selected_rf.columns))

        # Fit models and predict

        # Perform Stratified K-Fold cross-validation
        #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # # Define scoring metrics for PLOS DIED
        # scoring = {
        #     'accuracy': make_scorer(accuracy_score),
        #     'f1': make_scorer(f1_score),
        #     'recall': make_scorer(recall_score),
        #     'precision': make_scorer(precision_score),
        #     #'roc_auc': 'roc_auc'
        # }

        # Define scoring with appropriate average for multiclass for PLOSD
        scoring = {
            'accuracy': 'accuracy',  # Accuracy doesn't require 'average'
            'f1': make_scorer(f1_score, average='weighted'),
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'roc_auc': 'roc_auc_ovr',  # For multiclass AUC
        }

        # # Perform cross-validation with 5 folds
        ## cv_scores_rf = cross_val_score(rf, X_train_selected_rf, y_train_resampled, cv=5)
        ## cv_scores_gb = cross_val_score(gb, X_train_selected_gb, y_train_resampled, cv=5)

        cv_results_rf = cross_validate(
            rf, X_train_selected_rf, y_train_resampled,
            cv=5, scoring=scoring, return_train_score=False
        )


        # Calculate mean cross-validation metrics
        ## cv_mean_rf = cv_scores_rf.mean()
        ## cv_mean_gb = cv_scores_gb.mean()

        # Calculate averaged metrics for Random Forest
        cv_mean_rf = {metric: cv_results_rf[f'test_{metric}'].mean() for metric in scoring.keys()}

        # fit the model on entire training set
        rf.fit(X_train_selected_rf, y_train_resampled)

        # Make predictions on the test set
        y_pred_rf = rf.predict(X_test_selected_rf)

        # Add the prediction value
        # Create dynamic column names
        rf_col_name = f"RF_Prediction_{outcome}"

        # Add predictions as new columns
        case_y[rf_col_name] = y_pred_rf

        # Store results in a DataFrame
        new_row = pd.DataFrame([{
            'Day': day,
            'Outcome': outcome
        }])


        # Append the results to the existing DataFrame
        outcome_df = pd.concat([outcome_df, new_row], ignore_index=True)
    # filename = fr"C:\Users\Jiahui\PycharmProjects\NIS\PLOSM_case\Prediction_summary_case_{day}.csv"
    # outcome_df.to_csv(filename, index=False)

    # Save the DataFrame to a CSV file
    filename1 = fr"C:\Users\Jiahui\PycharmProjects\NIS\PLOSM_case\Prediction_data_output_Z006_new_{day}.csv"
    case_y.to_csv(filename1, index=False)


