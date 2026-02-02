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

# Regrouping HCUP_ED and I10_INJURY as binary 
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
NIS_All_Clean_Merge['PRDay_model'] = NIS_All_Clean_Merge['PRDay'].apply(map_PRDay) # Regroup the PRDay: if procedure provided before admission, count as day 0 
print(NIS_All_Clean_Merge['PRDay_model'].unique())

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
binary_columns = ['KID','C9100','C9102','Z006'] # Exclude the C9101 (Remission)

# Define feature sets and outcome columns
Response_column = ['PLOS','DIED'] 

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

for day in [1.0]: # To predict the outcome for patients staying in hospital longer than 1, use PRDay_list to replace 1.0
    print("######################################################")
    print(f"Prediction on day: {day}")

    # Update the
    model_data = NIS_All_Clean_Merge_final[Total_list].copy()
    #model_data = model_data[NIS_All_Clean_Merge_final['LOS'] >= day]
    model_data.loc[NIS_All_Clean_Merge_final['PRDay_model'] > day, PR_list] = 0

    print("Length of records", len(model_data))


    model_data = model_data[~model_data[Response_column].apply(lambda row: row.eq('').any(), axis=1)]
    model_data = model_data.dropna(subset=Response_column)


    # missing_values = model_data.isnull().sum()
    # print("Missing value 1")
    # print(missing_values)

    # Separate features and target
    X = model_data[Total_x_list]
    y = model_data[Response_column].astype(int)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check for missing values in X_train
    #missing_in_X_train = X_train.isnull().sum().sum()
    #print(f"Missing values in X_train: {missing_in_X_train}")

    # Check for missing values in y_train
    #missing_in_y_train = y_train.isnull().sum()
    #print(f"Missing values in y_train: {missing_in_y_train}")


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
        X_test['AGE'] = scaler.transform(X_test[['AGE']])

        X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train_resampled.columns)

        # Initialize models and RFECV for feature selection
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        rfecv_rf = RFECV(estimator=rf, step=20, cv=5, scoring='accuracy')
        rfecv_gb = RFECV(estimator=gb, step=20, cv=5, scoring='accuracy')

        # Fit RFECV and get selected features
        rfecv_rf.fit(X_train_resampled, y_train_resampled)
        rfecv_gb.fit(X_train_resampled, y_train_resampled)

        selected_features_rf = list(X_train_resampled.columns[rfecv_rf.support_])
        selected_features_gb = list(X_train_resampled.columns[rfecv_gb.support_])
        print("Length of selected rf", len(selected_features_rf))
        print("Length of selected gb", len(selected_features_gb))


        # Remove duplicates while preserving the order
        selected_features_rf = list(dict.fromkeys(selected_features_rf))
        print("Unique selected features rf:", len(selected_features_rf))

        selected_features_gb = list(dict.fromkeys(selected_features_gb))
        print("Unique selected features gb:", len(selected_features_gb))

        # Subset data for selected features && remove the duplicate features
        X_train_selected_rf = X_train_resampled.loc[:, ~X_train_resampled.columns.duplicated()][selected_features_rf]
        X_test_selected_rf = X_test.loc[:, ~X_test.columns.duplicated()][selected_features_rf]
        X_train_selected_gb = X_train_resampled.loc[:, ~X_train_resampled.columns.duplicated()][selected_features_gb]
        X_test_selected_gb = X_test.loc[:, ~X_test.columns.duplicated()][selected_features_gb]

        print("length columns in X_train_selected rf", len(X_test_selected_rf.columns))
        print("length columns in X_train_selected gb", len(X_test_selected_gb.columns))

        # Fit models and predict

        # Perform Stratified K-Fold cross-validation
        #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Define scoring metrics for PLOS DIED
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score),
            'recall': make_scorer(recall_score),
            'precision': make_scorer(precision_score),
            'roc_auc': 'roc_auc'
        }

        # # Perform cross-validation with 5 folds
        ## cv_scores_rf = cross_val_score(rf, X_train_selected_rf, y_train_resampled, cv=5)
        ## cv_scores_gb = cross_val_score(gb, X_train_selected_gb, y_train_resampled, cv=5)

        cv_results_rf = cross_validate(
            rf, X_train_selected_rf, y_train_resampled,
            cv=5, scoring=scoring, return_train_score=False
        )

        cv_results_gb = cross_validate(
            gb, X_train_selected_gb, y_train_resampled,
            cv=5, scoring=scoring, return_train_score=False
        )


        # Calculate averaged metrics for Random Forest
        cv_mean_rf = {metric: cv_results_rf[f'test_{metric}'].mean() for metric in scoring.keys()}

        # Calculate averaged metrics for Gradient Boosting
        cv_mean_gb = {metric: cv_results_gb[f'test_{metric}'].mean() for metric in scoring.keys()}

        # fit the model on entire training set
        rf.fit(X_train_selected_rf, y_train_resampled)
        gb.fit(X_train_selected_gb, y_train_resampled)

        # Make predictions on the test set
        y_pred_rf = rf.predict(X_test_selected_rf)
        y_pred_gb = gb.predict(X_test_selected_gb)

        # Add the prediction value
        # Create dynamic column names
        rf_col_name = f"RF_Prediction_{outcome}"
        gb_col_name = f"GB_Prediction_{outcome}"

        # Add predictions as new columns
        y_test[rf_col_name] = y_pred_rf
        y_test[gb_col_name] = y_pred_gb

        ################### For 2 levels outcome ####################
        # Make predictions prob on the test set
        rf_probs = rf.predict_proba(X_test_selected_rf)[:,1] # for the PLOS DIED
        gb_probs = gb.predict_proba(X_test_selected_gb)[:,1] # for the PLOS DIED

        # Add the prediction value
        # Create dynamic column names
        rf_prob_name = f"RF_Prob_{outcome}" # for the PLOS DIED
        gb_prob_name = f"GB_Prob_{outcome}" # for the PLOS DIED

        # Add predictions as new columns
        y_test[rf_prob_name] = rf_probs # for the PLOS DIED
        y_test[gb_prob_name] = gb_probs # for the PLOS DIED

        # Calculate accuracy scores
        train_accuracy_rf = rf.score(X_train_selected_rf, y_train_resampled)
        test_accuracy_rf = rf.score(X_test_selected_rf, y_test[outcome])

        train_accuracy_gb = gb.score(X_train_selected_gb, y_train_resampled)
        test_accuracy_gb = gb.score(X_test_selected_gb, y_test[outcome])

        # Calculate F1 score, precision, and recall
        f1_rf = f1_score(y_test[outcome], y_pred_rf, average='weighted')  # Use 'weighted' for multiclass
        precision_rf = precision_score(y_test[outcome], y_pred_rf, average='weighted')
        recall_rf = recall_score(y_test[outcome], y_pred_rf, average='weighted')
        auc_rf = roc_auc_score(y_test[outcome], rf_probs, average='weighted') # Only for PLOS DIED

        f1_gb = f1_score(y_test[outcome], y_pred_gb, average='weighted')
        precision_gb = precision_score(y_test[outcome], y_pred_gb, average='weighted')
        recall_gb = recall_score(y_test[outcome], y_pred_gb, average='weighted')
        auc_gb = roc_auc_score(y_test[outcome], gb_probs, average='weighted') # Only for PLOS DIED


        # Get feature importance
        rf_feature_importances = rf.feature_importances_
        gb_feature_importances = gb.feature_importances_


        # Filter features with importance > 0
        important_rf_features = [(selected_features_rf[i], rf_feature_importances[i])
                                 for i in range(len(rf_feature_importances)) if rf_feature_importances[i] > 0]
        important_gb_features = [(selected_features_gb[i], gb_feature_importances[i])
                                 for i in range(len(gb_feature_importances)) if gb_feature_importances[i] > 0]

        # Extract feature names and their importance
        rf_names = [f[0] for f in important_rf_features]
        rf_importances = [f[1] for f in important_rf_features]
        gb_names = [f[0] for f in important_gb_features]
        gb_importances = [f[1] for f in important_gb_features]


        # Store results in a DataFrame
        new_row = pd.DataFrame([{
            'Day': day,
            'Outcome': outcome,
            'Train_accuracy_rf': train_accuracy_rf,
            'CV_accuracy_rf': cv_mean_rf['accuracy'],  # Cross-validation mean accuracy
            'CV_F1_rf': cv_mean_rf['f1'],             # Cross-validation mean F1 score
            'CV_Recall_rf': cv_mean_rf['recall'],     # Cross-validation mean Recall
            'CV_Precision_rf': cv_mean_rf['precision'],  # Cross-validation mean Precision
            'CV_AUC_rf': cv_mean_rf['roc_auc'],
            'Accuracy_rf': test_accuracy_rf,
            'F1_rf': f1_rf,
            'Precision_rf': precision_rf,
            'Recall_rf': recall_rf,
            'AUC_rf': auc_rf,
            'Selected_feature_rf': rf_names,
            'Feature_importance_rf': rf_importances,

            'Train_accuracy_gb': train_accuracy_gb,
            'CV_accuracy_gb': cv_mean_gb['accuracy'],  # Cross-validation mean accuracy
            'CV_F1_gb': cv_mean_gb['f1'],             # Cross-validation mean F1 score
            'CV_Recall_gb': cv_mean_gb['recall'],     # Cross-validation mean Recall
            'CV_Precision_gb': cv_mean_gb['precision'],  # Cross-validation mean Precision
            'CV_AUC_gb': cv_mean_gb['roc_auc'],
            'Accuracy_gb': test_accuracy_gb,
            'F1_gb': f1_gb,
            'Precision_gb': precision_gb,
            'Recall_gb': recall_gb,
            'AUC_gb': auc_gb,
            'Selected_feature_gb': gb_names,
            'Feature_importance_gb': gb_importances
        }])


        # Append the results to the existing DataFrame
        outcome_df = pd.concat([outcome_df, new_row], ignore_index=True)
    filename = fr"C:\Users\Jiahui\PycharmProjects\NIS\Outcome\PLOSDIED\Prediction_summary_removeLOS_{day}.csv"
    outcome_df.to_csv(filename, index=False)

    # Save the DataFrame to a CSV file
    filename1 = fr"C:\Users\Jiahui\PycharmProjects\NIS\Outcome\PLOSDIED\Prediction_data_output_removeLOS_{day}.csv"
    y_test.to_csv(filename1, index=False)



