# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import cross_val_score


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
NIS_All_Clean_Merge_final = pd.concat([NIS_All_Clean_Merge.drop(columns=categorical_columns), encoded_df_cleaned_1],
                                      axis=1)

### Diagnosis and procedure codes
NIS_ALL_DX = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_ALL_DX_list.csv", dtype=dtypes)
NIS_ALL_PR = pd.read_csv(r"C:\Users\Jiahui\PycharmProjects\NIS\NIS_ALL_PR_list.csv", dtype=dtypes)
DX_list = NIS_ALL_DX['ICD'].tolist()
PR_list = NIS_ALL_PR['ICD'].tolist()

cmr_columns = ["CMR_AIDS","CMR_ALCOHOL","CMR_DEMENTIA","CMR_DEPRESS","CMR_DIAB_CX",
               "CMR_DIAB_UNCX","CMR_DRUG_ABUSE","CMR_HTN_CX","CMR_HTN_UNCX","CMR_LUNG_CHRONIC","CMR_OBESE","CMR_PERIVASC",
               "CMR_THYROID_HYPO","CMR_THYROID_OTH"]
binary_columns = ['KID','C9100','C9101','C9102','Z006'] #, 'Area'

# Define feature sets and outcome columns
Response_column = ['PLOS','DIED']
Total_x_list = ['AGE'] + [col for col in encoded_df_cleaned_1.columns] + DX_list + PR_list + cmr_columns + binary_columns
Total_list = ['AGE'] + [col for col in encoded_df_cleaned_1.columns] + DX_list + PR_list + cmr_columns + binary_columns + Response_column + ['TOTCHG']
PRDay_list =  sorted([day for day in NIS_All_Clean_Merge['PRDay_model'].unique() if day != 0])

# Loop through outcomes and days to generate binary predictions
outcome_df = pd.DataFrame()

for day in PRDay_list:
    print(f"Prediction on day: {day}")

    # Prepare data
    model_data = NIS_All_Clean_Merge_final[Total_list][NIS_All_Clean_Merge_final['PRDay_model'] <= day]
    model_data = model_data.dropna(subset=['PLOS'])
    model_data = model_data.dropna(subset=['DIED'])
    model_data = model_data[model_data['PLOS'] != '']
    model_data = model_data[model_data['DIED'] != '']

    # Define the training set
    X = model_data[Total_x_list]

    # Scale continuous features
    scaler = StandardScaler()
    X['AGE'] = scaler.fit_transform(X[['AGE']])
    X['AGE'] = scaler.transform(X[['AGE']])


    #### Predcit the best PLOS
    y_PLOS = model_data['PLOS'].astype(int)

    # Initialize models and RFECV for feature selection
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rfecv_rf = RFECV(estimator=rf, step=20, cv=5, scoring='accuracy')

    # Fit RFECV and get selected features
    rfecv_rf.fit(X, y_PLOS)
    selected_features_rf = X.columns[rfecv_rf.support_]

    # Subset data for selected features
    X_selected_PLOS = X[selected_features_rf]

    # Fit models to resampled training data
    rf.fit(X_selected_PLOS, y_PLOS)

    # Make predictions on the test set
    y_pred_PLOS = rf.predict(X_selected_PLOS)

    # Add the prediction value
    model_data["Pred_PLOS"] = y_pred_PLOS

    #### Predcit the best Mortality
    y_DIED = model_data['DIED'].astype(int)

    # Initialize models and RFECV for feature selection
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    rfecv_gb = RFECV(estimator=gb, step=20, cv=5, scoring='accuracy')

    # Fit RFECV and get selected features
    rfecv_gb.fit(X, y_DIED)
    selected_features_gb = X.columns[rfecv_gb.support_]

    # Subset data for selected features
    X_selected_DIED = X[selected_features_gb]

    # Fit models to resampled training data
    gb.fit(X_selected_DIED, y_DIED)

    # Make predictions on the test set
    y_pred_DIED = gb.predict(X_selected_DIED)

    # Add the prediction value
    model_data["Pred_DIED"] = y_pred_DIED

    #### Predict the Total Charges
    model_data_TC = model_data.copy()
    model_data_TC = model_data_TC.dropna(subset=['TOTCHG'])
    model_data_TC = model_data_TC[model_data_TC['TOTCHG'] != '']

    #model_data_TC = model_data_TC.dropna(subset=['TOTCHG'])
    print(model_data_TC.columns.tolist())

    Pred_list = ['Pred_DIED','Pred_PLOS']
    # Combine PLOS and DIED into single categorical outcome
    def map_PLOSD(value):
        return {"00": 0, "10": 1, "01": 2, "11": 3}.get(value, "")

    model_data_TC['Pred_PLOSDc'] = model_data_TC['Pred_PLOS'].astype(str) + model_data_TC['Pred_DIED'].astype(str)
    model_data_TC['Pred_PLOSD'] = model_data_TC['Pred_PLOSDc'].apply(map_PLOSD)

    model_data_TC['Pred_PLOS_only'] = model_data_TC['Pred_PLOSD'].apply(lambda x: 1 if x == 1  else 0)
    model_data_TC['Pred_Mortaltiy_only'] = model_data_TC['Pred_PLOSD'].apply(lambda x: 1 if x == 2  else 0)
    model_data_TC['Pred_PLOSM'] = model_data_TC['Pred_PLOSD'].apply(lambda x: 1 if x == 3  else 0)


    X_TC = model_data_TC[Total_x_list + ['Pred_PLOS_only', 'Pred_Mortaltiy_only', 'Pred_PLOSM']]
    y_TC = model_data_TC['TOTCHG'].astype(int)

    # Train-test split for regression
    X_train_TC, X_test_TC, y_train_TC, y_test_TC = train_test_split(X_TC, y_TC, test_size=0.2, random_state=42)

    # Train regression model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rfecv_rf = RFECV(estimator=rf, step=20, cv=5, scoring='neg_mean_squared_error')

    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    rfecv_gb = RFECV(estimator=gb, step=20, cv=5, scoring='neg_mean_squared_error')

    # Fit RFECV and get selected features
    rfecv_rf.fit(X_train_TC, y_train_TC)
    rfecv_gb.fit(X_train_TC, y_train_TC)

    selected_features_rf = X_train_TC.columns[rfecv_rf.support_]
    selected_features_gb = X_train_TC.columns[rfecv_gb.support_]

    # Subset data for selected features
    X_train_selected_rf = X_train_TC[selected_features_rf]
    X_test_selected_rf = X_test_TC[selected_features_rf]
    X_train_selected_gb = X_train_TC[selected_features_gb]
    X_test_selected_gb = X_test_TC[selected_features_gb]

    # Define Custom Scorers
    scorers = {
        'MAE': make_scorer(mean_absolute_error),
        'MSE': make_scorer(mean_squared_error),
        'R2': make_scorer(r2_score)
    }

    # Perform Cross-Validation
    cv_results_rf = cross_validate(rf, X_train_selected_rf, y_train_TC, cv=5, scoring=scorers, return_train_score=False)
    cv_results_gb = cross_validate(gb, X_train_selected_gb, y_train_TC, cv=5, scoring=scorers, return_train_score=False)

    # Fit models and predict
    rf.fit(X_train_selected_rf, y_train_TC)
    gb.fit(X_train_selected_gb, y_train_TC)

    # Make predictions on the test set
    y_pred_rf = rf.predict(X_test_selected_rf)
    y_pred_gb = gb.predict(X_test_selected_gb)

    # Predict and evaluate
    # Calculate evaluation metrics
    mse_rf = mean_squared_error(y_test_TC, y_pred_rf)
    mae_rf = mean_absolute_error(y_test_TC, y_pred_rf)
    r2_rf = r2_score(y_test_TC, y_pred_rf)

    mse_gb = mean_squared_error(y_test_TC, y_pred_gb)
    mae_gb = mean_absolute_error(y_test_TC, y_pred_gb)
    r2_gb = r2_score(y_test_TC, y_pred_gb)

    # Add predictions as new columns
    y_test_TC['RF_Pred'] = y_pred_rf
    y_test_TC['GB_Pred'] = y_pred_gb

    # Add the prediction value
    # Create dynamic column names
    rf_col_name = f"RF_Prediction"
    gb_col_name = f"GB_Prediction"

    # Add predictions as new columns
    y_test_TC[rf_col_name] = y_pred_rf
    y_test_TC[gb_col_name] = y_pred_gb

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
        'MSE_rf_cv': np.mean(cv_results_rf['test_MSE']),
        'MAE_rf_cv': np.mean(cv_results_rf['test_MAE']),
        'R2_rf_cv': np.mean(cv_results_rf['test_R2']),
        'MSE_rf': mse_rf,
        'MAE_rf': mae_rf,
        'R2_rf': r2_rf,
        'Selected_feature_rf': rf_names,
        'Feature_importance_rf': rf_importances,
        'MSE_gb_cv': np.mean(cv_results_gb['test_MSE']),
        'MAE_gb_cv': np.mean(cv_results_gb['test_MAE']),
        'R2_gb_cv': np.mean(cv_results_gb['test_R2']),
        'MSE_gb': mse_gb,
        'MAE_gb': mae_gb,
        'R2_gb': r2_gb,
        'Selected_feature_gb': gb_names,
        'Feature_importance_gb': gb_importances
        }])

    # Append the results to the existing DataFrame
    outcome_df = pd.concat([outcome_df, new_row], ignore_index=True)
    filename = fr"C:\Users\Jiahui\PycharmProjects\NIS\TCprediction_Best_Ind\Prediction_summary_{day}.csv"
    outcome_df.to_csv(filename, index=False)

    # Save the DataFrame to a CSV file
    filename1 = fr"C:\Users\Jiahui\PycharmProjects\NIS\TCprediction_Best_Ind\Prediction_data_output_{day}.csv"
    y_test_TC.to_csv(filename1, index=False)


