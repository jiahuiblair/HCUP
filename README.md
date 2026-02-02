# HCUP
Project background: This project develops a dynamic machine learning framework to predict inpatient outcomes for acute lymphoblastic leukemia (ALL) using HCUP National Inpatient Sample data. Unlike prior approaches that rely on static inputs and independent outcome modeling, the framework concurrently predicts prolonged length of stay and in-hospital mortality, while continuously estimating total charges using temporally updated patient information. The concurrent model achieved strong early-hospitalization performance, with accuracy and precision above 90% and F1-scores and recall exceeding 80% within the first seven days. Key predictors included patient demographics, insurance and financial indicators, surgical utilization, and diagnostic codes. This work demonstrates how dynamic, multi-outcome prediction can support timely clinical decision-making, improve care efficiency, and reduce financial burden in ALL inpatient management.

HCUP database: The project utilizes data from the Healthcare Cost and Utilization Project (HCUP), developed by the Agency for Healthcare Research and Quality (AHRQ). Specifically, it leverages the National Inpatient Sample (NIS), the largest all-payer inpatient healthcare database in the United States.

The NIS dataset is comprised of four primary discharge-level files, providing a comprehensive view of the inpatient experience:
   Core File: Contains demographics, expected primary payer, total charges, discharge status, and ICD-10 coding for diagnoses and procedures.
   Severity File: Provides illness severity and mortality risk records utilizing the 3M™ APR-DRG (All-Patient Refined Diagnosis Related Group) system.
   Hospital File: Details facility characteristics, including location, ownership, and bed size.
   Diagnosis and Procedure Groups File: Defines patient comorbidities at admission using the Elixhauser Comorbidity Software Refined for ICD-10-CM.

Patient outcome: 
Prolonged Length of Stay (PLOS): 75% of the length of stay was categorized as prolonged length of stay (1/0)
Mortality (1/0)
Total charges ($)

Model: Random Forest and Gradient Boost

Prediction process:
The continuous patient outcome prediction was defined as using the predicted PLOS and Mortality outcome (both combined outcome and individual outcome) as input to predict the total charges. 
The dynamic prediction process predicts the patient's outcome per day using available procedure information for the first 7 days of hospitalization.

Workflow
1. Data cleaning (ALL_1921_datacleaning.py) →
2. Select and convert ICD codes into binary version (ALL_1921_ICD.py) →
3. Predict the PLOS and mortality outcomes (both combined and individual outcomes) with feature engineering, dataset balancing technique, train/test models (ALL_1921_3_PD.py and ALL_1921_3_PLOSD) →
4. Predict the total charges with predicted outcomes from PLOS and mortality (ALL_1921_3_TC_Best_Ind.py and ALL_1921_3_TC_comb.py) →
5. Evaluate and visualize (Evaluation_AUC.py, Evaluation_AUC_PLOSD.py, and Evaluation_TC)  →
6. Feature visualization (Feature_output_analysis_PD.py and Feature_output_analysis_TC)








   
