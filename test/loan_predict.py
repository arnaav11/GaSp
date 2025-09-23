import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
import sys
import joblib
import xgboost as xgb 
import random




MODEL_FILE = 'loan_status_model_fixed.joblib'

try:
    
    model = joblib.load(MODEL_FILE)
    print("Improved XGBoost model loaded successfully.")
except FileNotFoundError:
    print(f"Error: The model file '{MODEL_FILE}' was not found. Please ensure it is in the same directory as this script.")
    print("Have you run the 'train_model_fixed.py' script first?")
    sys.exit()




test_cases = [
    
    {
        'term': ' 36 months', 'home_ownership': 'MORTGAGE', 'fico_range_low': 780.0, 'total_acc': 35.0,
        'pub_rec': 0.0, 'revol_util': 15.0, 'annual_inc': 120000.0, 'int_rate': 7.5,
        'dti': 8.5, 'purpose': 'debt_consolidation', 'mort_acc': 3.0, 'loan_amnt': 25000.0,
        'application_type': 'Individual', 'installment': 779.6, 'verification_status': 'Verified',
        'pub_rec_bankruptcies': 0.0, 'addr_state': 'CA', 'initial_list_status': 'f',
        'fico_range_high': 784.0, 'revol_bal': 10000.0, 'open_acc': 15.0, 'emp_length': 10.0,
        'time_to_earliest_cr_line': 450000.0, 'expected': 'Positive'
    },
    {
        'term': ' 36 months', 'home_ownership': 'RENT', 'fico_range_low': 720.0, 'total_acc': 20.0,
        'pub_rec': 0.0, 'revol_util': 30.0, 'annual_inc': 85000.0, 'int_rate': 10.99,
        'dti': 15.0, 'purpose': 'credit_card', 'mort_acc': 0.0, 'loan_amnt': 15000.0,
        'application_type': 'Individual', 'installment': 490.5, 'verification_status': 'Verified',
        'pub_rec_bankruptcies': 0.0, 'addr_state': 'IL', 'initial_list_status': 'w',
        'fico_range_high': 724.0, 'revol_bal': 12000.0, 'open_acc': 10.0, 'emp_length': 7.0,
        'time_to_earliest_cr_line': 300000.0, 'expected': 'Positive'
    },
    {
        'term': ' 36 months', 'home_ownership': 'MORTGAGE', 'fico_range_low': 700.0, 'total_acc': 25.0,
        'pub_rec': 0.0, 'revol_util': 45.0, 'annual_inc': 95000.0, 'int_rate': 12.5,
        'dti': 20.0, 'purpose': 'home_improvement', 'mort_acc': 1.0, 'loan_amnt': 10000.0,
        'application_type': 'Individual', 'installment': 334.3, 'verification_status': 'Source Verified',
        'pub_rec_bankruptcies': 0.0, 'addr_state': 'TX', 'initial_list_status': 'f',
        'fico_range_high': 704.0, 'revol_bal': 9000.0, 'open_acc': 18.0, 'emp_length': 11.0,
        'time_to_earliest_cr_line': 600000.0, 'expected': 'Positive'
    },
    {
        'term': ' 36 months', 'home_ownership': 'OWN', 'fico_range_low': 730.0, 'total_acc': 15.0,
        'pub_rec': 0.0, 'revol_util': 25.0, 'annual_inc': 75000.0, 'int_rate': 9.5,
        'dti': 18.0, 'purpose': 'medical', 'mort_acc': 0.0, 'loan_amnt': 5000.0,
        'application_type': 'Individual', 'installment': 160.0, 'verification_status': 'Verified',
        'pub_rec_bankruptcies': 0.0, 'addr_state': 'NY', 'initial_list_status': 'w',
        'fico_range_high': 734.0, 'revol_bal': 5000.0, 'open_acc': 8.0, 'emp_length': 5.0,
        'time_to_earliest_cr_line': 250000.0, 'expected': 'Positive'
    },
    {
        'term': ' 60 months', 'home_ownership': 'MORTGAGE', 'fico_range_low': 740.0, 'total_acc': 40.0,
        'pub_rec': 0.0, 'revol_util': 40.0, 'annual_inc': 150000.0, 'int_rate': 11.0,
        'dti': 15.0, 'purpose': 'vacation', 'mort_acc': 4.0, 'loan_amnt': 30000.0,
        'application_type': 'Individual', 'installment': 652.8, 'verification_status': 'Verified',
        'pub_rec_bankruptcies': 0.0, 'addr_state': 'TX', 'initial_list_status': 'f',
        'fico_range_high': 744.0, 'revol_bal': 20000.0, 'open_acc': 18.0, 'emp_length': 10.0,
        'time_to_earliest_cr_line': 700000.0, 'expected': 'Positive'
    },
    {
        'term': ' 36 months', 'home_ownership': 'MORTGAGE', 'fico_range_low': 710.0, 'total_acc': 22.0,
        'pub_rec': 0.0, 'revol_util': 35.0, 'annual_inc': 80000.0, 'int_rate': 10.0,
        'dti': 17.0, 'purpose': 'debt_consolidation', 'mort_acc': 2.0, 'loan_amnt': 12000.0,
        'application_type': 'Individual', 'installment': 387.0, 'verification_status': 'Verified',
        'pub_rec_bankruptcies': 0.0, 'addr_state': 'CA', 'initial_list_status': 'w',
        'fico_range_high': 714.0, 'revol_bal': 11000.0, 'open_acc': 11.0, 'emp_length': 8.0,
        'time_to_earliest_cr_line': 500000.0, 'expected': 'Positive'
    },
    {
        'term': ' 36 months', 'home_ownership': 'MORTGAGE', 'fico_range_low': 750.0, 'total_acc': 30.0,
        'pub_rec': 0.0, 'revol_util': 20.0, 'annual_inc': 100000.0, 'int_rate': 8.5,
        'dti': 10.0, 'purpose': 'debt_consolidation', 'mort_acc': 2.0, 'loan_amnt': 20000.0,
        'application_type': 'Individual', 'installment': 628.9, 'verification_status': 'Verified',
        'pub_rec_bankruptcies': 0.0, 'addr_state': 'CA', 'initial_list_status': 'f',
        'fico_range_high': 754.0, 'revol_bal': 15000.0, 'open_acc': 12.0, 'emp_length': 10.0,
        'time_to_earliest_cr_line': 400000.0, 'expected': 'Positive'
    },
    {
        'term': ' 36 months', 'home_ownership': 'RENT', 'fico_range_low': 740.0, 'total_acc': 18.0,
        'pub_rec': 0.0, 'revol_util': 10.0, 'annual_inc': 90000.0, 'int_rate': 7.0,
        'dti': 12.0, 'purpose': 'medical', 'mort_acc': 0.0, 'loan_amnt': 8000.0,
        'application_type': 'Individual', 'installment': 247.9, 'verification_status': 'Not Verified',
        'pub_rec_bankruptcies': 0.0, 'addr_state': 'NY', 'initial_list_status': 'w',
        'fico_range_high': 744.0, 'revol_bal': 4000.0, 'open_acc': 9.0, 'emp_length': 5.0,
        'time_to_earliest_cr_line': 200000.0, 'expected': 'Positive'
    },
    {
        'term': ' 36 months', 'home_ownership': 'MORTGAGE', 'fico_range_low': 715.0, 'total_acc': 28.0,
        'pub_rec': 0.0, 'revol_util': 28.0, 'annual_inc': 105000.0, 'int_rate': 11.5,
        'dti': 16.0, 'purpose': 'car', 'mort_acc': 1.0, 'loan_amnt': 18000.0,
        'application_type': 'Individual', 'installment': 593.7, 'verification_status': 'Source Verified',
        'pub_rec_bankruptcies': 0.0, 'addr_state': 'TX', 'initial_list_status': 'f',
        'fico_range_high': 719.0, 'revol_bal': 15000.0, 'open_acc': 14.0, 'emp_length': 6.0,
        'time_to_earliest_cr_line': 350000.0, 'expected': 'Positive'
    },
    {
        'term': ' 60 months', 'home_ownership': 'MORTGAGE', 'fico_range_low': 760.0, 'total_acc': 50.0,
        'pub_rec': 0.0, 'revol_util': 30.0, 'annual_inc': 200000.0, 'int_rate': 10.0,
        'dti': 10.0, 'purpose': 'debt_consolidation', 'mort_acc': 5.0, 'loan_amnt': 35000.0,
        'application_type': 'Individual', 'installment': 743.6, 'verification_status': 'Verified',
        'pub_rec_bankruptcies': 0.0, 'addr_state': 'CA', 'initial_list_status': 'w',
        'fico_range_high': 764.0, 'revol_bal': 40000.0, 'open_acc': 25.0, 'emp_length': 11.0,
        'time_to_earliest_cr_line': 800000.0, 'expected': 'Positive'
    },
    
    {
        'term': ' 60 months', 'home_ownership': 'RENT', 'fico_range_low': 660.0, 'total_acc': 15.0,
        'pub_rec': 1.0, 'revol_util': 85.0, 'annual_inc': 45000.0, 'int_rate': 20.0,
        'dti': 25.0, 'purpose': 'debt_consolidation', 'mort_acc': 0.0, 'loan_amnt': 20000.0,
        'application_type': 'Individual', 'installment': 530.45, 'verification_status': 'Source Verified',
        'pub_rec_bankruptcies': 1.0, 'addr_state': 'NY', 'initial_list_status': 'w',
        'fico_range_high': 664.0, 'revol_bal': 18000.0, 'open_acc': 8.0, 'emp_length': 2.0,
        'time_to_earliest_cr_line': 150000.0, 'expected': 'Negative'
    },
    {
        'term': ' 60 months', 'home_ownership': 'RENT', 'fico_range_low': 670.0, 'total_acc': 10.0,
        'pub_rec': 0.0, 'revol_util': 75.0, 'annual_inc': 30000.0, 'int_rate': 24.0,
        'dti': 30.0, 'purpose': 'other', 'mort_acc': 0.0, 'loan_amnt': 10000.0,
        'application_type': 'Individual', 'installment': 287.5, 'verification_status': 'Not Verified',
        'pub_rec_bankruptcies': 0.0, 'addr_state': 'CA', 'initial_list_status': 'f',
        'fico_range_high': 674.0, 'revol_bal': 7000.0, 'open_acc': 5.0, 'emp_length': 1.0,
        'time_to_earliest_cr_line': 50000.0, 'expected': 'Negative'
    },
    {
        'term': ' 36 months', 'home_ownership': 'OWN', 'fico_range_low': 680.0, 'total_acc': 8.0,
        'pub_rec': 1.0, 'revol_util': 95.0, 'annual_inc': 50000.0, 'int_rate': 21.0,
        'dti': 28.0, 'purpose': 'small_business', 'mort_acc': 0.0, 'loan_amnt': 8000.0,
        'application_type': 'Individual', 'installment': 302.2, 'verification_status': 'Verified',
        'pub_rec_bankruptcies': 1.0, 'addr_state': 'GA', 'initial_list_status': 'w',
        'fico_range_high': 684.0, 'revol_bal': 11000.0, 'open_acc': 4.0, 'emp_length': 0.0,
        'time_to_earliest_cr_line': 100000.0, 'expected': 'Negative'
    },
    {
        'term': ' 60 months', 'home_ownership': 'MORTGAGE', 'fico_range_low': 670.0, 'total_acc': 20.0,
        'pub_rec': 1.0, 'revol_util': 99.0, 'annual_inc': 70000.0, 'int_rate': 18.0,
        'dti': 30.0, 'purpose': 'debt_consolidation', 'mort_acc': 2.0, 'loan_amnt': 30000.0,
        'application_type': 'Individual', 'installment': 750.0, 'verification_status': 'Verified',
        'pub_rec_bankruptcies': 1.0, 'addr_state': 'FL', 'initial_list_status': 'f',
        'fico_range_high': 674.0, 'revol_bal': 25000.0, 'open_acc': 10.0, 'emp_length': 3.0,
        'time_to_earliest_cr_line': 250000.0, 'expected': 'Negative'
    },
    {
        'term': ' 60 months', 'home_ownership': 'RENT', 'fico_range_low': 660.0, 'total_acc': 12.0,
        'pub_rec': 0.0, 'revol_util': 88.0, 'annual_inc': 40000.0, 'int_rate': 25.0,
        'dti': 28.0, 'purpose': 'debt_consolidation', 'mort_acc': 0.0, 'loan_amnt': 15000.0,
        'application_type': 'Individual', 'installment': 420.0, 'verification_status': 'Not Verified',
        'pub_rec_bankruptcies': 0.0, 'addr_state': 'TX', 'initial_list_status': 'w',
        'fico_range_high': 664.0, 'revol_bal': 12000.0, 'open_acc': 7.0, 'emp_length': 1.0,
        'time_to_earliest_cr_line': 100000.0, 'expected': 'Negative'
    },
    {
        'term': ' 36 months', 'home_ownership': 'MORTGAGE', 'fico_range_low': 680.0, 'total_acc': 18.0,
        'pub_rec': 1.0, 'revol_util': 80.0, 'annual_inc': 55000.0, 'int_rate': 18.5,
        'dti': 27.0, 'purpose': 'other', 'mort_acc': 1.0, 'loan_amnt': 12000.0,
        'application_type': 'Individual', 'installment': 437.0, 'verification_status': 'Source Verified',
        'pub_rec_bankruptcies': 1.0, 'addr_state': 'NY', 'initial_list_status': 'f',
        'fico_range_high': 684.0, 'revol_bal': 10000.0, 'open_acc': 9.0, 'emp_length': 4.0,
        'time_to_earliest_cr_line': 300000.0, 'expected': 'Negative'
    },
    {
        'term': ' 36 months', 'home_ownership': 'RENT', 'fico_range_low': 660.0, 'total_acc': 10.0,
        'pub_rec': 2.0, 'revol_util': 95.0, 'annual_inc': 35000.0, 'int_rate': 23.0,
        'dti': 33.0, 'purpose': 'medical', 'mort_acc': 0.0, 'loan_amnt': 7000.0,
        'application_type': 'Individual', 'installment': 274.5, 'verification_status': 'Not Verified',
        'pub_rec_bankruptcies': 2.0, 'addr_state': 'GA', 'initial_list_status': 'w',
        'fico_range_high': 664.0, 'revol_bal': 6000.0, 'open_acc': 5.0, 'emp_length': 0.0,
        'time_to_earliest_cr_line': 40000.0, 'expected': 'Negative'
    },
    {
        'term': ' 60 months', 'home_ownership': 'MORTGAGE', 'fico_range_low': 690.0, 'total_acc': 25.0,
        'pub_rec': 0.0, 'revol_util': 80.0, 'annual_inc': 80000.0, 'int_rate': 19.5,
        'dti': 31.0, 'purpose': 'credit_card', 'mort_acc': 2.0, 'loan_amnt': 25000.0,
        'application_type': 'Individual', 'installment': 655.0, 'verification_status': 'Verified',
        'pub_rec_bankruptcies': 0.0, 'addr_state': 'CA', 'initial_list_status': 'f',
        'fico_range_high': 694.0, 'revol_bal': 20000.0, 'open_acc': 12.0, 'emp_length': 6.0,
        'time_to_earliest_cr_line': 450000.0, 'expected': 'Negative'
    },
    {
        'term': ' 36 months', 'home_ownership': 'RENT', 'fico_range_low': 670.0, 'total_acc': 15.0,
        'pub_rec': 1.0, 'revol_util': 92.0, 'annual_inc': 42000.0, 'int_rate': 21.5,
        'dti': 26.0, 'purpose': 'debt_consolidation', 'mort_acc': 0.0, 'loan_amnt': 18000.0,
        'application_type': 'Individual', 'installment': 670.0, 'verification_status': 'Verified',
        'pub_rec_bankruptcies': 1.0, 'addr_state': 'IL', 'initial_list_status': 'w',
        'fico_range_high': 674.0, 'revol_bal': 15000.0, 'open_acc': 8.0, 'emp_length': 2.0,
        'time_to_earliest_cr_line': 80000.0, 'expected': 'Negative'
    },
    {
        'term': ' 60 months', 'home_ownership': 'OWN', 'fico_range_low': 660.0, 'total_acc': 10.0,
        'pub_rec': 1.0, 'revol_util': 85.0, 'annual_inc': 50000.0, 'int_rate': 23.0,
        'dti': 30.0, 'purpose': 'other', 'mort_acc': 1.0, 'loan_amnt': 12000.0,
        'application_type': 'Individual', 'installment': 320.0, 'verification_status': 'Source Verified',
        'pub_rec_bankruptcies': 1.0, 'addr_state': 'NY', 'initial_list_status': 'f',
        'fico_range_high': 664.0, 'revol_bal': 10000.0, 'open_acc': 6.0, 'emp_length': 1.0,
        'time_to_earliest_cr_line': 70000.0, 'expected': 'Negative'
    }
]


random.shuffle(test_cases)



print("\n--- Making predictions on randomized test cases ---")

for i, loan_data_dict in enumerate(test_cases, 1):
    try:
        
        expected_outcome = loan_data_dict.pop('expected')
        
        
        new_data_df = pd.DataFrame([loan_data_dict])
        
        print(f"\n--- Test Case {i} ---")
        print("Input Data:")
        
        print(new_data_df.to_string(index=False))
        
        
        prediction = model.predict(new_data_df)
        
        
        predicted_outcome = 'Positive' if prediction[0] == 1 else 'Negative'
        
        
        print(f"\nPrediction for this loan: {predicted_outcome}")
        print(f"Expected outcome: {expected_outcome}")
        
    except Exception as e:
        print(f"An error occurred for Test Case {i}: {e}")
        
print("\nBatch prediction complete.")