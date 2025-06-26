import pandas as pd
from tabulate import tabulate
from scipy.stats import ttest_rel

# Random Forest Cross-Validation Results(i guess i shouldve saved the data as a csv file, mb)
rf_cv = pd.DataFrame({
    'Fold': [1, 2, 3, 4, 5],
    'Accuracy': [0.6111, 0.6111, 0.6944, 0.5833, 0.6250],
    'Macro F1': [0.4375, 0.4375, 0.5765, 0.4225, 0.3846]
})
rf_cv.loc['Average'] = ['Average', 0.6250, 0.4517]

print("Random Forest Cross-Validation Results")
print(tabulate(rf_cv, headers='keys', tablefmt='github'))
print()

# Random Forest Classification Report
rf_report = pd.DataFrame({
    'Precision': [0.21, 0.71, '', 0.46, 0.57],
    'Recall': [0.13, 0.82, '', 0.47, 0.62],
    'F1-score': [0.16, 0.76, 0.62, 0.46, 0.59],
    'Support': [47, 121, 168, 168, 168]
}, index=['0.0', '1.0', 'Accuracy', 'Macro avg', 'Weighted avg'])

print("Random Forest Classification Report")
print(tabulate(rf_report, headers='keys', tablefmt='github'))
print()

# Logistic Regression Cross-Validation Results
lr_cv = pd.DataFrame({
    'Fold': [1, 2, 3, 4, 5],
    'Accuracy': [0.4444, 0.4722, 0.4167, 0.5556, 0.3750],
    'Macro F1': [0.4444, 0.4370, 0.4126, 0.5000, 0.3169]
})
lr_cv.loc['Average'] = ['Average', 0.4528, 0.4222]

print("Logistic Regression Evaluation Metrics")
print(tabulate(lr_cv, headers='keys', tablefmt='github'))
print()

# Logistic Regression Classification Report
lr_report = pd.DataFrame({
    'Precision': [0.26, 0.69, '', 0.47, 0.57],
    'Recall': [0.49, 0.45, '', 0.47, 0.46],
    'F1-score': [0.34, 0.54, 0.46, 0.44, 0.48],
    'Support': [47, 121, 168, 168, 168]
}, index=['0.0', '1.0', 'Accuracy', 'Macro avg', 'Weighted avg'])

print(tabulate(lr_report, headers='keys', tablefmt='github'))


# Compare average accuracy and macro F1 between models
rf_avg_acc = rf_cv.loc['Average', 'Accuracy']
lr_avg_acc = lr_cv.loc['Average', 'Accuracy']
rf_avg_f1 = rf_cv.loc['Average', 'Macro F1']
lr_avg_f1 = lr_cv.loc['Average', 'Macro F1']

print("\nModel Comparison:")
print(f"Random Forest - Average Accuracy: {rf_avg_acc:.4f}, Average Macro F1: {rf_avg_f1:.4f}")
print(f"Logistic Regression - Average Accuracy: {lr_avg_acc:.4f}, Average Macro F1: {lr_avg_f1:.4f}")

# Paired t-test for cross-validation accuracy and macro F1

acc_ttest = ttest_rel(rf_cv.iloc[:5]['Accuracy'], lr_cv.iloc[:5]['Accuracy'])
f1_ttest = ttest_rel(rf_cv.iloc[:5]['Macro F1'], lr_cv.iloc[:5]['Macro F1'])

print("\nPaired t-test results (Random Forest vs Logistic Regression):")
print(f"Accuracy: t-statistic = {acc_ttest.statistic:.4f}, p-value = {acc_ttest.pvalue:.4f}")
print(f"Macro F1: t-statistic = {f1_ttest.statistic:.4f}, p-value = {f1_ttest.pvalue:.4f}")
# Paired t-tests for classification report metrics (Precision, Recall, F1-score)


metrics = ['Precision', 'Recall', 'F1-score']
print("\nPaired t-test results for classification report metrics (Random Forest vs Logistic Regression):")


for metric in metrics:
    # Only compare for classes '0.0' and '1.0' ( elevated(0) and low(1))
    rf_vals = pd.to_numeric(rf_report.loc[['0.0', '1.0'], metric], errors='coerce')
    lr_vals = pd.to_numeric(lr_report.loc[['0.0', '1.0'], metric], errors='coerce')
    # Skip if any value is missing
    if rf_vals.isnull().any() or lr_vals.isnull().any():
        print(f"{metric}: Not enough data for t-test.")
        continue
    ttest = ttest_rel(rf_vals, lr_vals)
    print(f"{metric}: t-statistic = {ttest.statistic:.4f}, p-value = {ttest.pvalue:.4f}")