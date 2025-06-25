import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

from classes import FrustrationDataset, FrustrationPredictor, train_model, evaluate_model


torch.manual_seed(42)
np.random.seed(42)


data = pd.read_csv('HR_data.csv')
features = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max', 'HR_AUC']
X = data[features].values
y = data['Frustrated'].values
groups = data['Individual'].values

# Standardize featres
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PyTorch tensors
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y).unsqueeze(1)  


################### Training and evaluation setup ###################
# Leave-One-Group-Out cross-validation

logo = LeaveOneGroupOut()


results = {
    'Random Forest': {'MAE': [], 'RMSE': [], 'R2': []},
    'Neural Network': {'MAE': [], 'RMSE': [], 'R2': []}
}

####################################
# 1. Phase features
for phase in [1, 2, 3]:
    data[f'HR_Mean_phase{phase}'] = data['HR_Mean'] * (data['Phase'] == phase)
    
# 2. Role features
data['HR_Mean_puzzler'] = data['HR_Mean'] * data['Puzzler']
data['HR_Mean_instructor'] = data['HR_Mean'] * (1 - data['Puzzler'])

# 3. features (changes between rounds)
data['HR_Mean_diff'] = data.groupby('Individual')['HR_Mean'].diff().fillna(0)

# Update features list
features.extend([
    'HR_Mean_phase1', 'HR_Mean_phase2', 'HR_Mean_phase3',
    'HR_Mean_puzzler', 'HR_Mean_instructor',
    'HR_Mean_diff'
])
###################################



logo = LeaveOneGroupOut()


print("\n Validating Group Separation")
for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_scaled, y, groups)):
    train_individuals = groups[train_idx]
    test_individual = groups[test_idx][0]
    
    # Check 1: Test individual not in training
    if test_individual in train_individuals:
        raise ValueError(f"Data leakage in fold {fold_idx}: Individual {test_individual} in both train/test")
    
    # Check 2: All test samples belong to the test individual
    n_test_samples = len(test_idx)
    n_correct_samples = sum(groups[test_idx] == test_individual)
    
    if n_test_samples != n_correct_samples:
        raise ValueError(
            f"Fold {fold_idx}: Test set contains {n_test_samples - n_correct_samples} "
            f"samples not from test individual {test_individual}"
        )
    
    print(f"Fold {fold_idx+1}: Test individual {test_individual} - {len(test_idx)} samples - OK")

print("All folds validated - no leakage detected\n")




all_predictions = []

for train_idx, test_idx in logo.split(X_scaled, y, groups):
    # Split data
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    ### lin reg

    rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=7,
    min_samples_split=5,
    random_state=42
)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)


    results['Random Forest']['MAE'].append(mean_absolute_error(y_test, y_pred_rf))
    results['Random Forest']['RMSE'].append(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
    results['Random Forest']['R2'].append(r2_score(y_test, y_pred_rf))

    ### neural network
    # Create datasets
    train_dataset = FrustrationDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Initialize 
    model = FrustrationPredictor(input_size=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    model = train_model(model, train_loader, criterion, optimizer, epochs=100)
    
    # Evaluate
    mae, rmse, r2, y_pred_nn = evaluate_model(model, X_test, y_test)
    
    results['Neural Network']['MAE'].append(mae) #Mean Absolute Error
    results['Neural Network']['RMSE'].append(rmse) # Root Mean Squared Error
    results['Neural Network']['R2'].append(r2) # Coefficient of Determination
    
    # Store predictions for this fold
    all_predictions.append({
    'individual': groups[test_idx][0],
    'y_true': y_test,
    'y_pred_rf': y_pred_rf,  # Changed from y_pred_svr
    'y_pred_nn': y_pred_nn
    })


################### RESULTS ####################
print(" Average Performance ")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric_name, values in metrics.items():
        print(f"{metric_name}: {np.mean(values):.3f} ± {np.std(values):.3f}")

# compare

# After all folds are complete
importances = rf.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance Score')
plt.show()

print("\n Model Comparison ")
for metric in ['MAE', 'RMSE', 'R2']:
    nn_scores = results['Neural Network'][metric]
    rf_scores = results['Random Forest'][metric]

    t_stat, p_val = ttest_rel(rf_scores, nn_scores)
    print(f"\n{metric}:")
    print(f"Neural Network: {np.mean(nn_scores):.3f} ± {np.std(nn_scores):.3f}")
    print(f"Random Forest: {np.mean(rf_scores):.3f} ± {np.std(rf_scores):.3f}")
    print(f"p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("Difference is statistically significant!")
    else:
        print("No significant difference found.")

# Plot model comparison
metrics = ['MAE', 'RMSE', 'R2']
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
nn_means = [np.mean(results['Neural Network'][m]) for m in metrics]
rf_means = [np.mean(results['Random Forest'][m]) for m in metrics]

rects1 = ax.bar(x + width/2, rf_means, width, label='Random Forest')
rects2 = ax.bar(x - width/2, nn_means, width, label='Neural Network')


ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

fig.tight_layout()
plt.show()

# Plot predictions vs true values for last fold
plt.figure(figsize=(10, 5))
plt.scatter(all_predictions[-1]['y_true'], all_predictions[-1]['y_pred_nn'], label='Neural Network')
plt.scatter(all_predictions[-1]['y_true'], all_predictions[-1]['y_pred_rf'], label='Random Forest')
plt.plot([min(y), max(y)], [min(y), max(y)], 'k--', label='Perfect Prediction')
plt.xlabel('True Frustration Level')
plt.ylabel('Predicted Frustration Level')
plt.title('Prediction vs True Values (Last Individual)')
plt.legend()
plt.show()