import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, 
                           confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel


from classes import create_classes, FrustrationClassifier, FrustrationDataset, train_classifier, evaluate_classifier

torch.manual_seed(42)
np.random.seed(42)

# Load  data
data = pd.read_csv('HR_data.csv')
data['Frustration_Class'] = data['Frustrated'].apply(create_classes)

# Feature engineering
features = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max', 'HR_AUC']
for phase in [1, 2, 3]:
    data[f'HR_Mean_phase{phase}'] = data['HR_Mean'] * (data['Phase'] == phase)
data['HR_Mean_puzzler'] = data['HR_Mean'] * data['Puzzler']
data['HR_Mean_instructor'] = data['HR_Mean'] * (1 - data['Puzzler'])
data['HR_Mean_diff'] = data.groupby('Individual')['HR_Mean'].diff().fillna(0)
features.extend([
    'HR_Mean_phase1', 'HR_Mean_phase2', 'HR_Mean_phase3',
    'HR_Mean_puzzler', 'HR_Mean_instructor', 'HR_Mean_diff'
])

#  arrays
X = data[features].values
y = data['Frustration_Class'].values
groups = data['Individual'].values

# Encode labels and standardize
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # 0=Low, 1=Medium, 2=High
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


logo = LeaveOneGroupOut()
results = {
    'RandomForest': {'Accuracy': [], 'F1': [], 'Report': []},
    'NeuralNetwork': {'Accuracy': [], 'F1': [], 'Report': []}
}
conf_matrices = []

#  LOIO loop
for train_idx, test_idx in logo.split(X_scaled, y_encoded, groups):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    
    #
    present_classes = np.unique(np.concatenate([y_train, y_test]))
    present_class_names = [label_encoder.classes_[i] for i in present_classes]
    
    #Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    # Store RF results with dynamic class handling
    results['RandomForest']['Accuracy'].append(accuracy_score(y_test, y_pred_rf))
    results['RandomForest']['F1'].append(f1_score(y_test, y_pred_rf, average='weighted'))
    results['RandomForest']['Report'].append(
        classification_report(
            y_test, y_pred_rf,
            labels=present_classes,
            target_names=present_class_names,
            output_dict=True
        )
    )
    
    #  Neural Network
    train_dataset = FrustrationDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    model = FrustrationClassifier(input_size=X_train.shape[1], 
                                num_classes=len(label_encoder.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model = train_classifier(model, train_loader, criterion, optimizer, epochs=100)
    
    # Evaluate NN 
    eval_results = evaluate_classifier(
        model, 
        X_test, 
        y_test,
        class_names=present_class_names,
        labels=present_classes
    )
    
    # Store NN 
    results['NeuralNetwork']['Accuracy'].append(eval_results['accuracy'])
    results['NeuralNetwork']['F1'].append(eval_results['f1'])
    results['NeuralNetwork']['Report'].append(eval_results['report'])
    
    # Store confusion matrices 
    if len(conf_matrices) < 1:
        conf_matrices.append({
            'RF': confusion_matrix(y_test, y_pred_rf, labels=present_classes),
            'NN': confusion_matrix(y_test, eval_results['predictions'], labels=present_classes),
            'individual': groups[test_idx][0],
            'classes': present_class_names
        })

# Results Analysis 
print("\n Average Performance Across Folds") 
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"Accuracy: {np.mean(metrics['Accuracy']):.3f} ± {np.std(metrics['Accuracy']):.3f}")
    print(f"F1 Score: {np.mean(metrics['F1']):.3f} ± {np.std(metrics['F1']):.3f}")
    
    print("\nLast Fold Classification Report:")
    last_report = metrics['Report'][-1]
    # Convert the dictionary report to a string format
    report_str = ""
    for class_name in label_encoder.classes_:
        report_str += f"{class_name}\n"
        report_str += f"  Precision: {last_report[class_name]['precision']:.2f}\n"
        report_str += f"  Recall:    {last_report[class_name]['recall']:.2f}\n"
        report_str += f"  F1-score:  {last_report[class_name]['f1-score']:.2f}\n"
        report_str += f"  Support:   {last_report[class_name]['support']}\n\n"
    report_str += f"Accuracy: {last_report['accuracy']:.2f}\n"
    report_str += f"Macro avg: {last_report['macro avg']['f1-score']:.2f}\n"
    report_str += f"Weighted avg: {last_report['weighted avg']['f1-score']:.2f}"
    print(report_str)

# Statistical comparison
print("\n Model Comparison") 
for metric in ['Accuracy', 'F1']:
    rf_scores = results['RandomForest'][metric]
    nn_scores = results['NeuralNetwork'][metric]
    
    t_stat, p_val = ttest_rel(rf_scores, nn_scores)
    print(f"\n{metric} Comparison:")
    print(f"Random Forest: {np.mean(rf_scores):.3f} ± {np.std(rf_scores):.3f}")
    print(f"Neural Network: {np.mean(nn_scores):.3f} ± {np.std(nn_scores):.3f}")
    print(f"p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("Significant difference (p < 0.05)")
    else:
        print("No significant difference")

# Visualization 
# Confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for i, (model, matrix) in enumerate(conf_matrices[0].items()):
    if model in ['RF', 'NN']:
        sns.heatmap(
            matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=ax[i]
        )
        model_name = 'Random Forest' if model == 'RF' else 'Neural Network'
        ax[i].set_title(f'{model_name} (Individual {conf_matrices[0]["individual"]})')
        ax[i].set_xlabel('Predicted')
        ax[i].set_ylabel('Actual')
plt.tight_layout()
plt.show()

# Feature importance
rf_final = RandomForestClassifier().fit(X_scaled, y_encoded)
importances = pd.DataFrame({
    'Feature': features,
    'Importance': rf_final.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(importances['Feature'], importances['Importance'])
plt.title('Feature Importance (Random Forest)', fontsize=14)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()