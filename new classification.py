import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE



df = pd.read_csv('HR_data.csv')

# Features and Target 
hr_features = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max', 'HR_AUC']
X = df[hr_features]
y_original = df['Frustrated']
groups = df['Individual']

#Binning 
y_binned = pd.cut(y_original, bins=[-0.5, 3.5, 10.5], labels=['Low', 'Elevated'], include_lowest=True)
label_encoder = LabelEncoder()
y = pd.Series(label_encoder.fit_transform(y_binned), index=X.index)

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=hr_features)

# Model
model_lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

models = {
    'Logistic Regression': model_lr,
    'Random Forest': model_rf
}

# Cross-Validation Setup 
n_splits = min(5, len(groups.unique()))
gkf = GroupKFold(n_splits=n_splits)
print(f"\nUsing GroupKFold with {n_splits} splits, grouping by '{groups.name}'.")

# Evaluate  Models 
for name, model in models.items():
    print(f"\n Evaluating {name} ")
    all_predictions = np.array([])
    all_true_labels = np.array([])
    fold_accuracies = []
    fold_f1s = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups)):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if len(X_test) == 0:
            print(f"  Fold {fold+1}: Skipping empty test fold.")
            continue
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='macro', zero_division=0)

        fold_accuracies.append(acc)
        fold_f1s.append(f1)
        all_predictions = np.append(all_predictions, predictions)
        all_true_labels = np.append(all_true_labels, y_test)

        print(f"  Fold {fold+1} Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")

    # Aggregate Scores
    print(f"\nAverage Accuracy for {name}: {np.mean(fold_accuracies):.4f}")
    print(f"Average Macro F1 Score for {name}: {np.mean(fold_f1s):.4f}")

    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_predictions, zero_division=0))

    #  Conf Matrix
    true_named = label_encoder.inverse_transform(all_true_labels.astype(int))
    pred_named = label_encoder.inverse_transform(all_predictions.astype(int))
    cm = confusion_matrix(true_named, pred_named, labels=['Low', 'Elevated'])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'Elevated'],
                yticklabels=['Low', 'Elevated'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {name}')
    plt.tight_layout()
    plt.show()

#class distribution 
print("\nFinal class distribution after binning:")
print(y_binned.value_counts())
