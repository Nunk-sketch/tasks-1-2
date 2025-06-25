import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, 
                           classification_report, confusion_matrix,
                           mean_absolute_error, mean_squared_error, r2_score)

class FrustrationPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, criterion, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(X_test)
        predictions = model(test_tensor).squeeze().numpy()  
        
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    return mae, rmse, r2, predictions
    
def create_classes(frustration_score):
    if frustration_score <= 3:
        return 'Low'
    elif frustration_score <= 6:
        return 'Medium'
    else:
        return 'High'

class FrustrationClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

class FrustrationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)  # Use LongTensor for classification
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_classifier(model, train_loader, criterion, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

def evaluate_classifier(model, X_test, y_test, class_names, labels=None):
    model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(X_test)
        outputs = model(test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        y_pred = predicted.numpy()
        
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'report': classification_report(
            y_test, y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True
        ),
        'predictions': y_pred
    }