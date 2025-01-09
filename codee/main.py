import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from nsga2 import NSGA2  # Assuming NSGA-II is implemented or imported from a library

# Step 1: Load the dataset
data = pd.read_excel('Final_Credit_Risk_Dataset.xlsx')

# Step 2: Preprocessing the data
# Convert categorical columns to numerical
categorical_columns = ['Company Type', 'Industry Type', 'Availability of Housing', 'Loan Purpose', 'Individual Application']
for col in categorical_columns:
    data[col] = pd.factorize(data[col])[0]

# Define features and target
X = data.drop(['Number of Default Events'], axis=1)
y = data['Number of Default Events'].apply(lambda x: 1 if x > 0 else 0)  # Binary classification (default or not)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build the three-way decision model
# Use LGBM to predict default probabilities
lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train, y_train)
probs = lgbm.predict_proba(X_test)[:, 1]

# Define thresholds for three-way decisions
def three_way_decision(prob, alpha, beta):
    if prob >= alpha:
        return 'POS'
    elif prob <= beta:
        return 'NEG'
    else:
        return 'BND'

# Step 4: Apply three-way decisions to the test set
alpha, beta = 0.7, 0.3  # Initial thresholds (will be optimized later)
y_pred_three_way = [three_way_decision(p, alpha, beta) for p in probs]

# Step 5: Stacking ensemble model
# First layer classifiers
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
ada = AdaBoostClassifier(random_state=42)

# Train first layer classifiers
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
ada.fit(X_train, y_train)

# Meta-classifier (second layer)
lr = LogisticRegression()
meta_features = np.column_stack([
    rf.predict(X_test),
    gb.predict(X_test),
    ada.predict(X_test)
])

lr.fit(meta_features, y_test)
y_pred_meta = lr.predict(meta_features)

# Step 6: Evaluate the stacking model
accuracy = accuracy_score(y_test, y_pred_meta)
precision = precision_score(y_test, y_pred_meta)
recall = recall_score(y_test, y_pred_meta)
f1 = f1_score(y_test, y_pred_meta)

print("Stacking Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Step 7: Optimize thresholds using NSGA-II (Pseudo-code)
def objective_function(alpha, beta):
    decision_cost = ...  # Calculate decision cost based on alpha and beta
    boundary_size = alpha - beta
    return decision_cost, boundary_size

nsga2 = NSGA2(objective_function, bounds=[(0.5, 0.9), (0.1, 0.5)])
optimal_thresholds = nsga2.run()

print("Optimal thresholds:", optimal_thresholds)
