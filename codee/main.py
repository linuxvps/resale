import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import geatpy as ea  # NSGA-II library for multi-objective optimization

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
xgb = XGBClassifier(random_state=42)

# Train first layer classifiers
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
ada.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Meta-classifier (second layer)
lr = LogisticRegression()
meta_features = np.column_stack([
    rf.predict(X_test),
    gb.predict(X_test),
    ada.predict(X_test),
    xgb.predict(X_test)
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

# Step 7: Optimize thresholds using NSGA-II
class ThreeWayDecisionOptimization(ea.Problem):
    def __init__(self):
        name = 'ThreeWayDecisionOptimization'
        M = 2  # Number of objectives
        maxormins = [1, 1]  # Minimize both objectives
        Dim = 2  # Number of decision variables (alpha, beta)
        varTypes = [0, 0]  # Continuous variables
        lb = [0.5, 0.1]
        ub = [0.9, 0.5]
        lbin = [1, 1]
        ubin = [1, 1]
        super().__init__(name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        Vars = pop.Phen  # Decision variables (alpha, beta)
        alpha = Vars[:, [0]]
        beta = Vars[:, [1]]
        decision_costs = np.abs(alpha - beta)  # Simplified cost function
        boundary_sizes = alpha - beta
        pop.ObjV = np.hstack([decision_costs, boundary_sizes])

# Run the optimization
problem = ThreeWayDecisionOptimization()
algorithm = ea.moea_NSGA2_templet(problem, ea.Population(Encoding='RI', NIND=100), MAXGEN=200, logTras=1)
res = ea.optimize(algorithm, verbose=False)
optimal_thresholds = res['Vars'][np.argmin(res['ObjV'][:, 0])]  # Get the best thresholds

print("Optimal thresholds:", optimal_thresholds)
