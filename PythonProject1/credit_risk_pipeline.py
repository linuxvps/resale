"""credit_risk_pipeline.py
Pipeline to replicate the hybrid credit‑risk evaluation model
described in “A Hybrid Credit Risk Evaluation Model Based on Three‑Way
Decisions and Stacking Ensemble Approach” on Parsian Bank loan data.

Steps
-----
1. Data loading from Excel
2. Pre‑processing & feature engineering
3. Class balancing with SMOTE
4. Train / test split
5. LightGBM training
6. Probability prediction
7. Loss‑matrix construction
8. Threshold optimisation (u*, v*) with NSGA‑II
9. Three‑way decision assignment
10. Evaluation & report
"""

import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 0) Imports
# ------------------------------------------------------------------
import os
from pathlib import Path
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, \
                            classification_report, confusion_matrix

import lightgbm as lgb

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# ------------------------------------------------------------------
# 1) Parameters (edit if needed)
# ------------------------------------------------------------------
DATA_PATH      = Path(__file__).with_suffix('') / 'ln_loans.xlsx'   # Excel in same dir
TARGET_COLUMN  = 'FILE_STATUS_TITLE2'        # Original status column
POSITIVE_STATI = {'سررسید گذشته', 'مشکوک الوصول', 'سوخت شده'}  # ⇒ label 1 (default)
NEGATIVE_STATI = {'فعال'}                                         # ⇒ label 0 (paid)

TEST_SIZE      = 0.2
RANDOM_STATE   = 42
SMOTE_K        = 5                     # neighbours for SMOTE
N_ESTIMATORS   = 300
LEARNING_RATE  = 0.05
MAX_DEPTH      = -1
N_GEN          = 80                    # NSGA‑II generations
POP_SIZE       = 120

# ------------------------------------------------------------------
# 2) Helper functions
# ------------------------------------------------------------------
def make_label(x: str) -> int:
    """Map Persian status strings to 0/1 label."""
    if pd.isna(x):
        return 1
    x = str(x).strip()
    if x in POSITIVE_STATI:
        return 1
    if x in NEGATIVE_STATI:
        return 0
    # Unknown → assume risky
    return 1

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Very simple preprocessing:
      • drop columns with >80 % missing
      • numeric ⇒ fill mean
      • categorical ⇒ fill mode then one‑hot encode
    """
    # drop high‑missing
    thresh = int(len(df)*0.2)
    df = df.dropna(axis=1, thresh=thresh).copy()

    # separate
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = [c for c in df.columns if c not in num_cols]

    # impute
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    # one‑hot
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def build_cost_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Compute instance‑wise cost proxy using loan amount.
    (مطابق جدول هزینه مقاله می‌توانید ضرایب را تغییر دهید)
    """
    return df['LOAN_AMOUNT'].to_numpy()

def total_cost(y_true, y_pred, loan_amt):
    """Total monetary cost based on predictions."""
    tp = (y_true==1) & (y_pred==1)
    tn = (y_true==0) & (y_pred==0)
    fp = (y_true==0) & (y_pred==1)   # false positive ⇒ reject good
    fn = (y_true==1) & (y_pred==0)   # false negative ⇒ approve defaulter
    cost = 0.0
    cost += 0.6 * loan_amt[fn].sum()   # FN
    cost += 0.1 * loan_amt[fp].sum()   # FP
    return cost

# ------------------------------------------------------------------
# 3) Load & preprocess data
# ------------------------------------------------------------------
print('Reading', DATA_PATH)
df_raw = pd.read_excel(DATA_PATH)

# برچسب‌گذاری
df_raw['label'] = df_raw[TARGET_COLUMN].apply(make_label)

# پیش‌پردازش
df = preprocess(df_raw.drop(columns=[TARGET_COLUMN]))

X = df.drop(columns=['label'])
y = df['label']

# ------------------------------------------------------------------
# 4) Train / test split & SMOTE
# ------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

sm = SMOTE(k_neighbors=SMOTE_K, random_state=RANDOM_STATE)
x_train_bal, y_train_bal = sm.fit_resample(x_train, y_train)

# ------------------------------------------------------------------
# 5) LightGBM train
# ------------------------------------------------------------------
lgb_params = {
    'objective':'binary',
    'metric':'None',
    'n_estimators': N_ESTIMATORS,
    'learning_rate': LEARNING_RATE,
    'random_state': RANDOM_STATE,
    'max_depth': MAX_DEPTH
}
model = lgb.LGBMClassifier(**lgb_params)
model.fit(x_train_bal, y_train_bal)

probs_test = model.predict_proba(x_test)[:,1]

# ------------------------------------------------------------------
# 6) Loss matrix (loan amounts)
# ------------------------------------------------------------------
loan_vector = build_cost_matrix(df_raw.loc[x_test.index])

# ------------------------------------------------------------------
# 7) Threshold optimisation with NSGA‑II
# ------------------------------------------------------------------
class ThresholdProblem(ElementwiseProblem):
    def __init__(self, y_true, probs, loan_amt):
        super().__init__(n_var=2, n_obj=2, n_constr=1,
                         xl=np.array([0.,0.]), xu=np.array([1.,1.]))
        self.y_true = y_true
        self.probs  = probs
        self.loan   = loan_amt

    def _evaluate(self, x, out, *_):
        u, v  = x
        g1    = u - v                # باید u ≤ v
        pred  = np.where(self.probs >= v, 1,
                         np.where(self.probs <= u, 0, 1))  # defer→rejected
        bac   = 1 - balanced_accuracy_score(self.y_true, pred)  # کمینه
        cost  = total_cost(self.y_true, pred, self.loan) / 1e9   # مقیاس
        out['F'] = [cost, bac]
        out['G'] = [g1]

problem     = ThresholdProblem(y_test.values, probs_test, loan_vector)
algorithm   = NSGA2(pop_size=POP_SIZE, eliminate_duplicates=True)
termination = get_termination('n_gen', N_GEN)

res = minimize(problem, algorithm, termination,
               seed=RANDOM_STATE, verbose=False)

# راه‌حل با کم‌ترین هزینه و بیش‌ترین BAC
idx  = np.lexsort((res.F[:,1], res.F[:,0]))
u_star, v_star = res.X[idx[0]]
print(f"Optimal thresholds: u*={u_star:.4f}  v*={v_star:.4f}")

# ------------------------------------------------------------------
# 8) Final evaluation
# ------------------------------------------------------------------
pred_final = np.where(probs_test >= v_star, 1,
                      np.where(probs_test <= u_star, 0, 1))

bac  = balanced_accuracy_score(y_test, pred_final)
auc  = roc_auc_score(y_test, probs_test)
cm   = confusion_matrix(y_test, pred_final)
cost = total_cost(y_test.values, pred_final, loan_vector)

print("\n=== Evaluation ===")
print("Balanced Accuracy :", bac)
print("AUC               :", auc)
print("Total Cost (IRR)  :", int(cost))
print("Confusion Matrix (rows: true 1/0):\n", cm)
print("\nDetailed Report:")
print(classification_report(y_test, pred_final))
