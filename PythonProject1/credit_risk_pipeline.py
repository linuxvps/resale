# -*- coding: utf-8 -*-
"""
credit_risk_pipeline_complete.py
مدل سه‌راهه با بهینه‌سازی NSGA‑II + استکینگ مرحلهٔ دوم + ماتریس زیان صریح
پیش‌نیاز: pandas, numpy, lightgbm, imbalanced‑learn, pymoo, openpyxl,
          scikit-learn, xgboost
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier,
                              StackingClassifier)
import lightgbm as lgb
from xgboost import XGBClassifier
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# ----------------------------------------------------------------------
# پیکره‌بندی
# ----------------------------------------------------------------------
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

DATA_FILE = r'C:\Users\nima\data\ln_loans.xlsx'
TARGET_COL = 'FILE_STATUS_TITLE2'
LOAN_AMT_COL = 'LOAN_AMOUNT'
INTEREST_RATE_COL = 'CURRENT_LOAN_RATES'
DURATION_Y_COL = 'LOAN_DURATION_YEAR'

GOOD_LABELS = {'فعال', 'پرداخت شده كامل', 'ضمانت نامه صادر شده',
               'خاتمه عادي', 'اعطاء كامل', 'اعطاء  ناقص', 'باطل شده', 'جاري'}
BAD_LABELS = {'سررسيد گذشته', 'مشكوك الوصول', 'وثيقه ناقص', 'سررسيد',
              'معوق', 'منقضي شده', 'ابطال مصوبه ضمانت نامه',
              'درخواست رد شده', 'سررسيدشده پرداخت نشده'}

TEST_SIZE = 0.2
RANDOM_STATE = 42
SMOTE_K = 5

LGB_PARAMS = dict(objective='binary', metric='None',
                  n_estimators=300, learning_rate=0.05,
                  max_depth=-1, random_state=RANDOM_STATE)

NSGA_POP = 100
NSGA_GEN = 200
EPS = 1e-6                    # حاشیه برای u+v<1

# ----------------------------------------------------------------------
# توابع کمکی
# ----------------------------------------------------------------------
def status_to_label(s: str) -> int:
    if pd.isna(s):
        return 1
    s = str(s).strip()
    if s in GOOD_LABELS:
        return 0
    if s in BAD_LABELS:
        return 1
    return 1

def yearly_interest(amount, rate_pct, years):
    return amount * (rate_pct / 100.0) * years

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, thresh=int(len(df) * 0.2)).copy()

    if 'LOAN_DATE' in df.columns:
        df['LOAN_DATE'] = pd.to_datetime(df['LOAN_DATE'], errors='coerce')
        df['loan_timestamp'] = df['LOAN_DATE'].astype('int64') // 10**9
        df['loan_timestamp'].replace(-9223372036, np.nan, inplace=True)
        df['loan_timestamp'].fillna(df['loan_timestamp'].median(), inplace=True)
        df.drop(columns=['LOAN_DATE'], inplace=True)

    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = [c for c in df.columns if c not in num_cols]

    for c in num_cols:
        df[c].fillna(df[c].mean(), inplace=True)
    for c in cat_cols:
        df[c].fillna(df[c].mode().iloc[0], inplace=True)

    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ----------------------------------------------------------------------
# ۱) بارگذاری داده
# ----------------------------------------------------------------------
print('Reading data …')
raw = pd.read_excel(DATA_FILE).dropna(axis=1, how='all')
raw['label'] = raw[TARGET_COL].apply(status_to_label)

raw[LOAN_AMT_COL] = pd.to_numeric(raw[LOAN_AMT_COL], errors='coerce') \
                       .fillna(raw[LOAN_AMT_COL].mean())
raw[INTEREST_RATE_COL] = pd.to_numeric(raw[INTEREST_RATE_COL],
                                       errors='coerce') \
                           .fillna(raw[INTEREST_RATE_COL].mean())
years_ser = pd.to_numeric(raw.get(DURATION_Y_COL, 1),
                          errors='coerce').fillna(1)

raw['interest_cash'] = yearly_interest(raw[LOAN_AMT_COL],
                                       raw[INTEREST_RATE_COL],
                                       years_ser)

df = preprocess(raw.drop(columns=[TARGET_COL]))

X_full = df.drop(columns=['label', 'interest_cash', LOAN_AMT_COL])
y_full = df['label']

x_tr, x_te, y_tr, y_te = train_test_split(
    X_full, y_full, test_size=TEST_SIZE, stratify=y_full,
    random_state=RANDOM_STATE)

# ----------------------------------------------------------------------
# ۲) مدل احتمال نکول (LightGBM با SMOTE)
# ----------------------------------------------------------------------
sm = SMOTE(k_neighbors=SMOTE_K, random_state=RANDOM_STATE)
x_tr_bal, y_tr_bal = sm.fit_resample(x_tr, y_tr)

print('Training LightGBM …')
lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
lgb_model.fit(x_tr_bal, y_tr_bal)
prob_te = lgb_model.predict_proba(x_te)[:, 1]

loan_amt_te = raw.loc[x_te.index, LOAN_AMT_COL].values
interest_te = raw.loc[x_te.index, 'interest_cash'].values
lambda_NP = loan_amt_te + interest_te
lambda_PN = interest_te

# ----------------------------------------------------------------------
# ۳) مسئله‌ی NSGA‑II
# ----------------------------------------------------------------------
class ThresholdProblem(ElementwiseProblem):
    def __init__(self, y, p, lnp, lpn):
        super().__init__(n_var=2, n_obj=2, n_constr=1,
                         xl=np.array([0., 0.]), xu=np.array([1., 1.]))
        self.y, self.p, self.lnp, self.lpn = y, p, lnp, lpn

    def _evaluate(self, x, out, *_):
        u, v = x
        g = u + v - (1.0 - EPS)        # قید: u+v < 1

        alpha = (self.lpn - v*self.lpn) / (u*self.lnp - v*self.lpn + self.lpn)
        beta  = (v*self.lpn) / (v*self.lpn + self.lnp - u*self.lnp)

        dec = np.where(self.p >= alpha, 1,
                       np.where(self.p <= beta, 0, 2))

        cost = np.where(dec == 0,
                        np.where(self.y == 1, self.lnp, 0),
                        np.where(dec == 1,
                                 np.where(self.y == 0, self.lpn, 0),
                                 np.where(self.y == 1, u*self.lnp, v*self.lpn)))
        f1 = cost.sum()
        f2 = np.sum(alpha - beta)      # مطابق معادله ۱۳

        out['F'] = [f1, f2]
        out['G'] = [g]

print('Running NSGA‑II …')
prob = ThresholdProblem(y_te.values, prob_te, lambda_NP, lambda_PN)
algo = NSGA2(pop_size=NSGA_POP, eliminate_duplicates=True)
res  = minimize(prob, algo, get_termination('n_gen', NSGA_GEN),
                seed=RANDOM_STATE, verbose=False)

def pick_solution(res, y, p, lnp, lpn):
    """کمترین #BND، سپس f1."""
    best_idx, best_bnd, best_f1 = None, np.inf, np.inf
    for i, (u, v) in enumerate(res.X):
        alpha = (lpn - v*lpn)/(u*lnp - v*lpn + lpn)
        beta  = (v*lpn)/(v*lpn + lnp - u*lnp)
        bnd_cnt = np.sum((p > beta) & (p < alpha))
        f1 = res.F[i, 0]
        if bnd_cnt < best_bnd or (bnd_cnt == best_bnd and f1 < best_f1):
            best_idx, best_bnd, best_f1 = i, bnd_cnt, f1
    return res.X[best_idx]

u_star, v_star = pick_solution(res, y_te.values, prob_te,
                               lambda_NP, lambda_PN)
print(f'Chosen (u*, v*) = ({u_star:.4f}, {v_star:.4f})')

# ----------------------------------------------------------------------
# ۴) آستانه‌ها و ناحیهٔ‌ تصمیم
# ----------------------------------------------------------------------
alpha_star = (lambda_PN - v_star*lambda_PN) / \
             (u_star*lambda_NP - v_star*lambda_PN + lambda_PN)
beta_star  = (v_star*lambda_PN) / \
             (v_star*lambda_PN + lambda_NP - u_star*lambda_NP)

region = np.where(prob_te >= alpha_star, 'POS',
                  np.where(prob_te <= beta_star, 'NEG', 'BND'))

# ----------------------------------------------------------------------
# ۵) استکینگ مرحلهٔ دوم برای BND
# ----------------------------------------------------------------------
base_ests = [
    ('rf',  RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    ('xgb', XGBClassifier(n_estimators=300, random_state=RANDOM_STATE,
                          eval_metric='logloss', use_label_encoder=False)),
    ('gbdt', GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ('ert', ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    ('ada', AdaBoostClassifier(random_state=RANDOM_STATE))
]

stack_model = StackingClassifier(
    estimators=base_ests,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    n_jobs=-1
)
print('Training stacking model for BND …')
stack_model.fit(x_tr, y_tr)     # مقاله: آموزش رو کل دیتای آموزش بدون SMOTE

# پیش‌بینی نهایی
final_pred = np.empty_like(y_te.values)
# POS → 1 (بدحساب)،  NEG → 0 (خوش‌حساب)
final_pred[region == 'POS'] = 1
final_pred[region == 'NEG'] = 0
# BND→ خروجی استکینگ
bnd_idx = np.where(region == 'BND')[0]
if bnd_idx.size:
    final_pred[bnd_idx] = stack_model.predict(x_te.iloc[bnd_idx])

# ----------------------------------------------------------------------
# ۶) ارزیابی
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ۶) ارزیابی دقیقاً مطابق مقاله
# ----------------------------------------------------------------------
from sklearn.metrics import confusion_matrix, roc_auc_score

# 6‑a  ماتریس سردرگمی
tn, fp, fn, tp = confusion_matrix(y_te, final_pred, labels=[0, 1]).ravel()

# 6‑b  شاخص‌های معادلات (16) تا (18)
recall_default = tp / (tp + fn) if (tp + fn) else 0          # TP / (TP+FN)
recall_nondef  = tn / (tn + fp) if (tn + fp) else 0          # TN / (TN+FP)
precision      = tp / (tp + fp) if (tp + fp) else 0          # برای FM

B_Acc = (recall_default + recall_nondef) / 2                 # Eq. 16
FM    = (1 + 1**2) * precision * recall_default \
        / (1**2 * (precision + recall_default)) if (precision + recall_default) else 0  # Eq. 17
GM    = np.sqrt(recall_default * recall_nondef)              # Eq. 18

# 6‑c  AUC  (Eq. 19)
AUC = roc_auc_score(y_te, prob_te)                           # همان تعریف کلاسیک

# 6‑d  Decision Cost  (Eq. 20)
#     برای هر رکوردِ تست، اگر برچسب اشتباه باشد λ_NP یا λ_PN افزوده می‌شود
cost_vec = np.where(y_te == 1,
                    np.where(final_pred == 1, 0, lambda_NP),  # پیش‌فرض → λ_NP اگر خطا
                    np.where(final_pred == 0, 0, lambda_PN)) # غیرپیش‌فرض → λ_PN اگر خطا
Dec_Cost = cost_vec.sum()

print(f'B_Acc  = {B_Acc:.4f}')
print(f'FM     = {FM:.4f}')
print(f'GM     = {GM:.4f}')
print(f'AUC    = {AUC:.4f}')
print(f'Cost   = {Dec_Cost:,.0f} IRR')
print(f'TP={tp}, TN={tn}, FP={fp}, FN={fn}')

# ----------------------------------------------------------------------
# ۷) ماتریس زیان صریح و ذخیره
# ----------------------------------------------------------------------
def build_loss_df(idx_array, lnp, lpn, u, v):
    return pd.DataFrame({
        'lambda_PP': 0.0,
        'lambda_PN': lpn,
        'lambda_NP': lnp,
        'lambda_NN': 0.0,
        'lambda_BP': u * lnp,
        'lambda_BN': v * lpn
    }, index=idx_array)

loss_test = build_loss_df(x_te.index, lambda_NP, lambda_PN,
                          u_star, v_star)
loss_test.to_excel('loss_matrix_test.xlsx', index_label='ROW_ID')

lambda_NP_all = raw[LOAN_AMT_COL] + raw['interest_cash']
lambda_PN_all = raw['interest_cash']
loss_all = build_loss_df(raw.index, lambda_NP_all, lambda_PN_all,
                         u_star, v_star)
loss_all.to_excel('loss_matrix_all.xlsx', index_label='ROW_ID')
print('Loss‑matrix files saved.')
