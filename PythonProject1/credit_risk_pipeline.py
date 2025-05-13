# -*- coding: utf-8 -*-
"""
credit_risk_pipeline_refined.py
پیاده‌سازی کامل مدل سه‌راهه بهینه‌شده با NSGA‑II
اجرای مستقل:  python credit_risk_pipeline_refined.py
"""

import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# ------------------------------------------------------------------
# پارامترهای قابل تغییر
# ------------------------------------------------------------------
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

DATA_FILE = r'C:\Users\nima\data\ln_loans.xlsx'
TARGET_COL = 'FILE_STATUS_TITLE2'
LOAN_AMT_COL = 'LOAN_AMOUNT'
INTEREST_RATE_COL = 'CURRENT_LOAN_RATES'
DURATION_Y_COL = 'LOAN_DURATION_YEAR'   # اگر وجود ندارد ثابت 1 لحاظ می‌شود

GOOD_LABELS = {'فعال', 'پرداخت شده كامل', 'ضمانت نامه صادر شده',
               'خاتمه عادي', 'اعطاء كامل', 'اعطاء  ناقص', 'باطل شده', 'جاري'}
BAD_LABELS = {'سررسيد گذشته', 'مشكوك الوصول', 'وثيقه ناقص', 'سررسيد',
              'معوق', 'منقضي شده', 'ابطال مصوبه ضمانت نامه',
              'درخواست رد شده', 'سررسيدشده پرداخت نشده'}

TEST_SIZE = 0.2
RANDOM_STATE = 42
SMOTE_K = 5

LGB_PARAMS = dict(objective='binary',
                  metric='None',
                  n_estimators=300,
                  learning_rate=0.05,
                  max_depth=-1,
                  random_state=RANDOM_STATE)

NSGA_POP = 100
NSGA_GEN = 200

# ------------------------------------------------------------------
# توابع کمکی
# ------------------------------------------------------------------
def status_to_label(s):
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

def preprocess(df):
    df = df.dropna(axis=1, thresh=int(len(df)*0.2)).copy()

    # تبدیل تاریخ وام به ثانیه
    if 'LOAN_DATE' in df.columns:
        df['LOAN_DATE'] = pd.to_datetime(df['LOAN_DATE'], errors='coerce')
        df['loan_timestamp'] = df['LOAN_DATE'].astype('int64') // 10**9
        df['loan_timestamp'] = df['loan_timestamp'].replace(
            -9223372036, np.nan).fillna(df['loan_timestamp'].median())
        df = df.drop(columns=['LOAN_DATE'])

    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = [c for c in df.columns if c not in num_cols]

    for c in num_cols:
        df[c] = df[c].fillna(df[c].mean())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0])

    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ------------------------------------------------------------------
# بارگذاری داده
# ------------------------------------------------------------------
print('Reading data …')
df = pd.read_excel(DATA_FILE).dropna(axis=1, how='all')

df['label'] = df[TARGET_COL].apply(status_to_label)

# تکمیل ستون‌های اصلی قبل از بهره
df[LOAN_AMT_COL] = pd.to_numeric(df[LOAN_AMT_COL], errors='coerce').fillna(
    df[LOAN_AMT_COL].mean())
df[INTEREST_RATE_COL] = pd.to_numeric(df[INTEREST_RATE_COL],
                                      errors='coerce').fillna(
    df[INTEREST_RATE_COL].mean())

years_series = pd.to_numeric(df.get(DURATION_Y_COL, 1),
                             errors='coerce').fillna(1)

df['interest_cash'] = yearly_interest(df[LOAN_AMT_COL],
                                      df[INTEREST_RATE_COL],
                                      years_series)

model_df = preprocess(df.drop(columns=[TARGET_COL]))

X = model_df.drop(columns=['label', 'interest_cash', LOAN_AMT_COL])
y = model_df['label']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

# بالانس
sm = SMOTE(k_neighbors=SMOTE_K, random_state=RANDOM_STATE)
x_train_bal, y_train_bal = sm.fit_resample(x_train, y_train)

# ------------------------------------------------------------------
# آموزش مدل احتمالات پیش‌فرض
# ------------------------------------------------------------------
print('Training LightGBM …')
clf = lgb.LGBMClassifier(**LGB_PARAMS)
clf.fit(x_train_bal, y_train_bal)

prob_test = clf.predict_proba(x_test)[:, 1]

loan_amt = df.loc[x_test.index, LOAN_AMT_COL].values
interest = df.loc[x_test.index, 'interest_cash'].values

lambda_NP = loan_amt + interest
lambda_PN = interest

# ------------------------------------------------------------------
# مسئله بهینه‌سازی
# ------------------------------------------------------------------
class ThresholdProblem(ElementwiseProblem):
    def __init__(self, y, p, lnp, lpn):
        super().__init__(n_var=2, n_obj=2, n_constr=1,
                         xl=np.array([0., 0.]), xu=np.array([1., 1.]))
        self.y, self.p, self.lnp, self.lpn = y, p, lnp, lpn

    def _evaluate(self, x, out, *_):
        u, v = x
        # قید u+v≤1
        g = u + v - 1

        alpha = (self.lpn - v*self.lpn) / (u*self.lnp - v*self.lpn + self.lpn)
        beta = (v*self.lpn) / (v*self.lpn + self.lnp - u*self.lnp)

        decision = np.where(self.p >= alpha, 1,
                            np.where(self.p <= beta, 0, 2))

        # f1: هزینه کل
        cost = np.where(decision == 0,
                        np.where(self.y == 1, self.lnp, 0),
                        np.where(decision == 1,
                                 np.where(self.y == 0, self.lpn, 0),
                                 np.where(self.y == 1, u*self.lnp, v*self.lpn)))
        f1 = cost.sum()

        # f2: میانگین پهنای مرزی
        f2 = np.mean(alpha - beta)

        out['F'] = [f1, f2]
        out['G'] = [g]

print('Running NSGA‑II …')
problem = ThresholdProblem(y_test.values, prob_test, lambda_NP, lambda_PN)
algorithm = NSGA2(pop_size=NSGA_POP, eliminate_duplicates=True)
result = minimize(problem,
                  algorithm,
                  get_termination('n_gen', NSGA_GEN),
                  seed=RANDOM_STATE,
                  verbose=False)

# بهترین جواب: حداقل f2 سپس f1
idx = np.lexsort((result.F[:, 0], result.F[:, 1]))
u_star, v_star = result.X[idx[0]]
print(f'Optimal (u*, v*) = ({u_star:.4f}, {v_star:.4f})')

# ------------------------------------------------------------------
# ارزیابی
# ------------------------------------------------------------------
alpha_star = (lambda_PN - v_star*lambda_PN) / (
        u_star*lambda_NP - v_star*lambda_PN + lambda_PN)
beta_star = (v_star*lambda_PN) / (
        v_star*lambda_PN + lambda_NP - u_star*lambda_NP)

final_decision = np.where(prob_test >= alpha_star, 1,
                          np.where(prob_test <= beta_star, 0, 2))

binary_pred = np.where(final_decision == 2, 1, final_decision)
bac = balanced_accuracy_score(y_test, binary_pred)
auc = roc_auc_score(y_test, prob_test)

print(f'Balanced Accuracy = {bac:.4f}')
print(f'AUC               = {auc:.4f}')

# ماتریس سه‌راهه برای اطلاع بیشتر
cm = confusion_matrix(y_test, final_decision,
                      labels=[1, 0, 2])  # ترتیب: default, good, border
print('Confusion Matrix (rows: true 1/0):\n', cm)
