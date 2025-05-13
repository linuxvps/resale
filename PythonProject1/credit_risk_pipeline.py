# -*- coding: utf-8 -*-
"""
credit_risk_pipeline.py
پیاده‌سازی کامل مدل «ارزیابی ریسک اعتباری ترکیبی مبتنی بر تصمیم‌گیری سه‌راهه و
بهینه‌سازی NSGA‑II» مطابق مقالهٔ مرجع. کد به‌صورت مستقل اجرا می‌شود:

    python credit_risk_pipeline.py

پیش‌نیازهای پایتون: lightgbm, imbalanced‑learn, pymoo, pandas, numpy.
"""
import warnings

warnings.filterwarnings('ignore')

import os
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import lightgbm as lgb
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# ----------------------------------------------------------------------
# پارامترهای کلی؛ در صورت نیاز مقداردهی دوباره کنید
# ----------------------------------------------------------------------
DATA_FILE = Path(__file__).with_suffix('') / 'ln_loans.xlsx'
TARGET_COL = 'FILE_STATUS_TITLE2'          # وضعیت واقعی وام (پیش‌فرض/غیردرزمان)
LOAN_AMT_COL = 'LOAN_AMOUNT'              # اصل وام
INTEREST_RATE_COL = 'CURRENT_LOAN_RATES'  # نرخ سود سالانه به درصد
DURATION_Y_COL = 'LOAN_DURATION_YEAR'     # مدت به سال؛ اگر وجود ندارد عدد ثابت دهید

GOOD_LABELS = {'فعال'}           # وام سالم
BAD_LABELS = {'سررسید گذشته', 'مشکوک الوصول', 'سوخت شده'}   # نکول

TEST_SIZE = 0.2
RANDOM_STATE = 42
SMOTE_K = 5
LGB_PARAMS = dict(objective='binary',
                  metric='None',
                  n_estimators=300,
                  learning_rate=0.05,
                  max_depth=-1,
                  random_state=RANDOM_STATE)

NSGA_POP = 100        # اندازه جمعیت
NSGA_GEN = 200        # تکرارها

# ----------------------------------------------------------------------
# توابع کمکی
# ----------------------------------------------------------------------
def persian_status_to_label(s):
    if pd.isna(s):
        return 1
    s = str(s).strip()
    if s in GOOD_LABELS:
        return 0
    if s in BAD_LABELS:
        return 1
    return 1  # پیش‌فرض: مشکوک

def yearly_interest(amt, rate_pct, years):
    return amt * (rate_pct / 100.0) * years

def preprocess(df):
    """حذف ستون‌های بسیار ناقص، تکمیل مقادیر و وان‌هات."""
    df = df.dropna(axis=1, thresh=int(len(df) * 0.2)).copy()
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = [c for c in df.columns if c not in num_cols]

    for c in num_cols:
        df[c] = df[c].fillna(df[c].mean())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0])

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

# ----------------------------------------------------------------------
# بارگذاری و آماده‌سازی داده‌ها
# ----------------------------------------------------------------------
print('Reading data …')
data = pd.read_excel(DATA_FILE)

# برچسب‌گذاری
data['label'] = data[TARGET_COL].apply(persian_status_to_label)

# درآمد سودی تخمینی برای هر وام طبق فرمول مقاله
years = data.get(DURATION_Y_COL, pd.Series(np.ones(len(data))))
data['interest_cash'] = yearly_interest(data[LOAN_AMT_COL], data[INTEREST_RATE_COL], years)

# استخراج فقط ستون‌های تحلیلی (بقیه با وان‌هات رمزگذاری می‌شوند)
model_df = preprocess(data.drop(columns=[TARGET_COL]))

X = model_df.drop(columns=['label', 'interest_cash', LOAN_AMT_COL])
y = model_df['label']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# بالانس کلاس‌ها با SMOTE
sm = SMOTE(k_neighbors=SMOTE_K, random_state=RANDOM_STATE)
x_train_bal, y_train_bal = sm.fit_resample(x_train, y_train)

# ----------------------------------------------------------------------
# آموزش LGBM و محاسبه احتمال نکول
# ----------------------------------------------------------------------
print('Training LightGBM …')
clf = lgb.LGBMClassifier(**LGB_PARAMS)
clf.fit(x_train_bal, y_train_bal)

prob_test = clf.predict_proba(x_test)[:, 1]  # P(default)

# بردار مقادیر پولی برای lambda‌ها
loan_amt = data.loc[x_test.index, LOAN_AMT_COL].values
interest = data.loc[x_test.index, 'interest_cash'].values

lambda_NP = loan_amt + interest      # پذیرشِ مقصر
lambda_PN = interest                 # ردِ بی‌مقصر

# ----------------------------------------------------------------------
# تعریف مسئله بهینه‌سازی دوهدفه دقیق طبق مقاله
# ----------------------------------------------------------------------
class ThresholdProblem(ElementwiseProblem):
    def __init__(self, y_true, p_def, lam_NP, lam_PN):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array([0.0, 0.0]),
                         xu=np.array([1.0, 1.0]))
        self.y = y_true
        self.p = p_def
        self.lnp = lam_NP
        self.lpn = lam_PN

    def _evaluate(self, x, out, *args, **kwargs):
        u, v = x
        # قید اصلی
        g = u + v - 1.0

        # آستانه‌های نمونه‌ای طبق فرمول (Eq. 5)
        alpha = (self.lpn - v * self.lpn) / (
                u * self.lnp - v * self.lpn + self.lpn)
        beta = (v * self.lpn) / (
                v * self.lpn + self.lnp - u * self.lnp)

        # تصمیم سه‌راهه
        pred = np.where(self.p >= alpha, 1,
                        np.where(self.p <= beta, 0, 2))  # 2 = BND

        # f1: هزینهٔ کل (Eq. 8)
        cost = np.where(pred == 0,
                        np.where(self.y == 1, self.lnp, 0),  # پذیرش اشتباه
                        np.where(pred == 1,
                                 np.where(self.y == 0, self.lpn, 0),  # رد اشتباه
                                 np.where(pred == 2,
                                          np.where(self.y == 1,
                                                   u * self.lnp,
                                                   v * self.lpn),
                                          0)))
        f1 = cost.sum()

        # f2: اندازه مرزی
        f2 = (alpha - beta).sum()

        out['F'] = [f1, f2]
        out['G'] = [g]

print('Running NSGA‑II …')
problem = ThresholdProblem(y_test.values, prob_test, lambda_NP, lambda_PN)
algo = NSGA2(pop_size=NSGA_POP, eliminate_duplicates=True)
res = minimize(problem,
               algo,
               get_termination('n_gen', NSGA_GEN),
               seed=RANDOM_STATE,
               verbose=False)

# گزینش بهترین راه‌حل (کمترین f2 سپس f1)
obj = res.F
idx = np.lexsort((obj[:, 0], obj[:, 1]))  # ابتدا f2 سپس f1
u_star, v_star = res.X[idx[0]]
print(f'Optimal (u*, v*) = ({u_star:.4f}, {v_star:.4f})')

# ----------------------------------------------------------------------
# ارزیابی نهایی
# ----------------------------------------------------------------------
def classify(p, alpha, beta):
    return np.where(p >= alpha, 1,
                    np.where(p <= beta, 0, 2))

alpha_star = (lambda_PN - v_star * lambda_PN) / (
        u_star * lambda_NP - v_star * lambda_PN + lambda_PN)
beta_star = (v_star * lambda_PN) / (
        v_star * lambda_PN + lambda_NP - u_star * lambda_NP)

pred_final = classify(prob_test, alpha_star, beta_star)

# تبدیل سه‌کلاسه به دودویی برای متریک BAC:
pred_binary = np.where(pred_final == 2, 1, pred_final)  # تعویق را خطا فرض می‌کنیم
bac = balanced_accuracy_score(y_test, pred_binary)
auc = roc_auc_score(y_test, prob_test)

print(f'Balanced Accuracy = {bac:.4f}')
print(f'AUC              = {auc:.4f}')
