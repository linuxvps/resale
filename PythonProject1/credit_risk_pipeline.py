# -*- coding: utf-8 -*-
"""
credit_risk_pipeline_full.py
مدل سه‌راهه با بهینه‌سازی NSGA‑II و ماتریس زیان صریح

اجرای مستقل:
    python credit_risk_pipeline_full.py

نیاز به پکیج‌های: pandas, numpy, lightgbm, imbalanced‑learn, pymoo, openpyxl
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
# پیکره‌بندی
# ----------------------------------------------------------------------
os.environ["LOKY_MAX_CPU_COUNT"] = "8"              # محدودکردن پردازنده‌ها (اختیاری)

DATA_FILE = r'C:\Users\nima\data\ln_loans.xlsx'     # مسیر فایل ورودی
TARGET_COL = 'FILE_STATUS_TITLE2'                   # وضعیت واقعی وام
LOAN_AMT_COL = 'LOAN_AMOUNT'                        # اصل وام
INTEREST_RATE_COL = 'CURRENT_LOAN_RATES'            # نرخ سود سالانه (٪)
DURATION_Y_COL = 'LOAN_DURATION_YEAR'               # مدت وام به سال (در صورت نبود =۱)

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

# ----------------------------------------------------------------------
# توابع کمکی
# ----------------------------------------------------------------------
def status_to_label(s: str) -> int:
    """تبدیل وضعیت فارسی به برچسب ۰/۱."""
    if pd.isna(s):
        return 1
    s = str(s).strip()
    if s in GOOD_LABELS:
        return 0
    if s in BAD_LABELS:
        return 1
    return 1      # ناشناخته ⇒ بدحساب

def yearly_interest(amount, rate_pct, years):
    return amount * (rate_pct / 100.0) * years

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """حذف ستون‌های ناقص، تبدیل تاریخ، تکمیل داده و وان‌هات."""
    df = df.dropna(axis=1, thresh=int(len(df)*0.2)).copy()

    if 'LOAN_DATE' in df.columns:
        df['LOAN_DATE'] = pd.to_datetime(df['LOAN_DATE'], errors='coerce')
        df['loan_timestamp'] = (df['LOAN_DATE']
                                .astype('int64') // 10**9)     # ثانیه
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
# ۱) بارگذاری و آماده‌سازی داده
# ----------------------------------------------------------------------
print('Reading data …')
raw = pd.read_excel(DATA_FILE).dropna(axis=1, how='all')

raw['label'] = raw[TARGET_COL].apply(status_to_label)

# تکمیل ستون‌های کلیدی پیش از محاسبه‌ی سود
raw[LOAN_AMT_COL] = pd.to_numeric(raw[LOAN_AMT_COL], errors='coerce') \
                       .fillna(raw[LOAN_AMT_COL].mean())
raw[INTEREST_RATE_COL] = pd.to_numeric(raw[INTEREST_RATE_COL], errors='coerce') \
                           .fillna(raw[INTEREST_RATE_COL].mean())
years_series = pd.to_numeric(raw.get(DURATION_Y_COL, 1),
                             errors='coerce').fillna(1)
raw['interest_cash'] = yearly_interest(raw[LOAN_AMT_COL],
                                       raw[INTEREST_RATE_COL],
                                       years_series)

df = preprocess(raw.drop(columns=[TARGET_COL]))

X = df.drop(columns=['label', 'interest_cash', LOAN_AMT_COL])
y = df['label']

x_tr, x_te, y_tr, y_te = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

# SMOTE
sm = SMOTE(k_neighbors=SMOTE_K, random_state=RANDOM_STATE)
x_tr_bal, y_tr_bal = sm.fit_resample(x_tr, y_tr)

# ----------------------------------------------------------------------
# ۲) آموزش LightGBM و احتمال نکول
# ----------------------------------------------------------------------
print('Training LightGBM …')
model = lgb.LGBMClassifier(**LGB_PARAMS)
model.fit(x_tr_bal, y_tr_bal)
prob_te = model.predict_proba(x_te)[:, 1]

# λ‌های پایه
loan_amt_te = raw.loc[x_te.index, LOAN_AMT_COL].values
interest_te = raw.loc[x_te.index, 'interest_cash'].values
lambda_NP = loan_amt_te + interest_te      # پذیرشِ بدحساب
lambda_PN = interest_te                    # ردِ خوش‌حساب

# ----------------------------------------------------------------------
# ۳) تعریف مسئله‌ی بهینه‌سازی
# ----------------------------------------------------------------------
# ------------------------------------------------------------------
# ۳) تعریف مسئله‌ی بهینه‌سازی (نسخه ۱۰۰٪ منطبق با مقاله)
# ------------------------------------------------------------------
EPS = 1e-6        # حاشیهٔ اطمینان برای نامساوی تیز u+v<1

class ThresholdProblem(ElementwiseProblem):
    def __init__(self, y, p, lnp, lpn):
        super().__init__(n_var=2, n_obj=2, n_constr=1,
                         xl=np.array([0., 0.]), xu=np.array([1., 1.]))
        self.y, self.p, self.lnp, self.lpn = y, p, lnp, lpn

    def _evaluate(self, x, out, *_):
        u, v = x
        # قید اصلی با نامساوی تیز
        g = u + v - (1.0 - EPS)         # g ≤ 0   ⇒   u+v < 1

        # آستانه‌های فردی (معادله ۵)
        alpha = (self.lpn - v*self.lpn) / (u*self.lnp - v*self.lpn + self.lpn)
        beta  = (v*self.lpn) / (v*self.lpn + self.lnp - u*self.lnp)

        # تصمیم سه‌راهه
        dec = np.where(self.p >= alpha, 1,
                       np.where(self.p <= beta, 0, 2))

        # f1: هزینهٔ کل (معادله ۸)
        cost = np.where(dec == 0,
                        np.where(self.y == 1, self.lnp, 0),
                        np.where(dec == 1,
                                 np.where(self.y == 0, self.lpn, 0),
                                 np.where(self.y == 1, u*self.lnp, v*self.lpn)))
        f1 = cost.sum()

        # f2: جمع پهناهای مرزی (معادله ۱۳)
        f2 = np.sum(alpha - beta)

        out['F'] = [f1, f2]
        out['G'] = [g]

print('Running NSGA‑II …')
prob = ThresholdProblem(y_te.values, prob_te, lambda_NP, lambda_PN)
algo = NSGA2(pop_size=NSGA_POP, eliminate_duplicates=True)
res  = minimize(prob, algo, get_termination('n_gen', NSGA_GEN),
                seed=RANDOM_STATE, verbose=False)

# راه‌حل منتخب
idx = np.lexsort((res.F[:, 0], res.F[:, 1]))         # کمترین f2 سپس f1
u_star, v_star = res.X[idx[0]]
print(f'Optimal (u*, v*) = ({u_star:.4f}, {v_star:.4f})')

# ----------------------------------------------------------------------
# ۴) ساخت ماتریس زیان صریح
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

loss_test = build_loss_df(x_te.index, lambda_NP, lambda_PN, u_star, v_star)
loss_test.to_excel('loss_matrix_test.xlsx', index_label='ROW_ID')
print('→ loss_matrix_test.xlsx saved')

# برای کل داده (در صورت نیاز)
lambda_NP_all = raw[LOAN_AMT_COL] + raw['interest_cash']
lambda_PN_all = raw['interest_cash']
loss_all = build_loss_df(raw.index, lambda_NP_all, lambda_PN_all, u_star, v_star)
loss_all.to_excel('loss_matrix_all.xlsx', index_label='ROW_ID')
print('→ loss_matrix_all.xlsx saved')

# ----------------------------------------------------------------------
# ۵) ارزیابی نهایی
# ----------------------------------------------------------------------
alpha_star = (lambda_PN - v_star*lambda_PN) / \
             (u_star*lambda_NP - v_star*lambda_PN + lambda_PN)
beta_star  = (v_star*lambda_PN) / \
             (v_star*lambda_PN + lambda_NP - u_star*lambda_NP)

dec_final = np.where(prob_te >= alpha_star, 1,
                     np.where(prob_te <= beta_star, 0, 2))

binary_pred = np.where(dec_final == 2, 1, dec_final)    # تعویق ← خطا
bac = balanced_accuracy_score(y_te, binary_pred)
auc = roc_auc_score(y_te, prob_te)

print(f'Balanced Accuracy = {bac:.4f}')
print(f'AUC               = {auc:.4f}')
