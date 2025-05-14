# -*- coding: utf-8 -*-
"""
credit_risk_pipeline_cv5.py
همان مدل سه‌راهه + NSGA‑II + استکینگ، این بار با ۵‑Fold CV
"""

import warnings, os, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             confusion_matrix)
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

# ---------- پیکره‌بندی کلی ----------
os.environ['LOKY_MAX_CPU_COUNT'] = '8'
DATA_FILE          = r'C:\Users\nima\data\ln_loans.xlsx'
TARGET_COL         = 'FILE_STATUS_TITLE2'
LOAN_AMT_COL       = 'LOAN_AMOUNT'
INTEREST_RATE_COL  = 'CURRENT_LOAN_RATES'
DURATION_Y_COL     = 'LOAN_DURATION_YEAR'

GOOD_LABELS = {'فعال', 'پرداخت شده كامل', 'ضمانت نامه صادر شده',
               'خاتمه عادي', 'اعطاء كامل', 'اعطاء  ناقص', 'باطل شده', 'جاري'}
BAD_LABELS  = {'سررسيد گذشته', 'مشكوك الوصول', 'وثيقه ناقص', 'سررسيد',
               'معوق', 'منقضي شده', 'ابطال مصوبه ضمانت نامه',
               'درخواست رد شده', 'سررسيدشده پرداخت نشده'}

RANDOM_STATE = 42
SMOTE_K      = 5
LGB_PARAMS   = dict(objective='binary', metric='None',
                    n_estimators=300, learning_rate=0.05,
                    max_depth=-1, random_state=RANDOM_STATE)
NSGA_POP, NSGA_GEN, EPS = 100, 200, 1e-6     # قید u+v<1

# ---------- توابع پایه ----------
def status_to_label(s):
    if pd.isna(s): return 1
    s = str(s).strip()
    return 0 if s in GOOD_LABELS else 1

def yearly_interest(amount, rate_pct, years):
    return amount * (rate_pct/100) * years

def preprocess(df):
    df = df.dropna(axis=1, thresh=int(len(df)*0.2)).copy()
    if 'LOAN_DATE' in df.columns:
        df['LOAN_DATE'] = pd.to_datetime(df['LOAN_DATE'], errors='coerce')
        df['loan_ts']   = df['LOAN_DATE'].astype('int64')//10**9
        df['loan_ts'].replace(-9223372036, np.nan, inplace=True)
        df['loan_ts'].fillna(df['loan_ts'].median(), inplace=True)
        df.drop(columns=['LOAN_DATE'], inplace=True)
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = [c for c in df.columns if c not in num_cols]
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    for c in cat_cols:
        df[c].fillna(df[c].mode().iloc[0], inplace=True)
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

class ThresholdProblem(ElementwiseProblem):
    def __init__(self, y, p, lnp, lpn):
        super().__init__(n_var=2, n_obj=2, n_constr=1,
                         xl=np.array([0.,0.]), xu=np.array([1.,1.]))
        self.y, self.p, self.lnp, self.lpn = y, p, lnp, lpn
    def _evaluate(self, x, out, *_):
        u, v = x
        g = u+v-(1-EPS)
        a = (self.lpn-v*self.lpn)/(u*self.lnp - v*self.lpn + self.lpn)
        b = (v*self.lpn)/(v*self.lpn + self.lnp - u*self.lnp)
        dec = np.where(self.p>=a,1,np.where(self.p<=b,0,2))
        cost = np.where(dec==0,
                        np.where(self.y==1,self.lnp,0),
                        np.where(dec==1,
                                 np.where(self.y==0,self.lpn,0),
                                 np.where(self.y==1,u*self.lnp,v*self.lpn)))
        f1 = cost.sum()
        f2 = np.sum(a-b)
        out['F'], out['G'] = [f1,f2],[g]

def pick_solution(res, y, p, lnp, lpn):
    best_i,best_bnd,best_f1 = None,np.inf,np.inf
    for i,(u,v) in enumerate(res.X):
        a = (lpn-v*lpn)/(u*lnp - v*lpn + lpn)
        b = (v*lpn)/(v*lpn + lnp - u*lnp)
        bnd = np.sum((p>b)&(p<a))
        if bnd<best_bnd or (bnd==best_bnd and res.F[i,0]<best_f1):
            best_i,best_bnd,best_f1=i,bnd,res.F[i,0]
    return res.X[best_i]

# ---------- آماده‌سازی داده یک‌بار برای همیشه ----------
print('Reading & preprocessing data …')
raw = pd.read_excel(DATA_FILE).dropna(axis=1, how='all')
raw['label'] = raw[TARGET_COL].apply(status_to_label)
raw[LOAN_AMT_COL]      = pd.to_numeric(raw[LOAN_AMT_COL],
                                       errors='coerce').fillna(
                                       raw[LOAN_AMT_COL].mean())
raw[INTEREST_RATE_COL] = pd.to_numeric(raw[INTEREST_RATE_COL],
                                       errors='coerce').fillna(
                                       raw[INTEREST_RATE_COL].mean())
years = pd.to_numeric(raw.get(DURATION_Y_COL,1),
                      errors='coerce').fillna(1)
raw['interest_cash'] = yearly_interest(raw[LOAN_AMT_COL],
                                       raw[INTEREST_RATE_COL], years)

df = preprocess(raw.drop(columns=[TARGET_COL]))
X_full = df.drop(columns=['label','interest_cash',LOAN_AMT_COL])
y_full = df['label']

# ---------- استکینگ مشترک (برای سرعت، خارج فولد آموزش می‌دهیم) ----------
base = [('rf',  RandomForestClassifier(n_estimators=200,
                                       random_state=RANDOM_STATE)),
        ('xgb', XGBClassifier(n_estimators=300, random_state=RANDOM_STATE,
                              eval_metric='logloss', use_label_encoder=False)),
        ('gbdt',GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ('ert', ExtraTreesClassifier(n_estimators=200,
                                     random_state=RANDOM_STATE)),
        ('ada', AdaBoostClassifier(random_state=RANDOM_STATE))]
stack_clf = StackingClassifier(estimators=base,
                               final_estimator=LogisticRegression(max_iter=1000),
                               cv=KFold(n_splits=5, shuffle=True,
                                        random_state=RANDOM_STATE),
                               n_jobs=-1)

# ---------- حلقهٔ ۵‑Fold ----------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
metrics = []   # هر ردیف: [BAcc, GM, FM, AUC, Cost]

print('\n===== 5‑Fold Cross‑Validation =====')
for fold,(tr_idx,te_idx) in enumerate(kf.split(X_full, y_full),1):
    X_tr, X_te = X_full.iloc[tr_idx], X_full.iloc[te_idx]
    y_tr, y_te = y_full.iloc[tr_idx], y_full.iloc[te_idx]

    # مرحله LGBM
    X_bal,y_bal = SMOTE(k_neighbors=SMOTE_K,random_state=RANDOM_STATE)\
                    .fit_resample(X_tr,y_tr)
    lgb_clf = lgb.LGBMClassifier(**LGB_PARAMS).fit(X_bal,y_bal)
    prob_te = lgb_clf.predict_proba(X_te)[:,1]

    lam_NP = (raw.loc[X_te.index, LOAN_AMT_COL] +
              raw.loc[X_te.index, 'interest_cash']).values
    lam_PN = raw.loc[X_te.index, 'interest_cash'].values

    # NSGA‑II
    res = minimize(ThresholdProblem(y_te.values, prob_te, lam_NP, lam_PN),
                   NSGA2(pop_size=NSGA_POP, eliminate_duplicates=True),
                   get_termination('n_gen', NSGA_GEN),
                   seed=RANDOM_STATE, verbose=False)
    u_star,v_star = pick_solution(res, y_te.values, prob_te, lam_NP, lam_PN)

    alpha = (lam_PN - v_star*lam_PN)/(u_star*lam_NP - v_star*lam_PN + lam_PN)
    beta  = (v_star*lam_PN)/(v_star*lam_PN + lam_NP - u_star*lam_NP)
    region = np.where(prob_te>=alpha,'POS',
                      np.where(prob_te<=beta,'NEG','BND'))

    # استکینگ فقط روی BND
    stack_clf.fit(X_tr,y_tr)
    final_pred = np.empty_like(y_te.values)
    final_pred[region=='POS'] = 1
    final_pred[region=='NEG'] = 0
    bnd_idx = np.where(region=='BND')[0]
    if bnd_idx.size:
        final_pred[bnd_idx] = stack_clf.predict(X_te.iloc[bnd_idx])

    tn,fp,fn,tp = confusion_matrix(y_te,final_pred,
                                   labels=[0,1]).ravel()
    rec_d = tp/(tp+fn) if tp+fn else 0
    rec_n = tn/(tn+fp) if tn+fp else 0
    prec  = tp/(tp+fp) if tp+fp else 0
    BAcc  = (rec_d+rec_n)/2
    FM    = 2*prec*rec_d/(prec+rec_d) if prec+rec_d else 0
    GM    = np.sqrt(rec_d*rec_n)
    AUC   = roc_auc_score(y_te, prob_te)
    cost_vec = np.where(y_te==1,
                        np.where(final_pred==1,0,lam_NP),
                        np.where(final_pred==0,0,lam_PN))
    Cost  = cost_vec.sum()
    metrics.append([BAcc,GM,FM,AUC,Cost])

    print(f'Fold {fold}:  BAcc={BAcc:.4f}  GM={GM:.4f}  '
          f'FM={FM:.4f}  AUC={AUC:.4f}  Cost={Cost:,.0f}')

# ---------- جمع‌بندی ----------
m = np.array(metrics)
print('\n—— 5‑Fold Mean ± Std ——')
for name,col in zip(['BAcc','GM','FM','AUC','Cost'],m.T):
    print(f'{name}: {col.mean():.4f} ± {col.std():.4f}')
