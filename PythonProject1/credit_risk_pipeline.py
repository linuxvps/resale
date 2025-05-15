import os
import time
import warnings

import numpy as np
import pandas as pd

from ResultManager import ResultManager

warnings.filterwarnings('ignore')

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
                              ExtraTreesClassifier, StackingClassifier)
import lightgbm as lgb
from xgboost import XGBClassifier

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

pd.set_option('display.max_columns', None)  # نمایش همه ستون‌ها
pd.set_option('display.expand_frame_repr', False)  # جلوگیری از شکستن به چند خط
pd.set_option('display.width', 200)  # تنظیم عرض کنسول (می‌تونی عدد رو بیشتر هم بذاری)
pd.set_option('display.float_format', '{:,.6f}'.format)  # فرمت عددی دلخواه

# ────────────────  پیکره‌بندی  ────────────────
os.environ['LOKY_MAX_CPU_COUNT'] = '8'
DATA_FILE = r'C:\Users\nima\data\ln_loans_5000.xlsx'
TARGET_COL = 'FILE_STATUS_TITLE2'
LOAN_AMT_COL = 'LOAN_AMOUNT'
INTEREST_RATE_COL = 'CURRENT_LOAN_RATES'
DURATION_Y_COL = 'LOAN_DURATION_YEAR'

GOOD_LABELS = {'فعال', 'پرداخت شده كامل', 'ضمانت نامه صادر شده', 'خاتمه عادي', 'اعطاء كامل', 'اعطاء  ناقص', 'باطل شده',
               'جاري'}
BAD_LABELS = {'سررسيد گذشته', 'مشكوك الوصول', 'وثيقه ناقص', 'سررسيد', 'معوق', 'منقضي شده', 'ابطال مصوبه ضمانت نامه',
              'درخواست رد شده', 'سررسيدشده پرداخت نشده'}

RANDOM_STATE = 42
SMOTE_K = 5

# LightGBM
LGB_PARAMS = dict(objective='binary', metric='None', n_estimators=300, learning_rate=0.05, max_depth=-1,
                  random_state=RANDOM_STATE)

# NSGA-II
NSGA_POP, NSGA_GEN, EPS = 100, 200, 1e-6

# حذف همبستگی
THRESH_CORR = 0.95
TOP_N_FEATS = 20

# تحلیل حساسیت
param_grid = [(50, 100), (80, 120), (100, 200), (150, 300)]
KFoldSize = 5



# ────────────────  توابع کمکی  ────────────────
def status_to_label(s):
    if pd.isna(s): return 1
    return 0 if str(s).strip() in GOOD_LABELS else 1


def yearly_interest(amount, rate_pct, years):
    return amount * (rate_pct / 100) * years


def drop_high_corr(df_num, thresh=THRESH_CORR):
    corr = df_num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > thresh)]
    return df_num.drop(columns=to_drop), to_drop


def preprocess(df):
    df = df.dropna(axis=1, thresh=int(len(df) * 0.2)).copy()
    if 'LOAN_DATE' in df.columns:
        df['LOAN_DATE'] = pd.to_datetime(df['LOAN_DATE'], errors='coerce')
        df['loan_ts'] = df['LOAN_DATE'].astype('int64') // 10 ** 9
        df['loan_ts'].replace(-9223372036, np.nan, inplace=True)
        df['loan_ts'].fillna(df['loan_ts'].median(), inplace=True)
        df.drop(columns=['LOAN_DATE'], inplace=True)
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = [c for c in df.columns if c not in num_cols]
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    for c in cat_cols:
        df[c].fillna(df[c].mode().iloc[0], inplace=True)
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

def compute_metrics(y_true, y_pred, prob, lam_np, lam_pn):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    rec_d = tp / (tp + fn) if tp + fn else 0
    rec_n = tn / (tn + fp) if tn + fp else 0
    prec = tp / (tp + fp) if tp + fp else 0
    bacc = (rec_d + rec_n) / 2
    fm = 2 * prec * rec_d / (prec + rec_d) if prec + rec_d else 0
    gm = np.sqrt(rec_d * rec_n)
    auc = roc_auc_score(y_true, prob)
    cost = np.where(y_true == 1, np.where(y_pred == 1, 0, lam_np), np.where(y_pred == 0, 0, lam_pn)).sum()

    return {'BAcc': bacc, 'FM': fm, 'GM': gm, 'AUC': auc, 'Cost': cost, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


class ThresholdProblem(ElementwiseProblem):
    def __init__(self, y, p, lnp, lpn):
        super().__init__(n_var=2, n_obj=2, n_constr=1, xl=np.array([0., 0.]), xu=np.array([1., 1.]))
        self.y, self.p, self.lnp, self.lpn = y, p, lnp, lpn

    def _evaluate(self, x, out, *_):
        u, v = x
        g = u + v - (1 - EPS)
        a = (self.lpn - v * self.lpn) / (u * self.lnp - v * self.lpn + self.lpn)
        b = (v * self.lpn) / (v * self.lpn + self.lnp - u * self.lnp)
        dec = np.where(self.p >= a, 1, np.where(self.p <= b, 0, 2))
        cost = np.where(dec == 0, np.where(self.y == 1, self.lnp, 0),
                        np.where(dec == 1, np.where(self.y == 0, self.lpn, 0),
                                 np.where(self.y == 1, u * self.lnp, v * self.lpn)))
        out['F'] = [cost.sum(), np.sum(a - b)]
        out['G'] = [g]


def pick_solution(res, y, p, lnp, lpn):
    best_i, best_bnd, best_f1 = None, np.inf, np.inf
    for i, (u, v) in enumerate(res.X):
        a = (lpn - v * lpn) / (u * lnp - v * lpn + lpn)
        b = (v * lpn) / (v * lpn + lnp - u * lnp)
        bnd = np.sum((p > b) & (p < a))
        if bnd < best_bnd or (bnd == best_bnd and res.F[i, 0] < best_f1):
            best_i, best_bnd, best_f1 = i, bnd, res.F[i, 0]
    return res.X[best_i]


# ────────────────  بارگذاری و حذف همبستگی ────────────────
print('Reading data …')
raw = pd.read_excel(DATA_FILE).dropna(axis=1, how='all')
raw['label'] = raw[TARGET_COL].apply(status_to_label)
raw[LOAN_AMT_COL] = pd.to_numeric(raw[LOAN_AMT_COL], errors='coerce').fillna(raw[LOAN_AMT_COL].mean())
raw[INTEREST_RATE_COL] = pd.to_numeric(raw[INTEREST_RATE_COL], errors='coerce').fillna(raw[INTEREST_RATE_COL].mean())
years = pd.to_numeric(raw.get(DURATION_Y_COL, 1), errors='coerce').fillna(1)

num_cols = raw.select_dtypes(include=['number']).columns
raw_num_filt, dropped_cols = drop_high_corr(raw[num_cols])
print(f'Removed {len(dropped_cols)} highly-correlated features (|ρ|>{THRESH_CORR}):')
for c in dropped_cols: print('  -', c)

raw.drop(columns=dropped_cols, inplace=True)
raw[raw_num_filt.columns] = raw_num_filt
raw['interest_cash'] = yearly_interest(raw[LOAN_AMT_COL], raw[INTEREST_RATE_COL], years)

df = preprocess(raw.drop(columns=[TARGET_COL]))
X_full = df.drop(columns=['label', 'interest_cash', LOAN_AMT_COL])
y_full = df['label']


# ────────────────  بلاک تحلیل حساسیت پارامتر NSGA-II ────────────────
def nsga_sensitivity(pop, ngen, y, p, lnp, lpn):
    t0 = time.time()
    res = minimize(ThresholdProblem(y, p, lnp, lpn), NSGA2(pop_size=pop, eliminate_duplicates=True),
                   get_termination('n_gen', ngen), seed=RANDOM_STATE, verbose=False)
    u_star, v_star = pick_solution(res, y, p, lnp, lpn)
    a = (lpn - v_star * lpn) / (u_star * lnp - v_star * lpn + lpn)
    b = (v_star * lpn) / (v_star * lpn + lnp - u_star * lnp)
    n_bnd = np.sum((p > b) & (p < a))
    cost = res.F[:, 0][np.lexsort((res.F[:, 0], res.F[:, 1]))][0]
    return cost, n_bnd, time.time() - t0


print('\nRunning NSGA-II sensitivity study …')
# یک فولد تصادفی برای تحلیل
fold1_tr, fold1_te = next(StratifiedKFold(n_splits=KFoldSize, shuffle=True, random_state=RANDOM_STATE).split(X_full, y_full))
X_tr, X_te = X_full.iloc[fold1_tr], X_full.iloc[fold1_te]
y_tr, y_te = y_full.iloc[fold1_tr], y_full.iloc[fold1_te]

X_bal, y_bal = SMOTE(k_neighbors=SMOTE_K, random_state=RANDOM_STATE).fit_resample(X_tr, y_tr)
prob_model = lgb.LGBMClassifier(**LGB_PARAMS).fit(X_bal, y_bal)
probs_f1 = prob_model.predict_proba(X_te)[:, 1]

lam_NP_f1 = (raw.loc[X_te.index, LOAN_AMT_COL] + raw.loc[X_te.index, 'interest_cash']).values
lam_PN_f1 = raw.loc[X_te.index, 'interest_cash'].values

sens_rows = []
for pop, ngen in param_grid:
    c, b, t = nsga_sensitivity(pop, ngen, y_te.values, probs_f1, lam_NP_f1, lam_PN_f1)
    sens_rows.append([pop, ngen, c, b, t])
    print(f'Pop={pop:3}, Gen={ngen:3} → Cost={c:,.0f}, #BND={b}, time={t:.1f}s')

sens_df = pd.DataFrame(sens_rows, columns=['PopSize', 'NGen', 'DecisionCost', 'NumBND', 'Seconds'])
sens_df.to_csv('nsga_sensitivity.csv', index=False)

# plooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooot
ResultManager().plot_sensitivity(sens_df)
# plooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooot

# ────────────────  تعریف استکینگ مشترک ────────────────
base = [('rf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ('xgb', XGBClassifier(n_estimators=300, random_state=RANDOM_STATE, eval_metric='logloss')),
        ('gbdt', GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ('ert', ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ('ada', AdaBoostClassifier(algorithm='SAMME', random_state=RANDOM_STATE))]

stack_clf = StackingClassifier(estimators=base, final_estimator=LogisticRegression(max_iter=1000),
                               cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), n_jobs=2)

# ──────────────── ۵-Fold Cross-Validation اصلی ────────────────
kf = StratifiedKFold(n_splits=KFoldSize, shuffle=True, random_state=RANDOM_STATE)
metrics, importances = [], []

print('\n===== 5-Fold Cross-Validation =====')
for fold, (tr_idx, te_idx) in enumerate(kf.split(X_full, y_full), 1):
    X_tr, X_te = X_full.iloc[tr_idx], X_full.iloc[te_idx]
    y_tr, y_te = y_full.iloc[tr_idx], y_full.iloc[te_idx]

    X_bal, y_bal = SMOTE(k_neighbors=SMOTE_K, random_state=RANDOM_STATE).fit_resample(X_tr, y_tr)
    lgb_clf = lgb.LGBMClassifier(**LGB_PARAMS).fit(X_bal, y_bal)
    prob_te = lgb_clf.predict_proba(X_te)[:, 1]

    importances.append(
        pd.DataFrame({'feature': X_bal.columns, 'importance': lgb_clf.feature_importances_, 'fold': fold}))

    lam_NP = (raw.loc[X_te.index, LOAN_AMT_COL] + raw.loc[X_te.index, 'interest_cash']).values
    lam_PN = raw.loc[X_te.index, 'interest_cash'].values

    res = minimize(ThresholdProblem(y_te.values, prob_te, lam_NP, lam_PN),
                   NSGA2(pop_size=NSGA_POP, eliminate_duplicates=True), get_termination('n_gen', NSGA_GEN),
                   seed=RANDOM_STATE, verbose=False)
    ResultManager().save_pareto_plot(res, fold)

    u_star, v_star = pick_solution(res, y_te.values, prob_te, lam_NP, lam_PN)

    alpha = (lam_PN - v_star * lam_PN) / (u_star * lam_NP - v_star * lam_PN + lam_PN)
    beta = (v_star * lam_PN) / (v_star * lam_PN + lam_NP - u_star * lam_NP)

    region = np.where(prob_te >= alpha, 'POS', np.where(prob_te <= beta, 'NEG', 'BND'))

    stack_clf.fit(X_tr, y_tr)
    final_pred = np.empty_like(y_te.values)
    final_pred[region == 'POS'] = 1
    final_pred[region == 'NEG'] = 0
    bnd_idx = np.where(region == 'BND')[0]
    if bnd_idx.size:
        final_pred[bnd_idx] = stack_clf.predict(X_te.iloc[bnd_idx])

    tn, fp, fn, tp = confusion_matrix(y_te, final_pred, labels=[0, 1]).ravel()
    rec_d = tp / (tp + fn) if tp + fn else 0
    rec_n = tn / (tn + fp) if tn + fp else 0
    prec = tp / (tp + fp) if tp + fp else 0
    BAcc = (rec_d + rec_n) / 2
    FM = 2 * prec * rec_d / (prec + rec_d) if prec + rec_d else 0
    GM = np.sqrt(rec_d * rec_n)
    AUC = roc_auc_score(y_te, prob_te)

    cost_before = np.where(region == 'POS', np.where(y_te == 0, lam_PN, 0),
                           np.where(region == 'NEG', np.where(y_te == 1, lam_NP, 0),
                                    np.where(y_te == 1, u_star * lam_NP, v_star * lam_PN)))
    Cost_before = cost_before.sum()

    cost_after = np.where(final_pred == 0, np.where(y_te == 1, lam_NP, 0), np.where(y_te == 0, lam_PN, 0))
    Cost_after = cost_after.sum()

    metrics.append([BAcc, GM, FM, AUC, Cost_after, tp, tn, fp, fn])

    print(f'\033[92mFold {fold}:  BAcc={BAcc:.4f}  GM={GM:.4f}  '
          f'FM={FM:.4f}  AUC={AUC:.4f}  '
          f'Cost_before={Cost_before:,.0f}  Cost_after={Cost_after:,.0f}\033[0m')

# ────────────────  خروجی نهایی جدول‌ها ────────────────
m = np.array(metrics)
print('\n—— 5-Fold Mean ± Std ——')
for name, col in zip(['BAcc', 'GM', 'FM', 'AUC', 'Cost'], m.T):
    print(f'{name}: {col.mean():.4f} ± {col.std():.4f}')

imp_df = (pd.concat(importances).groupby('feature')['importance'].agg(['mean', 'std']).sort_values('mean',
                                                                                                   ascending=False).head(
    TOP_N_FEATS).reset_index())
imp_df.to_csv('top20_feature_importance.csv', index=False)
print('Feature-importance table → top20_feature_importance.csv')

# ────────────────  Baseline single-stage models ────────────────
baseline_models = {'RandomForest': RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
                   'XGBoost': XGBClassifier(n_estimators=400, random_state=RANDOM_STATE, eval_metric='logloss'),
                   'SVM-RBF': (lambda: __import__('sklearn.svm', fromlist=['SVC'])
                               .SVC(probability=True, kernel='rbf', C=2,gamma='scale',random_state=RANDOM_STATE))()
                   }

# ارزیابی یکسان مدل‌ها با تابع مشترک
results = []
kf = StratifiedKFold(n_splits=KFoldSize, shuffle=True, random_state=RANDOM_STATE)
for fold, (tr_idx, te_idx) in enumerate(kf.split(X_full, y_full), 1):
    X_tr, X_te = X_full.iloc[tr_idx], X_full.iloc[te_idx]
    y_tr, y_te = y_full.iloc[tr_idx], y_full.iloc[te_idx]
    lam_NP = (raw.loc[te_idx, LOAN_AMT_COL] + raw.loc[te_idx, 'interest_cash']).values
    lam_PN = raw.loc[te_idx, 'interest_cash'].values
    for name, clf in baseline_models.items():
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)
        m = compute_metrics(y_te, pred, prob, lam_NP, lam_PN)
        m.update({'Model': name, 'Fold': fold})
        results.append(m)

# خلاصه میانگین و انحراف معیار
res_df = pd.DataFrame(results)
summary = res_df.groupby('Model').agg(['mean', 'std']).reset_index()
# مرتب‌سازی بر حسب میانگین هزینه
summary = summary.sort_values(('Cost', 'mean'))
# ۱) ستون‌های مورد نظر برای ارزیابی
prop_cols = ['BAcc', 'GM', 'FM', 'AUC', 'Cost', 'TP', 'TN', 'FP', 'FN']

# ۲) ساخت خلاصهٔ مدل پیشنهادی فقط با میانگین
prop_df = pd.DataFrame(metrics, columns=prop_cols)
prop_means = prop_df.mean()
prop_row = {'Method': 'Proposed'}
for metric in prop_cols:
    prop_row[metric] = prop_means[metric]

# ۳) ساخت ردیف‌های مدل‌های پایه فقط با میانگین
base_rows = []
for model in res_df['Model'].unique():
    df_model = res_df[res_df['Model'] == model]
    row = {'Method': model}
    for metric in prop_cols:
        if metric in df_model.columns:
            row[metric] = df_model[metric].mean()
        else:
            row[metric] = None
    base_rows.append(row)

# ۴) ادغام همه در یک جدول نهایی
final_table = pd.DataFrame(base_rows + [prop_row])

# ۵) مرتب‌سازی و نمایش
final_table = final_table.sort_values(by='BAcc', ascending=False)

# تبدیل ستون‌های صحیح به عدد صحیح بدون اعشار
for col in ['TP', 'TN', 'FP', 'FN']:
    final_table[col] = final_table[col].round().astype(int)

# تبدیل ستون Cost به رشته با کاما جداکننده و بدون اعشار
final_table['Cost'] = final_table['Cost'].round().apply(lambda x: f"{int(x):,}")

# انتخاب ترتیب نهایی ستون‌ها
final_table = final_table[['Method', 'BAcc', 'AUC', 'FM', 'GM', 'Cost', 'TP', 'TN', 'FP', 'FN']]

# چاپ نهایی با رنگ آبی
print('\n—— 📊 جدول مقایسه‌ای مدل‌ها ——')
print('\033[94m' + final_table.to_string(index=False) + '\033[0m')

# ۶) ذخیره فایل نهایی
final_table.to_csv('comparison_table.csv', index=False)
print('✅ جدول نهایی ذخیره شد → comparison_table.csv')

# ذخیره خروجی
res_df.to_csv('baseline_models_metrics.csv', index=False)
print('Baseline metrics saved → baseline_models_metrics.csv')
