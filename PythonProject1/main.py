import random
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

# در pymoo نسخه‌های جدید مسیر NSGA-II تغییر کرده است
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from dataHandler.LoanDataHandler import LoanDataHandler
from processor.LoanPreprocessor import LoanPreprocessor
from repository.LoanRepository import LoanRepository


# تابع پیش‌پردازش داده از دیتابیس
def preProcessDataFromDB():
    repository = LoanRepository()
    preprocessor = LoanPreprocessor(imputation_strategy="median")
    data_handler = LoanDataHandler(repository, preprocessor)
    return data_handler.load_and_process_data(limit_records=5000)


# آموزش مدل LightGBM و دریافت احتمال نکول
def trainLGBMModel(X_train, y_train, X_test):
    lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    lgbm.fit(X_train, y_train)
    p_pred = lgbm.predict_proba(X_test)[:, 1]
    return p_pred, lgbm


# محاسبه ضررهای مالی از اطلاعات جریان نقدی
def computeLosses(cash_flow_info):
    principal = cash_flow_info['approval_amount'].values
    interest = cash_flow_info['interest_amount'].values
    lPN_arr = [interest[i] for i in range(len(principal))]
    lNP_arr = [principal[i] + interest[i] for i in range(len(principal))]
    return np.array(lPN_arr), np.array(lNP_arr)


# تعریف تابع هدف بهینه‌سازی با pymoo (NSGA-II)
class ObjectiveProblem(Problem):
    def __init__(self, p_pred, lPN_arr, lNP_arr):
        self.p_pred = p_pred
        self.lPN_arr = lPN_arr
        self.lNP_arr = lNP_arr
        super().__init__(n_var=2, n_obj=2, n_constr=1, xl=np.array([0, 0]), xu=np.array([1, 1]))

    def _evaluate(self, X, out, *args, **kwargs):
        n_ind = X.shape[0]
        f1 = np.zeros(n_ind)  # هزینه تصمیم
        f2 = np.zeros(n_ind)  # اندازه ناحیه مرزی
        for i in range(n_ind):
            u = X[i, 0]
            v = X[i, 1]
            cost_total = 0.0
            boundary_sum = 0.0
            for j in range(len(self.p_pred)):
                pn = self.lPN_arr[j]
                np_val = self.lNP_arr[j]
                bp = u * np_val
                bn = v * pn
                numerator_alpha = pn - bn
                denominator_alpha = numerator_alpha + bp
                alpha = 1.0 if denominator_alpha == 0 else numerator_alpha / denominator_alpha
                numerator_beta = bn
                denominator_beta = bn + (np_val - bp)
                beta = 0.0 if denominator_beta == 0 else numerator_beta / denominator_beta
                p_val = self.p_pred[j]
                if p_val >= alpha:
                    cost_local = pn * (1 - p_val)
                elif p_val <= beta:
                    cost_local = np_val * p_val
                else:
                    cost_local = bp * p_val + bn * (1 - p_val)
                cost_total += cost_local
                boundary_sum += (alpha - beta)
            f1[i] = cost_total
            f2[i] = boundary_sum
        # قید: u + v <= 1
        g = X[:, 0] + X[:, 1] - 1.0
        out["F"] = np.column_stack([f1, f2])
        out["G"] = g.reshape(-1, 1)


# بهینه‌سازی u و v با استفاده از NSGA-II در pymoo
def nsga2_find_uv(p_pred, lPN_arr, lNP_arr, pop_size=20, generations=10):
    problem = ObjectiveProblem(p_pred, lPN_arr, lNP_arr)
    algorithm = NSGA2(pop_size=pop_size)
    termination = ('n_gen', generations)
    res = minimize(problem, algorithm, termination, seed=1, verbose=False)
    F = res.F
    # انتخاب بهترین جواب بر اساس کمینه‌سازی f1 و در صورت تساوی، f2
    best_idx = 0
    best_f1 = F[0, 0]
    best_f2 = F[0, 1]
    for i in range(len(F)):
        if F[i, 0] < best_f1 or (abs(F[i, 0] - best_f1) < 1e-9 and F[i, 1] < best_f2):
            best_f1 = F[i, 0]
            best_f2 = F[i, 1]
            best_idx = i
    return res.X[best_idx, 0], res.X[best_idx, 1]


# اعمال تصمیم سه‌طرفه بر اساس آستانه‌های به‌دست آمده
def applyThreeWayDecision(p_pred, lPN_arr, lNP_arr, u, v):
    labels = []
    boundary_indices = []
    for i in range(len(p_pred)):
        p_val = p_pred[i]
        pn = lPN_arr[i]
        np_val = lNP_arr[i]
        bp = u * np_val
        bn = v * pn
        numerator_alpha = pn - bn
        denominator_alpha = numerator_alpha + bp
        alpha = 1.0 if denominator_alpha == 0 else numerator_alpha / denominator_alpha
        numerator_beta = bn
        denominator_beta = bn + (np_val - bp)
        beta = 0.0 if denominator_beta == 0 else numerator_beta / denominator_beta
        if p_val >= alpha:
            labels.append(1)
        elif p_val <= beta:
            labels.append(0)
        else:
            labels.append(-1)
            boundary_indices.append(i)
    return np.array(labels), boundary_indices


# آموزش مدل استکینگ برای نمونه‌های مرزی
def trainStacking(X_train, y_train, base_models, meta_model):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    level_one_output = np.zeros((len(X_train), len(base_models)))
    X_arr = np.array(X_train)
    y_arr = np.array(y_train)
    for idx, model in enumerate(base_models):
        for train_idx, val_idx in kf.split(X_arr):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_val)[:, 1]
            level_one_output[val_idx, idx] = preds
    meta_model.fit(level_one_output, y_arr)
    return base_models, meta_model


def predictStacking(X_test, base_models, meta_model):
    base_preds = np.zeros((len(X_test), len(base_models)))
    for idx, model in enumerate(base_models):
        base_preds[:, idx] = model.predict_proba(X_test)[:, 1]
    return meta_model.predict(base_preds)


# اجرای کل فرآیند در یک اسکریپت
if __name__ == "__main__":
    # دریافت و پیش‌پردازش داده
    X_train_res, y_train_res, X_test, y_test = preProcessDataFromDB()
    # آموزش مدل LightGBM و دریافت احتمال نکول
    p_pred_test, _ = trainLGBMModel(X_train_res, y_train_res, X_test)
    # محاسبه ضررهای مالی از اطلاعات جریان نقدی
    cash_flow = X_test[['approval_amount', 'interest_amount']]
    lPN_arr_test, lNP_arr_test = computeLosses(cash_flow)
    # بهینه‌سازی u و v با pymoo
    best_u, best_v = nsga2_find_uv(p_pred_test, lPN_arr_test, lNP_arr_test, pop_size=20, generations=10)
    # اعمال تصمیم سه‌طرفه بر اساس آستانه‌های به‌دست آمده
    twd_labels, boundary_indices = applyThreeWayDecision(p_pred_test, lPN_arr_test, lNP_arr_test, best_u, best_v)
    # استخراج نمونه‌های مرزی
    X_test_boundary = X_test.iloc[boundary_indices]
    y_test_boundary = y_test.iloc[boundary_indices]
    # آموزش استکینگ برای نمونه‌های مرزی
    base_models = [
        LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=0),
        RandomForestClassifier(n_estimators=100, random_state=0),
        XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=0),
        GradientBoostingClassifier(n_estimators=100, random_state=0),
        ExtraTreesClassifier(n_estimators=100, random_state=0),
        AdaBoostClassifier(n_estimators=100, random_state=0)
    ]
    meta_model = LogisticRegression()
    base_models, meta_model = trainStacking(X_train_res, y_train_res, base_models, meta_model)
    # پیش‌بینی نهایی برای نمونه‌های مرزی با مدل استکینگ
    y_pred_boundary = predictStacking(X_test_boundary, base_models, meta_model)
    for i, idx in enumerate(boundary_indices):
        twd_labels[idx] = y_pred_boundary[i]
    # ارزیابی عملکرد نهایی
    cm = confusion_matrix(y_test, twd_labels)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    balanced_acc = 0
    if (TP + FP) != 0 and (TN + FN) != 0:
        balanced_acc = ((TP / (TP + FP)) + (TN / (TN + FN))) / 2.0
    auc_score = roc_auc_score(y_test, twd_labels)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    fmeasure = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    gmean = (sensitivity * specificity) ** 0.5
    total_cost = 0
    y_test_arr = np.array(y_test)
    twd_labels_arr = np.array(twd_labels)
    for i in range(len(y_test_arr)):
        if y_test_arr[i] == 1 and twd_labels_arr[i] == 0:
            total_cost += lNP_arr_test[i]
        elif y_test_arr[i] == 0 and twd_labels_arr[i] == 1:
            total_cost += lPN_arr_test[i]
    print("Balanced Accuracy:", balanced_acc)
    print("AUC:", auc_score)
    print("F-Measure:", fmeasure)
    print("G-Mean:", gmean)
    print("Decision Cost:", total_cost)
    print("Confusion Matrix (TN, FP, FN, TP):", TN, FP, FN, TP)
