import random

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

from dataHandler.LoanDataHandler import LoanDataHandler
from processor.LoanPreprocessor import LoanPreprocessor
from repository.LoanRepository import LoanRepository

# λ ضرر مالی

# ------------------------ توابع پیش‌پردازش داده ------------------------
def preProcessDataFromDB():
    repository = LoanRepository()  # نیازی به پارامتر ندارد ✅
    preprocessor = LoanPreprocessor(imputation_strategy="median")
    data_handler = LoanDataHandler(repository, preprocessor)
    return data_handler.load_and_process_data(limit_records=5000)

# ------------------------ توابع مدل LightGBM ------------------------
def trainLGBMModel(X_train, y_train, X_test):
    """
    مدل LightGBM را با داده‌های آموزشی آموزش می‌دهد و سپس برای داده‌های تست،
    احتمال پیش‌فرض (یعنی احتمال نکول) را محاسبه و خروجی می‌دهد.
    """
    lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    print("قبل از آموزش مدل، نوع داده‌های X_train:")
    print(X_train.dtypes)
    lgbm.fit(X_train, y_train)
    print("بعد از آموزش مدل.")
    p_pred = lgbm.predict_proba(X_test)[:, 1]
    return p_pred, lgbm


# ------------------------ توابع محاسبه ضرر ------------------------
def computeLosses(cash_flow_info):
    """
    براساس اطلاعات جریان نقدی، ضرر PN (سود از دست رفته) و ضرر NP (اصل + سود) را محاسبه می‌کند.
    در این مثال، ستون‌های 'approval_amount' و 'interest_amount' به ترتیب نقش اصل و سود را دارند.
    """
    principal = cash_flow_info['approval_amount'].values
    interest = cash_flow_info['interest_amount'].values
    λPN_arr = []
    λNP_arr = []
    for i in range(len(principal)):
        pn_val = interest[i]
        np_val = principal[i] + interest[i]
        λPN_arr.append(pn_val)
        λNP_arr.append(np_val)
    return np.array(λPN_arr), np.array(λNP_arr)

# ------------------------ توابع محاسبه تابع هدف و NSGA-II ------------------------
def computeObjective(u, v, p_pred, λPN_arr, λNP_arr):
    """
    برای هر نمونه آستانه‌های α و β را محاسبه کرده و هزینه تصمیم‌گیری و اندازه ناحیه مرزی را برمی‌گرداند.
    هدف: کمینه کردن مجموع هزینه تصمیم و اندازه ناحیه مرزی.
    """
    cost_total = 0.0
    boundary_sum = 0.0
    for i in range(len(p_pred)):
        pn = λPN_arr[i]
        np_ = λNP_arr[i]
        bp = u * np_
        bn = v * pn
        numerator_alpha = (pn - bn)
        denominator_alpha = numerator_alpha + (bp - 0)
        alpha_i = 1.0 if denominator_alpha == 0 else numerator_alpha / denominator_alpha
        numerator_beta = bn
        denominator_beta = bn + (np_ - bp)
        beta_i = 0.0 if denominator_beta == 0 else numerator_beta / denominator_beta
        p_val = p_pred[i]
        if p_val >= alpha_i:
            cost_local = pn * (1 - p_val)
        elif p_val <= beta_i:
            cost_local = np_ * p_val
        else:
            cost_local = (bp * p_val) + (bn * (1 - p_val))
        cost_total += cost_local
        boundary_sum += (alpha_i - beta_i)
    return cost_total, boundary_sum


def nsga2_find_uv(p_pred, λPN_arr, λNP_arr, population_size=30, generations=20):
    """
    پیاده‌سازی ساده NSGA-II جهت یافتن مقادیر بهینه u و v که تابع هدف (هزینه کلی و اندازه ناحیه مرزی) را کمینه کند.
    """
    population = []
    for _ in range(population_size):
        u = random.uniform(0, 1)
        v = random.uniform(0, 1)
        if u + v > 1:
            v = 1 - u
        population.append((u, v))

    for gen in range(generations):
        evaluated_pop = []
        for (u, v) in population:
            cost_val, boundary_val = computeObjective(u, v, p_pred, λPN_arr, λNP_arr)
            evaluated_pop.append((u, v, cost_val, boundary_val))
        evaluated_pop.sort(key=lambda x: (x[2], x[3]))
        top_half = evaluated_pop[:population_size // 2]
        new_population = []
        for (u, v, cost_val, bd_val) in top_half:
            u_new = u + random.uniform(-0.01, 0.01)
            v_new = v + random.uniform(-0.01, 0.01)
            u_new = max(u_new, 0)
            v_new = max(v_new, 0)
            if u_new + v_new > 1:
                v_new = 1 - u_new
            new_population.append((u_new, v_new))
        population = new_population
        while len(population) < population_size:
            ru = random.uniform(0, 1)
            rv = random.uniform(0, 1)
            if ru + rv > 1:
                rv = 1 - ru
            population.append((ru, rv))

    best_val = float('inf')
    best_bound = float('inf')
    best_uv = (0, 0)
    for (u, v) in population:
        cost_val, boundary_val = computeObjective(u, v, p_pred, λPN_arr, λNP_arr)
        if cost_val < best_val:
            best_val = cost_val
            best_bound = boundary_val
            best_uv = (u, v)
        elif abs(cost_val - best_val) < 1e-9 and boundary_val < best_bound:
            best_uv = (u, v)
            best_bound = boundary_val
    return best_uv


def applyThreeWayDecision(p_pred, λPN_arr, λNP_arr, u, v):
    """
    بر اساس مقادیر u و v، آستانه‌های سه‌طرفه (α و β) محاسبه شده و بر اساس آن‌ها:
    - اگر p >= α: برچسب ۱ (پذیرش/نکول)
    - اگر p <= β: برچسب ۰ (رد/غیر نکول)
    - در غیر این صورت: برچسب -1 (ناحیه مرزی)
    همچنین اندیس نمونه‌های مرزی جمع‌آوری می‌شود.
    """
    labels = []
    boundary_indices = []
    for i in range(len(p_pred)):
        p_val = p_pred[i]
        pn = λPN_arr[i]
        np_ = λNP_arr[i]
        bp = u * np_
        bn = v * pn
        numerator_alpha = (pn - bn)
        denominator_alpha = numerator_alpha + (bp - 0)
        alpha_i = 1.0 if denominator_alpha == 0 else numerator_alpha / denominator_alpha
        numerator_beta = bn
        denominator_beta = bn + (np_ - bp)
        beta_i = 0.0 if denominator_beta == 0 else numerator_beta / denominator_beta
        if p_val >= alpha_i:
            labels.append(1)
        elif p_val <= beta_i:
            labels.append(0)
        else:
            labels.append(-1)
            boundary_indices.append(i)
    return np.array(labels), boundary_indices


# ------------------------ توابع مدل استکینگ ------------------------
def trainStacking(X_train, y_train, base_models, meta_model):
    """
    مدل استکینگ ساده:
    - در لایه اول، چند مدل پایه با استفاده از KFold آموزش داده می‌شوند.
    - خروجی‌های احتمال پیش‌بینی این مدل‌ها به عنوان ویژگی ورودی برای آموزش مدل متا استفاده می‌شود.
    """
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
    """
    برای داده‌های جدید X_test، خروجی هر مدل پایه محاسبه شده و سپس مدل متا بر اساس آن‌ها
    برچسب نهایی را پیش‌بینی می‌کند.
    """
    base_preds = np.zeros((len(X_test), len(base_models)))
    for idx, model in enumerate(base_models):
        base_preds[:, idx] = model.predict_proba(X_test)[:, 1]
    final_preds = meta_model.predict(base_preds)
    return final_preds


# ------------------------ بخش اصلی (Main) ------------------------
if __name__ == "__main__":
    # مرحله اول: دریافت و پیش‌پردازش داده‌ها از دیتابیس
    X_train_res, y_train_res, X_test, y_test = preProcessDataFromDB()

    # مرحله دوم: آموزش مدل LightGBM و گرفتن احتمال پیش‌فرض برای داده‌های تست
    p_pred_test, lgbm_model = trainLGBMModel(X_train_res, y_train_res, X_test)

    # استفاده از اطلاعات جریان نقدی واقعی موجود در X_test؛ فرض بر این است که ستون‌های 'approval_amount' و 'interest_amount' موجودند.
    data_test_cashflow = X_test[['approval_amount', 'interest_amount']]

    # مرحله سوم: محاسبه ضررهای پایه PN و NP بر اساس اطلاعات جریان نقدی
    λPN_arr_test, λNP_arr_test = computeLosses(data_test_cashflow)

    # az λPN_arr_test, λNP_arr_test به عنوان وردی تابع هدف استفاده میکنیم

    # مرحله چهارم: استفاده از NSGA-II برای یافتن بهینه u و v
    best_u, best_v = nsga2_find_uv(p_pred_test, λPN_arr_test, λNP_arr_test,
                                   population_size=20, generations=10)

    # مرحله پنجم: اعمال تصمیم سه‌طرفه با استفاده از آستانه‌های به‌دست آمده
    twd_labels, boundary_indices = applyThreeWayDecision(p_pred_test, λPN_arr_test, λNP_arr_test,
                                                         best_u, best_v)

    # جداکردن نمونه‌های مرزی برای تصمیم‌گیری نهایی
    X_test_boundary = X_test.iloc[boundary_indices]
    y_test_boundary = y_test.iloc[boundary_indices]

    # مرحله ششم: آموزش مدل استکینگ برای نمونه‌های مرزی
    base_models = [
        LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=0),
        RandomForestClassifier(n_estimators=100, random_state=0),
        XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=0),
        GradientBoostingClassifier(n_estimators=100, random_state=0),
        ExtraTreesClassifier(n_estimators=100, random_state=0),
        AdaBoostClassifier(n_estimators=100, random_state=0)
    ]
    meta_model = LogisticRegression()
    base_models_trained, meta_model_trained = trainStacking(X_train_res, y_train_res, base_models, meta_model)

    # مرحله هفتم: پیش‌بینی نهایی برای نمونه‌های مرزی با استفاده از مدل استکینگ
    y_pred_boundary = predictStacking(X_test_boundary, base_models_trained, meta_model_trained)

    # جایگزینی برچسب نهایی نمونه‌های مرزی در آرایه تصمیم سه‌طرفه
    for i, idx in enumerate(boundary_indices):
        twd_labels[idx] = y_pred_boundary[i]

    # مرحله هشتم: ارزیابی نهایی عملکرد مدل
    cm = confusion_matrix(y_test, twd_labels)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    balanced_acc = 0
    if (TP + FP) != 0 and (TN + FN) != 0:
        balanced_acc = ((TP / (TP + FP)) + (TN / (TN + FN))) / 2.0

    auc_score = roc_auc_score(y_test, twd_labels)

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    fmeasure = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    gmean = 0
    if (TP + FN) != 0 and (TN + FP) != 0:
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        gmean = (sensitivity * specificity) ** 0.5

    total_cost = 0.0
    y_test_arr = np.array(y_test)
    twd_labels_arr = np.array(twd_labels)
    for i in range(len(y_test_arr)):
        if y_test_arr[i] == 1 and twd_labels_arr[i] == 0:
            total_cost += λNP_arr_test[i]
        elif y_test_arr[i] == 0 and twd_labels_arr[i] == 1:
            total_cost += λPN_arr_test[i]

    print("Balanced Accuracy:", balanced_acc)
    print("AUC:", auc_score)
    print("F-Measure:", fmeasure)
    print("G-Mean:", gmean)
    print("Decision Cost:", total_cost)
    print("Confusion Matrix (TN, FP, FN, TP):", TN, FP, FN, TP)
