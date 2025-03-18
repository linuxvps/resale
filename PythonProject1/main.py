import os
from decimal import Decimal

import numpy as np
# ------------------------ وارد کردن کتابخانه‌های مدل‌سازی و ارزیابی ------------------------
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, classification_report, confusion_matrix, \
    precision_score, recall_score, f1_score

# ------------------------ وارد کردن کتابخانه‌های مربوط به دریافت و پیش‌پردازش داده ------------------------
from dataHandler.LoanDataHandler import LoanDataHandler
from processor.LoanPreprocessor import LoanPreprocessor, protected_columns
from repository.LoanRepository import LoanRepository

# ------------------------ وارد کردن توابع و کلاس‌های مربوط به بهینه‌سازی ------------------------
from optimization.ObjectiveProblem import optimize_threshold_scales


# ------------------------ دریافت و پیش‌پردازش داده ------------------------
def pre_process_data_from_db():
    repository = LoanRepository()
    preprocessor = LoanPreprocessor(imputation_strategy="median")
    data_handler = LoanDataHandler(repository, preprocessor)
    return data_handler.load_and_process_data(limit_records=50000)


# ------------------------ آموزش مدل LightGBM ------------------------
def train_lgbm_model(x_train, y_train, x_test):
    lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    print("شروع آموزش مدل LightGBM...")
    lgbm.fit(x_train, y_train)
    print("آموزش مدل به پایان رسید.")
    # دریافت احتمال نکول (کلاس مثبت)
    p_pred = lgbm.predict_proba(x_test)[:, 1]
    return p_pred


# ------------------------ محاسبه ضررهای مالی از اطلاعات جریان نقدی ------------------------
def compute_losses(cash_flow_info):
    principal = cash_flow_info['approval_amount'].values
    interest = cash_flow_info['interest_amount'].values
    # loss_PN: ضرر زمانی که یک نمونه غیر نکول (non-default) به اشتباه به عنوان نکول تشخیص داده شود (فقط سود از دست رفته)
    # loss_NP: ضرر زمانی که یک نمونه نکول (default) به اشتباه به عنوان غیر نکول تشخیص داده شود (اصل + سود)
    loss_PN = interest
    loss_NP = principal + interest
    return loss_PN, loss_NP


# ------------------------ اعمال تصمیم سه‌طرفه ------------------------
def apply_three_way_decision(p_pred, loss_pn_arr, loss_np_arr, u, v):
    # محاسبه bp و bn با استفاده از u و v
    bp = u * loss_np_arr
    bn = v * loss_pn_arr

    # محاسبه آستانه مثبت (alpha)
    numerator_alpha = loss_pn_arr - bn
    denom_alpha = numerator_alpha + bp
    alpha = np.where(denom_alpha == 0, 1.0, numerator_alpha / denom_alpha)

    # محاسبه آستانه منفی (beta)
    numerator_beta = bn
    denom_beta = bn + (loss_np_arr - bp)
    beta = np.where(denom_beta == 0, 0.0, numerator_beta / denom_beta)

    # تصمیم‌گیری سه‌طرفه:
    # اگر احتمال پیش‌بینی ≥ alpha → برچسب 1 (قبول)
    # اگر احتمال پیش‌بینی ≤ beta → برچسب 0 (رد)
    # در غیر این صورت → برچسب -1 (تصمیم تأخیری)
    labels = np.where(p_pred >= alpha, 1, np.where(p_pred <= beta, 0, -1))
    boundary_indices = np.where(labels == -1)[0]
    return labels, boundary_indices


# ------------------------ ارزیابی عملکرد ------------------------
def evaluate_performance(y_true, y_pred, loss_pn_arr, loss_np_arr):
    print("\n" * 3)
    print("ارزیابی عملکرد کلی مدل")
    print("\n" * 3)

    ba = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cost_decision = np.sum(loss_np_arr[(y_true == 1) & (y_pred == 0)]) + np.sum(
        loss_pn_arr[(y_true == 0) & (y_pred == 1)])

    print("Balanced Accuracy:", ba)
    print("AUC:", auc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    print("Decision Cost:", cost_decision)


# ------------------------ اجرای کل فرآیند ------------------------
if __name__ == "__main__":
    # تنظیم تعداد هسته‌های CPU (در صورت نیاز)
    os.environ["LOKY_MAX_CPU_COUNT"] = "8"

    # ۱. دریافت و پیش‌پردازش داده
    x_train_res, y_train_res, x_test, y_test = pre_process_data_from_db()

    # ۲. آموزش مدل LightGBM برای دریافت احتمال نکول
    p_pred_test = train_lgbm_model(x_train_res, y_train_res, x_test)

    # ۳. محاسبه ضررهای مالی از اطلاعات جریان نقدی (استفاده از ستون‌های محافظت‌شده)
    cash_flow = x_test[protected_columns]
    loss_pn_arr_test, loss_np_arr_test = compute_losses(cash_flow)

    # ۴. بهینه‌سازی پارامترهای u و v با استفاده از NSGA-II
    best_u, best_v = optimize_threshold_scales(p_pred_test, loss_pn_arr_test, loss_np_arr_test, population_size=20, num_generations=10)
    print("بهترین u:", Decimal(best_u))
    print("بهترین v:", Decimal(best_v))

    # ۵. اعمال تصمیم سه‌طرفه
    twd_labels, boundary_indices = apply_three_way_decision(p_pred_test, loss_pn_arr_test, loss_np_arr_test, best_u,
                                                            best_v)

    # ۶. بهبود تصمیم‌های نمونه‌های مرزی با استفاده از مدل‌های Ensemble
    bagging_model = BaggingClassifier(
        estimator=ExtraTreesClassifier(n_estimators=100, random_state=42),
        n_estimators=10,  # تعداد نسخه‌های مدل پایه که می‌خواهیم آموزش ببینند
        random_state=42
    )
    bagging_model.fit(x_train_res, y_train_res)

    # پیش‌بینی بر روی نمونه‌های مرزی و جایگزینی تصمیم تأخیری (-1) با تصمیم مدل Ensemble
    x_test_boundary = x_test.iloc[boundary_indices]
    y_pred_boundary = bagging_model.predict(x_test_boundary)
    twd_labels[boundary_indices] = y_pred_boundary

    # ۸. ارزیابی عملکرد کلی مدل
    evaluate_performance(np.array(y_test), np.array(twd_labels), loss_pn_arr_test, loss_np_arr_test)
