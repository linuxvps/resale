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
    loan_repository = LoanRepository()
    loan_preprocessor = LoanPreprocessor(imputation_strategy="median")
    loan_data_handler = LoanDataHandler(loan_repository, loan_preprocessor)
    return loan_data_handler.load_and_process_data(limit_records=50000)


# ------------------------ آموزش مدل LightGBM ------------------------
def train_lightgbm_model(x_train, y_train, x_test):
    lightgbm_classifier = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    print("شروع آموزش مدل LightGBM...")
    lightgbm_classifier.fit(x_train, y_train)
    print("آموزش مدل به پایان رسید.")
    # دریافت احتمال نکول (کلاس مثبت)
    # تمام سطرها را انتخاب کن، ولی فقط ستون شماره ۱ را بردار.
    predicted_probabilities = lightgbm_classifier.predict_proba(x_test)[:, 1]
    return predicted_probabilities


# ------------------------ محاسبه ضررهای مالی از اطلاعات جریان نقدی ------------------------
def compute_financial_losses(cash_flow_info):
    principal_amount = cash_flow_info['approval_amount'].values
    interest_amount = cash_flow_info['interest_amount'].values
    # ضرر زمانی که یک نمونه غیر نکول (non-default) به اشتباه به عنوان نکول تشخیص داده شود (فقط سود از دست رفته)
    false_positive_loss = interest_amount
    # ضرر زمانی که یک نمونه نکول (default) به اشتباه به عنوان غیر نکول تشخیص داده شود (اصل + سود)
    false_negative_loss = principal_amount + interest_amount
    return false_positive_loss, false_negative_loss


# ------------------------ اعمال تصمیم سه‌طرفه ------------------------
def apply_three_way_decision(predicted_probabilities, false_positive_loss, false_negative_loss, upper_threshold_scale, lower_threshold_scale):
    # محاسبه ضرایب تنظیمی برای آستانه‌های تصمیم‌گیری
    boundary_penalty_positive = upper_threshold_scale * false_negative_loss
    boundary_penalty_negative = lower_threshold_scale * false_positive_loss

    # محاسبه آستانه مثبت (alpha)
    numerator_alpha = false_positive_loss - boundary_penalty_negative
    denominator_alpha = numerator_alpha + boundary_penalty_positive
    alpha_threshold = np.where(denominator_alpha == 0, 1.0, numerator_alpha / denominator_alpha)

    # محاسبه آستانه منفی (beta)
    numerator_beta = boundary_penalty_negative
    denominator_beta = boundary_penalty_negative + (false_negative_loss - boundary_penalty_positive)
    beta_threshold = np.where(denominator_beta == 0, 0.0, numerator_beta / denominator_beta)

    # تصمیم‌گیری سه‌طرفه
    three_way_decision_labels = np.where(predicted_probabilities >= alpha_threshold, 1, np.where(predicted_probabilities <= beta_threshold, 0, -1))
    uncertain_boundary_sample_indices = np.where(three_way_decision_labels == -1)[0]
    return three_way_decision_labels, uncertain_boundary_sample_indices


# ------------------------ ارزیابی عملکرد ------------------------
def evaluate_model_performance(true_labels, predicted_labels, false_positive_loss, false_negative_loss):
    print("\n" * 3)
    print("ارزیابی عملکرد کلی مدل")
    print("\n" * 3)

    balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)
    area_under_curve = roc_auc_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    classification_rep = classification_report(true_labels, predicted_labels)
    decision_cost = np.sum(false_negative_loss[(true_labels == 1) & (predicted_labels == 0)]) + np.sum(
        false_positive_loss[(true_labels == 0) & (predicted_labels == 1)])

    print("Balanced Accuracy:", balanced_accuracy)
    print("AUC:", area_under_curve)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", confusion_mat)
    print("Classification Report:\n", classification_rep)
    print("Decision Cost:", decision_cost)


# ------------------------ اجرای کل فرآیند ------------------------
if __name__ == "__main__":
    # تنظیم تعداد هسته‌های CPU (در صورت نیاز)
    os.environ["LOKY_MAX_CPU_COUNT"] = "8"

    # دریافت و پیش‌پردازش داده
    x_train_resampled, y_train_resampled, x_test, y_test = pre_process_data_from_db()

    # آموزش مدل LightGBM برای دریافت احتمال نکول
    predicted_probabilities_test = train_lightgbm_model(x_train_resampled, y_train_resampled, x_test)

    # محاسبه ضررهای مالی از اطلاعات جریان نقدی (استفاده از ستون‌های محافظت‌شده)
    cash_flow_data = x_test[protected_columns]
    false_positive_loss_test, false_negative_loss_test = compute_financial_losses(cash_flow_data)

    # بهینه‌سازی پارامترهای آستانه
    optimized_upper_threshold_scale, optimized_lower_threshold_scale = optimize_threshold_scales(
        predicted_probabilities_test, false_positive_loss_test, false_negative_loss_test, population_size=20, num_generations=10
    )

    print("بهترین مقدار برای مقیاس آستانه بالا:", Decimal(optimized_upper_threshold_scale))
    print("بهترین مقدار برای مقیاس آستانه پایین:", Decimal(optimized_lower_threshold_scale))

    # اعمال تصمیم سه‌طرفه
    three_way_decision_labels, uncertain_boundary_sample_indices = apply_three_way_decision(
        predicted_probabilities_test, false_positive_loss_test, false_negative_loss_test, optimized_upper_threshold_scale, optimized_lower_threshold_scale
    )

    # بهبود تصمیم‌های نمونه‌های مرزی با استفاده از مدل ترکیبی
    ensemble_bagging_classifier = BaggingClassifier(
        estimator=ExtraTreesClassifier(n_estimators=100, random_state=42),
        n_estimators=10,  # تعداد مدل‌های پایه در Bagging
        random_state=42
    )
    ensemble_bagging_classifier.fit(x_train_resampled, y_train_resampled)

    # پیش‌بینی برای نمونه‌های نامطمئن
    x_test_boundary_samples = x_test.iloc[uncertain_boundary_sample_indices]
    predicted_labels_for_boundary_samples = ensemble_bagging_classifier.predict(x_test_boundary_samples)
    three_way_decision_labels[uncertain_boundary_sample_indices] = predicted_labels_for_boundary_samples

    # ارزیابی عملکرد
    evaluate_model_performance(np.array(y_test), np.array(three_way_decision_labels), false_positive_loss_test, false_negative_loss_test)
