import os

import numpy as np
# کتابخانه‌های مدل‌سازی و ارزیابی
from lightgbm import LGBMClassifier
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, classification_report, confusion_matrix, \
    precision_score, recall_score, f1_score

# کتابخانه‌های مربوط به دریافت و پیش‌پردازش داده (کلاس‌های موجود در پروژه)
from dataHandler.LoanDataHandler import LoanDataHandler
from processor.LoanPreprocessor import LoanPreprocessor, protected_columns
from repository.LoanRepository import LoanRepository


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
    p_pred = lgbm.predict_proba(x_test)[:, 1]
    return p_pred

# ------------------------ محاسبه ضرر ------------------------
def compute_losses(cash_flow_info):
    principal = cash_flow_info['approval_amount'].values
    interest = cash_flow_info['interest_amount'].values
    lpn_arr = interest
    lnp_arr = principal + interest
    return lpn_arr, lnp_arr


# ------------------------ تعریف تابع هدف NSGA-II (وکتورایز شده) ------------------------
class ObjectiveProblem(Problem):
    def __init__(self, p_pred, lpn_arr, lnp_arr):
        self.p_pred = p_pred
        self.lpn_arr = lpn_arr
        self.lnp_arr = lnp_arr
        super().__init__(n_var=2, n_obj=2, n_constr=1,
                         xl=np.array([0.0, 0.0]),
                         xu=np.array([1.0, 1.0]))

    def _evaluate(self, x, out, *args, **kwargs):
        n_ind = x.shape[0]
        f1 = np.zeros(n_ind)
        f2 = np.zeros(n_ind)
        for i in range(n_ind):
            u, v = x[i]
            bp = u * self.lnp_arr
            bn = v * self.lpn_arr
            numerator_alpha = self.lpn_arr - bn
            denom_alpha = numerator_alpha + bp
            alpha = np.where(denom_alpha == 0, 1.0, numerator_alpha / denom_alpha)
            numerator_beta = bn
            denom_beta = bn + (self.lnp_arr - bp)
            beta = np.where(denom_beta == 0, 0.0, numerator_beta / denom_beta)
            cost_local = np.where(self.p_pred >= alpha,
                                  self.lpn_arr * (1 - self.p_pred),
                                  np.where(self.p_pred <= beta,
                                           self.lnp_arr * self.p_pred,
                                           bp * self.p_pred + bn * (1 - self.p_pred)))
            f1[i] = np.sum(cost_local)
            f2[i] = np.sum(alpha - beta)
        g = x[:, 0] + x[:, 1] - 1.0
        out["F"] = np.column_stack([f1, f2])
        out["G"] = g.reshape(-1, 1)



# ------------------------ بهینه‌سازی u و v با NSGA-II ------------------------
def nsga2_find_uv(p_pred, lpn_arr, lnp_arr, pop_size=20, generations=10):
    problem = ObjectiveProblem(p_pred, lpn_arr, lnp_arr)
    algorithm = NSGA2(pop_size=pop_size)
    res = minimize(problem, algorithm, ('n_gen', generations), seed=1, verbose=False)
    f = res.F
    idx = np.lexsort((f[:, 1], f[:, 0]))[0]
    return res.X[idx, 0], res.X[idx, 1]


# ------------------------ اعمال تصمیم سه‌طرفه ------------------------
def apply_three_way_decision(p_pred, loss_pn_arr, loss_np_arr, u, v):
    bp = u * loss_np_arr
    bn = v * loss_pn_arr
    numerator_alpha = loss_pn_arr - bn
    denom_alpha = numerator_alpha + bp
    alpha = np.where(denom_alpha == 0, 1.0, numerator_alpha / denom_alpha)
    numerator_beta = bn
    denom_beta = bn + (loss_np_arr - bp)
    beta = np.where(denom_beta == 0, 0.0, numerator_beta / denom_beta)
    labels = np.where(p_pred >= alpha, 1, np.where(p_pred <= beta, 0, -1))
    boundary_indices = np.where(labels == -1)[0]
    return labels, boundary_indices

# ------------------------ ارزیابی عملکرد ------------------------
def evaluate_performance(y_true, y_pred, lpn_arr, lnp_arr):
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
    cost_decision = np.sum(lnp_arr[(y_true == 1) & (y_pred == 0)]) + np.sum(lpn_arr[(y_true == 0) & (y_pred == 1)])
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
    os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # مقدار را بر اساس تعداد هسته‌های CPU خودت تنظیم کن

    # ۱. دریافت و پیش‌پردازش داده
    x_train_res, y_train_res, x_test, y_test = pre_process_data_from_db()

    # ۲. احتمال این که یک نم.نه نکول شه
    p_pred_test = train_lgbm_model(x_train_res, y_train_res, x_test)

    # ۳. محاسبه ضررهای مالی از اطلاعات جریان نقدی
    cash_flow = x_test[protected_columns]
    # در این کد، متغیر pn به معنای ضرر PN است که به "سود از دست رفته" اشاره دارد. یعنی زمانی که یک نمونه غیر نکول (non-default) به اشتباه به عنوان نکول (default) تشخیص داده شود، تنها سود (interest_amount) از دست می‌رود
    # مقدار ضرر این  که سود از دست برده و مقدار ضرر این ک اینکه سود و اصل پول از دست بره
    loss_pn_arr_test, loss_np_arr_test = compute_losses(cash_flow)

    # ۴. بهینه‌سازی u و v با NSGA-II
    best_u, best_v = nsga2_find_uv(p_pred_test, loss_pn_arr_test, loss_np_arr_test, pop_size=20, generations=10)
    print("بهترین u:", best_u)
    print("بهترین v:", best_v)

    # ۵. اعمال تصمیم سه‌طرفه
    # u =  وزن هزینه از دست دادن اصل و سود وام

    twd_labels, boundary_indices = apply_three_way_decision(p_pred_test, loss_pn_arr_test, loss_np_arr_test, best_u, best_v)


    # استفاده از BaggingClassifier با مدل پایه LGBMClassifier
    bagging_model = BaggingClassifier(
        estimator=ExtraTreesClassifier(n_estimators=100, random_state=42),
        n_estimators=10,  # تعداد نسخه‌های مدل پایه که می‌خواهیم آموزش ببینند
        random_state=42
    )
    bagging_model.fit(x_train_res, y_train_res)

    # پیش‌بینی روی نمونه‌های مرزی
    x_test_boundary = x_test.iloc[boundary_indices]
    y_pred_boundary = bagging_model.predict(x_test_boundary)
    twd_labels[boundary_indices] = y_pred_boundary

    # ۸. ارزیابی عملکرد کلی مدل
    evaluate_performance(np.array(y_test), np.array(twd_labels), loss_pn_arr_test, loss_np_arr_test)

