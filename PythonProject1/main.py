import numpy as np
# کتابخانه‌های مدل‌سازی و ارزیابی
from lightgbm import LGBMClassifier
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier, VotingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, classification_report, confusion_matrix, \
    precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# کتابخانه‌های مربوط به دریافت و پیش‌پردازش داده (کلاس‌های موجود در پروژه)
from dataHandler.LoanDataHandler import LoanDataHandler
from processor.LoanPreprocessor import LoanPreprocessor
from repository.LoanRepository import LoanRepository


# ------------------------ دریافت و پیش‌پردازش داده ------------------------
def preProcessDataFromDB():
    repository = LoanRepository()
    preprocessor = LoanPreprocessor(imputation_strategy="median")
    data_handler = LoanDataHandler(repository, preprocessor)
    return data_handler.load_and_process_data(limit_records=50000)


# ------------------------ آموزش مدل LightGBM ------------------------
def trainLGBMModel(X_train, y_train, X_test):
    lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    print("شروع آموزش مدل LightGBM...")
    lgbm.fit(X_train, y_train)
    print("آموزش مدل به پایان رسید.")
    p_pred = lgbm.predict_proba(X_test)[:, 1]
    return p_pred, lgbm


# ------------------------ محاسبه ضرر ------------------------
def computeLosses(cash_flow_info):
    # استفاده از عملگرهای وکتورایز برای محاسبه ضرر
    principal = cash_flow_info['approval_amount'].values
    interest = cash_flow_info['interest_amount'].values
    lPN_arr = interest           # هزینه PN به صورت مستقیم
    lNP_arr = principal + interest  # هزینه NP به صورت وکتور
    return lPN_arr, lNP_arr


# ------------------------ تعریف تابع هدف NSGA-II (وکتورایز شده) ------------------------
class ObjectiveProblem(Problem):
    def __init__(self, p_pred, lPN_arr, lNP_arr):
        self.p_pred = p_pred
        self.lPN_arr = lPN_arr
        self.lNP_arr = lNP_arr
        super().__init__(n_var=2, n_obj=2, n_constr=1,
                         xl=np.array([0.0, 0.0]),
                         xu=np.array([1.0, 1.0]))

    def _evaluate(self, X, out, *args, **kwargs):
        n_ind = X.shape[0]
        f1 = np.zeros(n_ind)
        f2 = np.zeros(n_ind)
        # محاسبات هر فرد (u, v) به صورت وکتورایز درون داده‌های نمونه
        for i in range(n_ind):
            u, v = X[i]
            bp = u * self.lNP_arr
            bn = v * self.lPN_arr
            numerator_alpha = self.lPN_arr - bn
            denom_alpha = numerator_alpha + bp
            alpha = np.where(denom_alpha == 0, 1.0, numerator_alpha / denom_alpha)
            numerator_beta = bn
            denom_beta = bn + (self.lNP_arr - bp)
            beta = np.where(denom_beta == 0, 0.0, numerator_beta / denom_beta)
            cost_local = np.where(self.p_pred >= alpha,
                                  self.lPN_arr * (1 - self.p_pred),
                                  np.where(self.p_pred <= beta,
                                           self.lNP_arr * self.p_pred,
                                           bp * self.p_pred + bn * (1 - self.p_pred)))
            f1[i] = np.sum(cost_local)
            f2[i] = np.sum(alpha - beta)
        # قید: u + v <= 1
        g = X[:, 0] + X[:, 1] - 1.0
        out["F"] = np.column_stack([f1, f2])
        out["G"] = g.reshape(-1, 1)


# ------------------------ بهینه‌سازی u و v با NSGA-II ------------------------
def nsga2_find_uv(p_pred, lPN_arr, lNP_arr, pop_size=20, generations=10):
    problem = ObjectiveProblem(p_pred, lPN_arr, lNP_arr)
    algorithm = NSGA2(pop_size=pop_size)
    res = minimize(problem, algorithm, ('n_gen', generations), seed=1, verbose=False)
    F = res.F
    # انتخاب بهترین جواب به صورت لکسیکوگرافی (ابتدا f1 سپس f2)
    idx = np.lexsort((F[:, 1], F[:, 0]))[0]
    return res.X[idx, 0], res.X[idx, 1]


# ------------------------ اعمال تصمیم سه‌طرفه ------------------------
def applyThreeWayDecision(p_pred, lPN_arr, lNP_arr, u, v):
    bp = u * lNP_arr
    bn = v * lPN_arr
    numerator_alpha = lPN_arr - bn
    denom_alpha = numerator_alpha + bp
    alpha = np.where(denom_alpha == 0, 1.0, numerator_alpha / denom_alpha)
    numerator_beta = bn
    denom_beta = bn + (lNP_arr - bp)
    beta = np.where(denom_beta == 0, 0.0, numerator_beta / denom_beta)
    labels = np.where(p_pred >= alpha, 1, np.where(p_pred <= beta, 0, -1))
    boundary_indices = np.where(labels == -1)[0]
    return labels, boundary_indices


# ------------------------ ارزیابی عملکرد ------------------------
def evaluate_performance(y_true, y_pred, lPN_arr, lNP_arr):
    ba = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cost_decision = np.sum(lNP_arr[(y_true == 1) & (y_pred == 0)]) + np.sum(lPN_arr[(y_true == 0) & (y_pred == 1)])
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
    # ۱. دریافت و پیش‌پردازش داده
    X_train_res, y_train_res, X_test, y_test = preProcessDataFromDB()

    # ۲. آموزش مدل LightGBM و دریافت احتمال نکول
    p_pred_test, _ = trainLGBMModel(X_train_res, y_train_res, X_test)

    # ۳. محاسبه ضررهای مالی از اطلاعات جریان نقدی
    cash_flow = X_test[['approval_amount', 'interest_amount']]
    lPN_arr_test, lNP_arr_test = computeLosses(cash_flow)

    # ۴. بهینه‌سازی u و v با NSGA-II
    best_u, best_v = nsga2_find_uv(p_pred_test, lPN_arr_test, lNP_arr_test, pop_size=20, generations=10)
    print("بهترین u:", best_u, "بهترین v:", best_v)

    # ۵. اعمال تصمیم سه‌طرفه
    twd_labels, boundary_indices = applyThreeWayDecision(p_pred_test, lPN_arr_test, lNP_arr_test, best_u, best_v)

    # ۶. استفاده از VotingClassifier برای مدل بگینگ با چند مدل پایه
    # تعریف مدل‌های پایه
    estimators = [
        ('lgb', LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=0)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=0)),
        ('xgb', XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=0)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=0)),
        ('et', ExtraTreesClassifier(n_estimators=100, random_state=0)),
        ('ada', AdaBoostClassifier(n_estimators=100, random_state=0))
    ]
    # استفاده از VotingClassifier به صورت soft voting برای ترکیب مدل‌ها
    voting_model = VotingClassifier(estimators=estimators, voting='soft')
    voting_model.fit(X_train_res, y_train_res)

    # ۷. پیش‌بینی نمونه‌های مرزی با مدل VotingClassifier
    X_test_boundary = X_test.iloc[boundary_indices]
    y_pred_boundary = voting_model.predict(X_test_boundary)
    twd_labels[boundary_indices] = y_pred_boundary

    # ۸. ارزیابی عملکرد کلی مدل
    evaluate_performance(np.array(y_test), np.array(twd_labels), lPN_arr_test, lNP_arr_test)
