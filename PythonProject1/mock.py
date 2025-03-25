from math import sqrt

from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# تابع محاسبه F‑Measure؛ اگر b=1 معادل F1-Score خواهد بود.
def calc_fm(precision, recall, b=1):
    if (precision + recall) == 0:
        return 0.0
    return (1 + b**2) * (precision * recall) / (b**2 * precision + recall)

# تابع محاسبه G‑Mean؛ میانگین هندسی بین حساسیت (Sensitivity) و ویژگی (Specificity)
def calc_gm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # ماتریس سردرگمی به صورت [[TN, FP], [FN, TP]]
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    if (TP + FN) == 0 or (TN + FP) == 0:
        return 0.0
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return sqrt(sensitivity * specificity)

# تابع جامع ارزیابی مدل
def evaluate_model(y_true, y_pred, y_prob=None, b=1, cost_fp=1, cost_fn=1):
    b_acc = balanced_accuracy_score(y_true, y_pred)
    auc = None
    if y_prob is not None:
        try:
            if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                y_score = y_prob[:, 1]
            else:
                y_score = y_prob
            auc = roc_auc_score(y_true, y_score)
        except Exception as e:
            auc = None
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    fm = calc_fm(prec, rec, b)
    gm = calc_gm(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    cost = FP * cost_fp + FN * cost_fn
    metrics = {
        "Balanced Accuracy": b_acc,
        "AUC": auc,
        "F-Measure": fm,
        "G-Mean": gm,
        "Cost": cost,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN
    }
    return metrics

# تابع کلی آموزش و ارزیابی مدل
def train_and_evaluate(model, x_train, y_train, x_test, y_test, b=1, cost_fp=1, cost_fn=1):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    try:
        y_prob = model.predict_proba(x_test)
    except Exception:
        y_prob = None
    return evaluate_model(y_test, y_pred, y_prob, b, cost_fp, cost_fn)

if __name__ == "__main__":
    # تولید داده‌های مصنوعی برای طبقه‌بندی دودویی
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=2, n_redundant=10,
                               n_classes=2, random_state=42)
    # تقسیم داده‌ها به آموزش (70٪) و تست (30٪)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # تعریف مدل‌های مختلف با تنظیماتی برای رفع هشدارها
    models = {
        "Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "LR": LogisticRegression(max_iter=1000),  # افزایش max_iter برای همگام‌سازی
        "NN": MLPClassifier(max_iter=300),
        "SVM": SVC(probability=True),
        "AdaBoost": AdaBoostClassifier(algorithm="SAMME"),  # استفاده از الگوریتم SAMME
        "ERT": ExtraTreesClassifier(),
        "GBDT": GradientBoostingClassifier(),
        "LGBM": LGBMClassifier(verbose=-1),  # کاهش پیام‌های خروجی
        "RF": RandomForestClassifier(),
        "XGB": XGBClassifier(eval_metric='logloss', verbosity=0),  # حذف پارامتر use_label_encoder و کاهش verbosity
        "Stacking": StackingClassifier(estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('knn', KNeighborsClassifier())
        ], final_estimator=RandomForestClassifier())
    }

    results = {}
    # آموزش و ارزیابی هر مدل
    for name, model in models.items():
        print(f"در حال آموزش و ارزیابی مدل: {name}")
        metrics = train_and_evaluate(model, x_train, y_train, x_test, y_test, b=1, cost_fp=1, cost_fn=1)
        results[name] = metrics
        print(f"نتایج مدل {name}: {metrics}\n")

    print("نتایج کلی:")
    for name, metric in results.items():
        print(f"{name}: {metric}")
