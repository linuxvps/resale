import logging
import os
from datetime import datetime
from decimal import Decimal
from math import sqrt

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier, BaggingClassifier)
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, f1_score, confusion_matrix,
                             balanced_accuracy_score, roc_auc_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine, Column, Integer, BigInteger, String, Date, DateTime, Numeric, Float, Text, \
    SmallInteger
from sqlalchemy.orm import declarative_base, sessionmaker
from xgboost import XGBClassifier

# تنظیم logging به جای استفاده از print
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

Base = declarative_base()
# لیست ستون‌های محافظت‌شده
protected_columns = ['approval_amount', 'interest_amount']
results = {}

# تابع کمکی برای تبدیل نتایج SQLAlchemy به DataFrame به صورت استاندارد
def sqlalchemy_results_to_df(loans, model):
    if not loans:
        logging.warning("هیچ داده‌ای از پایگاه داده دریافت نشد.")
        return pd.DataFrame()
    columns = [col.name for col in model.__table__.columns]
    data = {col: [getattr(loan, col) for loan in loans] for col in columns}
    df = pd.DataFrame(data)
    logging.info(f"✅ {len(df)} رکورد از دیتابیس دریافت شد.")
    return df

# مدل ParsianLoan مطابق مقاله
class ParsianLoan(Base):
    __tablename__ = "parsian_loan"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    insert_sysdate = Column(DateTime, nullable=False, default=datetime.utcnow)
    branch_code = Column(Integer, nullable=False)
    branchname = Column(String(100), nullable=True)
    client_id = Column(Integer, nullable=True)
    loan_file_numberr = Column(BigInteger, nullable=True)
    sit_flag = Column(String(1), nullable=True)
    interest_rate = Column(Numeric(19, 2), nullable=True)
    total_repayment_up_to_now = Column(Numeric(28, 8), nullable=True)
    commission_amount_remain = Column(Numeric(28, 8), nullable=True)
    charge = Column(Numeric(28, 8), nullable=True)
    discount = Column(Numeric(28, 8), nullable=True)
    advance_pay_to_total_cash = Column(Numeric(28, 8), nullable=True)
    advance_pay_to_remain_non_cash = Column(Numeric(28, 8), nullable=True)
    is_installment = Column(String(1), nullable=True)
    interest_sum = Column(Numeric(28, 8), nullable=True)
    installment_number_remain = Column(Integer, nullable=True)
    receivable_installment_number = Column(Integer, nullable=True)
    first_passed = Column(Date, nullable=True)
    total_payment_up_to_now = Column(Numeric(28, 8), nullable=True)
    finalized_loan_amount = Column(Numeric(28, 8), nullable=True)
    penalty = Column(Numeric(28, 8), nullable=True)
    first_payment_date_in_du = Column(Date, nullable=True)
    principal_sum = Column(Numeric(28, 8), nullable=True)
    advance_pay = Column(Numeric(28, 8), nullable=True)
    sit_duration = Column(Integer, nullable=True)
    sit_duration_day = Column(Integer, nullable=True)
    sit_distribute_phases = Column(Integer, nullable=True)
    sit_fast_receive_percent = Column(Float, nullable=True)
    frequency = Column(Integer, nullable=True)
    customer_obligation_amount = Column(Numeric(28, 8), nullable=True)
    customer_share_cash_amount = Column(Numeric(28, 8), nullable=True)
    customer_share_non_cash_amount = Column(Numeric(28, 8), nullable=True)
    bank_share_cash_amount = Column(Numeric(28, 8), nullable=True)
    bank_share_non_cash_amount = Column(Numeric(28, 8), nullable=True)
    first_over_due = Column(Date, nullable=True)
    loan_duration_day = Column(Integer, nullable=True)
    loan_file_number = Column(BigInteger, nullable=True)
    create_date = Column(Date, nullable=True)
    long_title = Column(String(255), nullable=True)
    status = Column(String(255), nullable=True)
    contract = Column(String(255), nullable=True)
    approval_amount = Column(Numeric(28, 8), nullable=True)
    title = Column(String(255), nullable=True)
    inc_commission_amount = Column(Numeric(28, 8), nullable=True)
    interest_amount = Column(Numeric(28, 8), nullable=True)
    obligation_penalty = Column(Numeric(28, 8), nullable=True)
    passed_date = Column(Date, nullable=True)
    penalty_interest = Column(Numeric(28, 8), nullable=True)
    to_due_date = Column(Numeric(28, 8), nullable=True)
    to_end_of_month = Column(Numeric(28, 8), nullable=True)
    due_date = Column(Date, nullable=True)

    def __repr__(self):
        return f"<ParsianLoan(id={self.id}, branch_code={self.branch_code}, client_id={self.client_id})>"

# مدل LoanFeatures طبق مقاله
class LoanFeatures(Base):
    __tablename__ = "loan_features"

    feature_id = Column(Integer, primary_key=True, autoincrement=True)
    column_name = Column(String(255), nullable=False)
    table_name = Column(String(255), nullable=False)
    importance_level = Column(SmallInteger, nullable=False)
    importance_description = Column(Text, nullable=False)

    def __repr__(self):
        return f"<LoanFeatures(feature_id={self.feature_id}, column_name='{self.column_name}', table_name='{self.table_name}', importance_level={self.importance_level})>"

# ==================== تعریف کلاس LoanRepository ====================
class LoanRepository:
    def __init__(self):
        self.session = self.create_database_session()

    @staticmethod
    def create_database_session():
        # استفاده از متغیر محیطی برای رشته اتصال؛ در صورت عدم وجود، مقدار پیش‌فرض استفاده می‌شود
        db_conn = os.getenv("DB_CONNECTION_STRING", "mysql+pymysql://root:pass@localhost:3306/ln")
        engine = create_engine(db_conn)
        SessionLocal = sessionmaker(bind=engine)
        return SessionLocal()

    def fetch_loans(self, limit: int = 10000) -> pd.DataFrame:
        loans = self.session.query(ParsianLoan).limit(limit).all()
        df = sqlalchemy_results_to_df(loans, ParsianLoan)
        return df

# ==================== تعریف کلاس LoanPreprocessor ====================
class LoanPreprocessor:
    def __init__(self, imputation_strategy: str = "mean"):
        self.imputer = SimpleImputer(strategy=imputation_strategy)

    def convert_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.datetime64):
                # تبدیل تاریخ به timestamp به عنوان عدد شناور (ثانیه از epoch)
                df[col] = pd.to_datetime(df[col]).apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
            elif df[col].dtype == 'object':
                # تلاش برای تبدیل به عدد، در صورت عدم موفقیت از LabelEncoder استفاده می‌شود
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
        return df

    def encode_labels(self, column: pd.Series) -> pd.Series:
        le = LabelEncoder()
        return le.fit_transform(column.astype(str))

    def convert_labels(self, df: pd.DataFrame, label_column: str = "status") -> pd.DataFrame:
        if label_column not in df.columns:
            raise ValueError(f"⚠️ ستون '{label_column}' در داده وجود ندارد. لطفاً نام صحیح ستون برچسب را مشخص کنید.")
        logging.info(f"🟢 ستون برچسب انتخاب شده: {label_column}")
        logging.info("🔹 مقدارهای `status` قبل از تبدیل:")
        logging.info(df[label_column].value_counts().to_string())
        default_statuses = {'مشكوك الوصول', 'معوق', 'سررسيد گذشته'}
        df[label_column] = df[label_column].apply(lambda x: 1 if x in default_statuses else 0)
        logging.info("🔹 تعداد برچسب‌های نکول و غیرنکول پس از تبدیل:")
        logging.info(df[label_column].value_counts().to_string())
        return df

    def remove_highly_correlated_features(self, data, threshold, class_column=None):
        new_data = data.copy()
        numeric_cols = new_data.select_dtypes(include=[np.number]).columns.tolist()
        if class_column and class_column in numeric_cols:
            numeric_cols.remove(class_column)
        numeric_cols = [col for col in numeric_cols if col not in protected_columns]
        corr_matrix = new_data[numeric_cols].corr()
        attributes_to_remove = set()
        for i in range(len(numeric_cols) - 1):
            col_i = numeric_cols[i]
            for j in range(i + 1, len(numeric_cols)):
                col_j = numeric_cols[j]
                corr_value = corr_matrix.loc[col_i, col_j]
                if abs(corr_value) > threshold:
                    logging.info(f"🔴 همبستگی بالا بین: {col_i} و {col_j} | مقدار: {corr_value:.4f} | حذف: {col_j}")
                    attributes_to_remove.add(col_j)
        for col in attributes_to_remove:
            if class_column and col == class_column:
                continue
            if protected_columns and col in protected_columns:
                continue
            logging.info(f"✅ ویژگی حذف شد: {col}")
            new_data.drop(columns=[col], inplace=True)
        logging.info("ماتریس همبستگی:")
        logging.info(corr_matrix.to_string())
        return new_data

    def select_features(self, X, y):
        lgbm_estimator = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        rfecv = RFECV(estimator=lgbm_estimator, step=1, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
        rfecv.fit(X, y)
        selected_features = list(X.columns[rfecv.support_])
        # افزودن ویژگی‌های محافظت‌شده در صورت عدم وجود در لیست انتخاب‌شده
        for col in protected_columns:
            if col in X.columns and col not in selected_features:
                selected_features.append(col)
        not_selected_features = [col for col in X.columns if col not in selected_features]
        logging.info("ویژگی‌های انتخاب شده: " + ", ".join(selected_features))
        logging.info("ویژگی‌های انتخاب نشده: " + ", ".join(not_selected_features))
        return X.loc[:, selected_features]


    def preprocess(self, df: pd.DataFrame, label_column: str = "status") -> (pd.DataFrame, pd.Series):
        df = self.convert_labels(df, label_column)
        df = self.convert_dataframe_columns(df)
        # حذف ستون‌هایی که ممکن است اطلاعات اضافی داشته باشند
        df.drop(columns=["create_date"], errors="ignore", inplace=True)
        df = self.remove_highly_correlated_features(df, threshold=0.9, class_column=label_column)
        df_imputed = pd.DataFrame(self.imputer.fit_transform(df), columns=df.columns)
        X = df_imputed.drop(columns=[label_column])
        y = df_imputed[label_column]
        return X, y

# کلاس مدیریت داده‌ها
class LoanDataHandler:
    def __init__(self, repository: LoanRepository, preprocessor: LoanPreprocessor):
        self.repository = repository
        self.preprocessor = preprocessor

    def load_and_process_data(self, limit_records: int = 10000) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
        df = self.repository.fetch_loans(limit_records)
        X, y = self.preprocessor.preprocess(df)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        x_train_selected = self.preprocessor.select_features(x_train, y_train)
        x_test_selected = x_test[x_train_selected.columns]
        return x_train_selected, y_train, x_test_selected, y_test

# تعریف مسئله بهینه‌سازی آستانه‌ها
class ThresholdOptimizationProblem(Problem):
    def __init__(self, predicted_probs, false_pos_cost, false_neg_cost):
        self.predicted_probs = predicted_probs
        self.false_pos_cost = false_pos_cost
        self.false_neg_cost = false_neg_cost
        super().__init__(
            n_var=2,
            n_obj=2,
            n_constr=1,
            xl=np.array([0.0, 0.0]),
            xu=np.array([1.0, 1.0])
        )

    def calculate_adjusted_costs(self, scale_fn, scale_fp):
        adjusted_fn_cost = scale_fn * self.false_neg_cost
        adjusted_fp_cost = scale_fp * self.false_pos_cost
        return adjusted_fn_cost, adjusted_fp_cost

    def calculate_thresholds(self, adjusted_fn_cost, adjusted_fp_cost):
        numerator_upper = self.false_pos_cost - adjusted_fp_cost
        denominator_upper = numerator_upper + adjusted_fn_cost
        upper_threshold = np.where(denominator_upper == 0, 1.0, numerator_upper / denominator_upper)
        numerator_lower = adjusted_fp_cost
        denominator_lower = adjusted_fp_cost + (self.false_neg_cost - adjusted_fn_cost)
        lower_threshold = np.where(denominator_lower == 0, 0.0, numerator_lower / denominator_lower)
        return upper_threshold, lower_threshold

    def compute_sample_costs(self, upper_threshold, lower_threshold, adjusted_fn_cost, adjusted_fp_cost):
        sample_costs = np.where(self.predicted_probs >= upper_threshold,
                                self.false_pos_cost * (1 - self.predicted_probs),
                                np.where(self.predicted_probs <= lower_threshold,
                                         self.false_neg_cost * self.predicted_probs,
                                         adjusted_fn_cost * self.predicted_probs + adjusted_fp_cost * (1 - self.predicted_probs)))
        return sample_costs

    def _evaluate(self, solution, out, *args, **kwargs):
        num_solutions = solution.shape[0]
        total_costs = np.zeros(num_solutions)
        total_boundary_width = np.zeros(num_solutions)
        for i in range(num_solutions):
            scale_fn, scale_fp = solution[i]
            adj_fn_cost, adj_fp_cost = self.calculate_adjusted_costs(scale_fn, scale_fp)
            upper_threshold, lower_threshold = self.calculate_thresholds(adj_fn_cost, adj_fp_cost)
            sample_costs = self.compute_sample_costs(upper_threshold, lower_threshold, adj_fn_cost, adj_fp_cost)
            total_costs[i] = np.sum(sample_costs)
            total_boundary_width[i] = np.sum(upper_threshold - lower_threshold)
        constraint = solution[:, 0] + solution[:, 1] - 1.0
        out["F"] = np.column_stack([total_costs, total_boundary_width])
        out["G"] = constraint.reshape(-1, 1)

def optimize_threshold_scales(predicted_probs, false_pos_cost, false_neg_cost, population_size=20, num_generations=10):
    problem_instance = ThresholdOptimizationProblem(predicted_probs, false_pos_cost, false_neg_cost)
    nsga2_algo = NSGA2(pop_size=population_size)
    optimization_result = minimize(problem_instance, nsga2_algo, ('n_gen', num_generations), seed=1, verbose=False)
    objectives = optimization_result.F
    best_index = np.lexsort((objectives[:, 1], objectives[:, 0]))[0]
    best_scale_fn, best_scale_fp = optimization_result.X[best_index]
    return best_scale_fn, best_scale_fp

# تابع برای آموزش مدل LightGBM
def train_lightgbm_model(x_train, y_train, x_test):
    lightgbm_classifier = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    logging.info("شروع آموزش مدل LightGBM...")
    lightgbm_classifier.fit(x_train, y_train)
    logging.info("آموزش مدل به پایان رسید.")
    predicted_probabilities = lightgbm_classifier.predict_proba(x_test)[:, 1]
    return predicted_probabilities

# تابع محاسبه هزینه‌های مالی بر مبنای جریان نقدی
def compute_financial_losses(cash_flow_info):
    principal_amount = cash_flow_info['approval_amount'].values
    interest_amount = cash_flow_info['interest_amount'].values
    false_positive_loss = interest_amount
    false_negative_loss = principal_amount + interest_amount
    return false_positive_loss, false_negative_loss

def get_classifier(classifier_type='bagging'):
    if classifier_type.lower() == 'stacking':
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42)),
            ('gbdt', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('ert', ExtraTreesClassifier(n_estimators=100, random_state=42)),
            ('ada', AdaBoostClassifier(algorithm="SAMME", n_estimators=100, random_state=42))
        ]
        meta_estimator = LogisticRegression(max_iter=1000, random_state=42)
        classifier = StackingClassifier(estimators=base_estimators, final_estimator=meta_estimator, cv=5, n_jobs=-1)
        return classifier
    elif classifier_type.lower() == 'bagging':
        classifier = BaggingClassifier(estimator=ExtraTreesClassifier(n_estimators=100, random_state=42),
                                       n_estimators=10, random_state=42)
        return classifier
    else:
        raise ValueError("نوع طبقه‌بندی‌کننده باید 'bagging' یا 'stacking' باشد.")


# تابع تصمیم‌گیری سه‌راهه با استفاده از آستانه‌های بهینه و مدل استکینگ برای نمونه‌های حوزه تأخیر
def apply_three_way_decision(predicted_probabilities, false_positive_loss, false_negative_loss, upper_threshold_scale, lower_threshold_scale):
    boundary_penalty_positive = upper_threshold_scale * false_negative_loss
    boundary_penalty_negative = lower_threshold_scale * false_positive_loss

    numerator_alpha = false_positive_loss - boundary_penalty_negative
    denominator_alpha = numerator_alpha + boundary_penalty_positive
    alpha_threshold = np.where(denominator_alpha == 0, 1.0, numerator_alpha / denominator_alpha)

    numerator_beta = boundary_penalty_negative
    denominator_beta = boundary_penalty_negative + (false_negative_loss - boundary_penalty_positive)
    beta_threshold = np.where(denominator_beta == 0, 0.0, numerator_beta / denominator_beta)

    alpha_threshold = np.maximum(alpha_threshold, beta_threshold)

    three_way_decision_labels = np.where(predicted_probabilities >= alpha_threshold, 1,
                                         np.where(predicted_probabilities <= beta_threshold, 0, -1))
    uncertain_boundary_sample_indices = np.where(three_way_decision_labels == -1)[0]
    return three_way_decision_labels, uncertain_boundary_sample_indices

# توابع محاسبه معیارهای ارزیابی مدل
def calc_fm(precision, recall, b=1):
    if (precision + recall) == 0:
        return 0.0
    return (1 + b**2) * (precision * recall) / (b**2 * precision + recall)

def calc_gm(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    if (TP + FN) == 0 or (TN + FP) == 0:
        return 0.0
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return sqrt(sensitivity * specificity)

def evaluate_model_performance(true_labels, predicted_labels, false_positive_loss, false_negative_loss):
    logging.info("\n\nارزیابی عملکرد کلی مدل\n\n")
    balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)
    auc = roc_auc_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    classification_rep = classification_report(true_labels, predicted_labels)
    decision_cost = np.sum(false_negative_loss[(true_labels == 1) & (predicted_labels == 0)]) + \
                    np.sum(false_positive_loss[(true_labels == 0) & (predicted_labels == 1)])
    logging.info(f"Balanced Accuracy: {balanced_accuracy}")
    logging.info(f"AUC: {auc}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"F1 Score: {f1}")
    logging.info("Confusion Matrix:")
    logging.info(f"[[TN: {cm[0, 0]}, FP: {cm[0, 1]}], [FN: {cm[1, 0]}, TP: {cm[1, 1]}]]")
    logging.info("Classification Report:\n" + classification_rep)
    logging.info(f"Decision Cost: {decision_cost}")
    fm = calc_fm(precision, recall, b=1)
    gm = calc_gm(true_labels, predicted_labels)
    logging.info(f"fm: {fm}")
    logging.info(f"gm: {gm}")
    return {
        "Balanced Accuracy": balanced_accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "FM": fm,
        "GM": gm,
        "Decision Cost": decision_cost,
        "TP": cm[1, 1],
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0]
    }

def apply_smote(X, y, random_state=42):
    logging.info("تعداد نمونه‌های آموزشی قبل از SMOTE:")
    logging.info(pd.Series(y).value_counts().to_string())
    sm = SMOTE(random_state=random_state)
    X_resampled, y_resampled = sm.fit_resample(X, y)
    logging.info("تعداد نمونه‌های آموزشی بعد از SMOTE:")
    logging.info(pd.Series(y_resampled).value_counts().to_string())
    return X_resampled, y_resampled



def train_and_evaluate(model, x_train, y_train, x_test, y_test, b=1, cost_fp=1, cost_fn=1):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    try:
        y_prob = model.predict_proba(x_test)
    except Exception:
        y_prob = None
    return evaluate_model(y_test, y_pred, y_prob, b, cost_fp, cost_fn)


# تابع جامع ارزیابی مدل که معیارهای مختلفی مانند Balanced Accuracy، AUC، F‑Measure، G‑Mean، هزینه تصمیم‌گیری و تعداد TP، TN، FP، FN را محاسبه می‌کند.
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
        except Exception:
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

if __name__ == "__main__":
    os.environ["LOKY_MAX_CPU_COUNT"] = "8"
    # بارگذاری و پیش‌پردازش داده‌ها
    loan_repository = LoanRepository()
    loan_preprocessor = LoanPreprocessor(imputation_strategy="median")
    loan_data_handler = LoanDataHandler(loan_repository, loan_preprocessor)
    x_train, y_train, x_test, y_test = loan_data_handler.load_and_process_data(limit_records=100_000)
    original_data = {"x_train": x_train.copy(), "y_train": y_train.copy(), "x_test": x_test.copy(), "y_test": y_test.copy()}

    # استفاده از SMOTE برای متعادل‌سازی داده‌های آموزشی
    x_train, y_train = apply_smote(x_train, y_train)
    predicted_probabilities_test = train_lightgbm_model(x_train, y_train, x_test)
    # استخراج اطلاعات جریان نقدی برای محاسبه هزینه‌ها
    cash_flow_data = x_test[protected_columns]
    false_positive_loss_test, false_negative_loss_test = compute_financial_losses(cash_flow_data)
    # بهینه‌سازی آستانه‌ها با استفاده از NSGA-II
    optimized_upper_threshold_scale, optimized_lower_threshold_scale = optimize_threshold_scales(
        predicted_probabilities_test, false_positive_loss_test, false_negative_loss_test, population_size=100, num_generations=200
    )
    logging.info("بهترین مقدار برای مقیاس آستانه بالا: " + format(Decimal(optimized_upper_threshold_scale), '.20f'))
    logging.info("بهترین مقدار برای مقیاس آستانه پایین: " + format(Decimal(optimized_lower_threshold_scale), '.20f'))

    # اعمال تصمیم‌گیری سه‌راهه
    three_way_decision_labels, uncertain_boundary_sample_indices = apply_three_way_decision(
        predicted_probabilities_test,
        false_positive_loss_test,
        false_negative_loss_test,
        optimized_upper_threshold_scale,
        optimized_lower_threshold_scale
    )
    # برای نمونه‌های حوزه تأخیر از مدل استکینگ استفاده می‌شود
    classifier = get_classifier('bagging')
    if len(uncertain_boundary_sample_indices) > 0:
        x_test_boundary_samples = x_test.iloc[uncertain_boundary_sample_indices]
        classifier.fit(x_train, y_train)
        predicted_labels_for_boundary_samples = classifier.predict(x_test_boundary_samples)
        three_way_decision_labels[uncertain_boundary_sample_indices] = predicted_labels_for_boundary_samples

    myRes = evaluate_model_performance(
        np.array(y_test),
        np.array(three_way_decision_labels),
        false_positive_loss_test,
        false_negative_loss_test
    )

    models = {
        "Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "LR": LogisticRegression(max_iter=3000),
        "NN": MLPClassifier(max_iter=300),
        "AdaBoost": AdaBoostClassifier(algorithm="SAMME"),
        "ERT": ExtraTreesClassifier(),
        "GBDT": GradientBoostingClassifier(),
        "LGBM": LGBMClassifier(verbose=-1),
        "RF": RandomForestClassifier(),
        "XGB": XGBClassifier(eval_metric='logloss', verbosity=0),
        "Stacking": StackingClassifier(estimators=[
            ('lda', LinearDiscriminantAnalysis()),
            ('knn', KNeighborsClassifier())
        ], final_estimator=RandomForestClassifier())
    }
    for name, model in models.items():
        logging.info(f"در حال آموزش و ارزیابی مدل: {name}")
        metrics = train_and_evaluate(model, x_train, y_train, x_test, y_test, b=1, cost_fp=1, cost_fn=1)
        results[name] = metrics
        logging.info(f"نتایج مدل {name}: {metrics}")
    results["myModel"] = {
        "Balanced Accuracy": myRes["Balanced Accuracy"],
        "AUC": myRes["AUC"],
        "F-Measure": myRes["FM"],
        "G-Mean": myRes["GM"],
        "Cost": myRes["Decision Cost"],
        "TP": myRes["TP"],
        "TN": myRes["TN"],
        "FP": myRes["FP"],
        "FN": myRes["FN"]
    }
    logging.error("نتایج کلی:")
    for name, metric in results.items():
        logging.info(f"{name}: {metric}")
