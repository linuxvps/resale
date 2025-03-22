import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, BigInteger, String, Date, DateTime, Numeric, Float, Text, SmallInteger
from sqlalchemy.orm import declarative_base, Session, sessionmaker
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from decimal import Decimal

# تعریف پایه مدل SQLAlchemy
Base = declarative_base()

# protected_columns از قبل تعریف شده برای استفاده در پیش‌پردازش
protected_columns = ['approval_amount', 'interest_amount']

# ==================== تعریف مدل ParsianLoan ====================
class ParsianLoan(Base):
    __tablename__ = "parsian_loan"

    id = Column("id", BigInteger, primary_key=True, autoincrement=True)
    insert_sysdate = Column("insert_sysdate", DateTime, nullable=False, default=datetime.utcnow)
    branch_code = Column("branch_code", Integer, nullable=False)
    branchname = Column("branchname", String(100), nullable=True)
    client_id = Column("client_id", Integer, nullable=True)
    loan_file_numberr = Column("loan_file_numberr", BigInteger, nullable=True)
    sit_flag = Column("sit_flag", String(1), nullable=True)
    interest_rate = Column("interest_rate", Numeric(19, 2), nullable=True)
    total_repayment_up_to_now = Column("total_repayment_up_to_now", Numeric(28, 8), nullable=True)
    commission_amount_remain = Column("commission_amount_remain", Numeric(28, 8), nullable=True)
    charge = Column("charge", Numeric(28, 8), nullable=True)
    discount = Column("discount", Numeric(28, 8), nullable=True)
    advance_pay_to_total_cash = Column("advance_pay_to_total_cash", Numeric(28, 8), nullable=True)
    advance_pay_to_remain_non_cash = Column("advance_pay_to_remain_non_cash", Numeric(28, 8), nullable=True)
    is_installment = Column("is_installment", String(1), nullable=True)
    interest_sum = Column("interest_sum", Numeric(28, 8), nullable=True)
    installment_number_remain = Column("installment_number_remain", Integer, nullable=True)
    receivable_installment_number = Column("receivable_installment_number", Integer, nullable=True)
    first_passed = Column("first_passed", Date, nullable=True)
    total_payment_up_to_now = Column("total_payment_up_to_now", Numeric(28, 8), nullable=True)
    finalized_loan_amount = Column("finalized_loan_amount", Numeric(28, 8), nullable=True)
    penalty = Column("penalty", Numeric(28, 8), nullable=True)
    first_payment_date_in_du = Column("first_payment_date_in_du", Date, nullable=True)
    principal_sum = Column("principal_sum", Numeric(28, 8), nullable=True)
    advance_pay = Column("advance_pay", Numeric(28, 8), nullable=True)
    sit_duration = Column("sit_duration", Integer, nullable=True)
    sit_duration_day = Column("sit_duration_day", Integer, nullable=True)
    sit_distribute_phases = Column("sit_distribute_phases", Integer, nullable=True)
    sit_fast_receive_percent = Column("sit_fast_receive_percent", Float, nullable=True)
    frequency = Column("frequency", Integer, nullable=True)
    customer_obligation_amount = Column("customer_obligation_amount", Numeric(28, 8), nullable=True)
    customer_share_cash_amount = Column("customer_share_cash_amount", Numeric(28, 8), nullable=True)
    customer_share_non_cash_amount = Column("customer_share_non_cash_amount", Numeric(28, 8), nullable=True)
    bank_share_cash_amount = Column("bank_share_cash_amount", Numeric(28, 8), nullable=True)
    bank_share_non_cash_amount = Column("bank_share_non_cash_amount", Numeric(28, 8), nullable=True)
    first_over_due = Column("first_over_due", Date, nullable=True)
    loan_duration_day = Column("loan_duration_day", Integer, nullable=True)
    loan_file_number = Column("loan_file_number", BigInteger, nullable=True)
    create_date = Column("create_date", Date, nullable=True)
    long_title = Column("long_title", String(255), nullable=True)
    status = Column("status", String(255), nullable=True)
    contract = Column("contract", String(255), nullable=True)
    approval_amount = Column("approval_amount", Numeric(28, 8), nullable=True)
    title = Column("title", String(255), nullable=True)
    inc_commission_amount = Column("inc_commission_amount", Numeric(28, 8), nullable=True)
    interest_amount = Column("interest_amount", Numeric(28, 8), nullable=True)
    obligation_penalty = Column("obligation_penalty", Numeric(28, 8), nullable=True)
    passed_date = Column("passed_date", Date, nullable=True)
    penalty_interest = Column("penalty_interest", Numeric(28, 8), nullable=True)
    to_due_date = Column("to_due_date", Numeric(28, 8), nullable=True)
    to_end_of_month = Column("to_end_of_month", Numeric(28, 8), nullable=True)
    due_date = Column("due_date", Date, nullable=True)

    def __repr__(self):
        return f"<ParsianLoan(id={self.id}, branch_code={self.branch_code}, client_id={self.client_id})>"

# ==================== تعریف مدل LoanFeatures ====================
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
    """مدیریت عملیات خواندن داده‌ها از پایگاه داده."""

    def __init__(self):
        self.session = self.create_database_session()

    @staticmethod
    def create_database_session() -> Session:
        engine = create_engine("mysql+pymysql://root:pass@localhost:3306/ln")
        SessionLocal = sessionmaker(bind=engine)
        return SessionLocal()

    def fetch_loans(self, limit: int = 10000) -> pd.DataFrame:
        loans = self.session.query(ParsianLoan).limit(limit).all()
        if not loans:
            print("⚠️ هیچ داده‌ای از پایگاه داده دریافت نشد.")
            return pd.DataFrame()
        df = pd.DataFrame([loan.__dict__ for loan in loans])
        df.drop(columns=["_sa_instance_state"], inplace=True, errors="ignore")
        print(f"✅ {len(df)} رکورد از دیتابیس دریافت شد.")
        return df

# ==================== تعریف کلاس LoanPreprocessor ====================
class LoanPreprocessor:
    """کلاس مسئول پیش‌پردازش داده‌های وام."""

    def __init__(self, imputation_strategy: str = "mean"):
        self.imputer = SimpleImputer(strategy=imputation_strategy)

    def convert_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.datetime64):
                df[col] = (pd.to_datetime(df[col]) - pd.Timestamp("2000-01-01")).dt.days
            elif df[col].dtype == 'object':
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
        print(f"🟢 ستون برچسب انتخاب شده: {label_column}")
        print("🔹 مقدارهای `status` قبل از تبدیل:")
        print(df[label_column].value_counts())
        default_statuses = {'مشكوك الوصول', 'معوق', 'سررسيد گذشته'}
        df[label_column] = df[label_column].apply(lambda x: 1 if x in default_statuses else 0)
        print("🔹 تعداد برچسب‌های نکول و غیرنکول پس از تبدیل:")
        print(df[label_column].value_counts())
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
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    print(f"🔴 همبستگی بالا بین: {col_i} و {col_j} | مقدار: {corr_value:.4f} | حذف: {col_j}")
                    attributes_to_remove.add(col_j)
        for col in attributes_to_remove:
            if class_column and col == class_column:
                continue
            if protected_columns and col in protected_columns:
                continue
            print(f"✅ ویژگی حذف شد: {col}")
            new_data.drop(columns=[col], inplace=True)
        print("ماتریس همبستگی:")
        print(corr_matrix.to_string())
        return new_data

    def preprocess(self, df: pd.DataFrame, label_column: str = "status") -> (pd.DataFrame, pd.Series):
        df = self.convert_labels(df, label_column)
        df = self.convert_dataframe_columns(df)
        df.drop(columns=["create_date"], errors="ignore", inplace=True)
        df = self.remove_highly_correlated_features(df, threshold=0.9, class_column=label_column)
        df_imputed = pd.DataFrame(self.imputer.fit_transform(df), columns=df.columns)
        X = df_imputed.drop(columns=[label_column])
        y = df_imputed[label_column]
        return X, y

# ==================== تعریف کلاس LoanDataHandler ====================
class LoanDataHandler:
    """مدیریت دریافت، پردازش و تقسیم داده‌های وام به مجموعه‌های آموزش و تست."""

    def __init__(self, repository: LoanRepository, preprocessor: LoanPreprocessor):
        self.repository = repository
        self.preprocessor = preprocessor

    def load_and_process_data(self, limit_records: int = 10000) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
        df = self.repository.fetch_loans(limit_records)
        x, y = self.preprocessor.preprocess(df)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        return x_train, y_train, x_test, y_test

# ==================== تعریف کلاس ThresholdOptimizationProblem و تابع optimize_threshold_scales ====================
class ThresholdOptimizationProblem(Problem):
    """
    مسئله تعیین مقیاس‌های تنظیمی برای محاسبه آستانه‌های تصمیم‌گیری.
    """

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

# ==================== توابع مربوط به مدل و ارزیابی ====================
def pre_process_data_from_db():
    loan_repository = LoanRepository()
    loan_preprocessor = LoanPreprocessor(imputation_strategy="median")
    loan_data_handler = LoanDataHandler(loan_repository, loan_preprocessor)
    return loan_data_handler.load_and_process_data(limit_records=100_000)

def train_lightgbm_model(x_train, y_train, x_test):
    lightgbm_classifier = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    print("شروع آموزش مدل LightGBM...")
    lightgbm_classifier.fit(x_train, y_train)
    print("آموزش مدل به پایان رسید.")
    predicted_probabilities = lightgbm_classifier.predict_proba(x_test)[:, 1]
    return predicted_probabilities

def compute_financial_losses(cash_flow_info):
    principal_amount = cash_flow_info['approval_amount'].values
    interest_amount = cash_flow_info['interest_amount'].values
    false_positive_loss = interest_amount
    false_negative_loss = principal_amount + interest_amount
    return false_positive_loss, false_negative_loss

def apply_three_way_decision(predicted_probabilities, false_positive_loss, false_negative_loss, upper_threshold_scale, lower_threshold_scale):
    boundary_penalty_positive = upper_threshold_scale * false_negative_loss
    boundary_penalty_negative = lower_threshold_scale * false_positive_loss
    numerator_alpha = false_positive_loss - boundary_penalty_negative
    denominator_alpha = numerator_alpha + boundary_penalty_positive
    alpha_threshold = np.where(denominator_alpha == 0, 1.0, numerator_alpha / denominator_alpha)
    numerator_beta = boundary_penalty_negative
    denominator_beta = boundary_penalty_negative + (false_negative_loss - boundary_penalty_positive)
    beta_threshold = np.where(denominator_beta == 0, 0.0, numerator_beta / denominator_beta)
    three_way_decision_labels = np.where(predicted_probabilities >= alpha_threshold, 1, np.where(predicted_probabilities <= beta_threshold, 0, -1))
    uncertain_boundary_sample_indices = np.where(three_way_decision_labels == -1)[0]
    return three_way_decision_labels, uncertain_boundary_sample_indices

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
    decision_cost = np.sum(false_negative_loss[(true_labels == 1) & (predicted_labels == 0)]) + np.sum(false_positive_loss[(true_labels == 0) & (predicted_labels == 1)])
    print("Balanced Accuracy:", balanced_accuracy)
    print("AUC:", area_under_curve)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", confusion_mat)
    print("Classification Report:\n", classification_rep)
    print("Decision Cost:", decision_cost)

# ==================== اجرای کل فرآیند ====================
if __name__ == "__main__":
    os.environ["LOKY_MAX_CPU_COUNT"] = "8"
    x_train_resampled, y_train_resampled, x_test, y_test = pre_process_data_from_db()
    predicted_probabilities_test = train_lightgbm_model(x_train_resampled, y_train_resampled, x_test)
    cash_flow_data = x_test[protected_columns]
    false_positive_loss_test, false_negative_loss_test = compute_financial_losses(cash_flow_data)
    optimized_upper_threshold_scale, optimized_lower_threshold_scale = optimize_threshold_scales(
        predicted_probabilities_test, false_positive_loss_test, false_negative_loss_test, population_size=20, num_generations=10
    )
    print("بهترین مقدار برای مقیاس آستانه بالا:", Decimal(optimized_upper_threshold_scale))
    print("بهترین مقدار برای مقیاس آستانه پایین:", Decimal(optimized_lower_threshold_scale))
    three_way_decision_labels, uncertain_boundary_sample_indices = apply_three_way_decision(
        predicted_probabilities_test, false_positive_loss_test, false_negative_loss_test, optimized_upper_threshold_scale, optimized_lower_threshold_scale
    )
    ensemble_bagging_classifier = BaggingClassifier(
        estimator=ExtraTreesClassifier(n_estimators=100, random_state=42),
        n_estimators=10,
        random_state=42
    )
    ensemble_bagging_classifier.fit(x_train_resampled, y_train_resampled)
    x_test_boundary_samples = x_test.iloc[uncertain_boundary_sample_indices]
    predicted_labels_for_boundary_samples = ensemble_bagging_classifier.predict(x_test_boundary_samples)
    three_way_decision_labels[uncertain_boundary_sample_indices] = predicted_labels_for_boundary_samples
    evaluate_model_performance(np.array(y_test), np.array(three_way_decision_labels), false_positive_loss_test, false_negative_loss_test)
