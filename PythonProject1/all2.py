import logging
import os
from datetime import datetime

import pandas as pd
from colorlog import ColoredFormatter
# ------------------------------------------------------------
# بخش مربوط به SQLAlchemy برای اتصال به دیتابیس و تعریف انتیتی
# ------------------------------------------------------------
from sqlalchemy import create_engine, Column, Integer, BigInteger, String, Date, DateTime, Numeric, Float
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
protected_columns = ['approval_amount', 'interest_amount']

formatter = ColoredFormatter("%(log_color)s%(asctime)s - %(levelname)s - %(message)s", datefmt=None, reset=True,
    log_colors={'DEBUG': 'cyan', 'INFO': 'white', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'bold_red', })

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

###########################################
# پلات
###########################################
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


class Plot:
    """
    یک کلاس عمومی برای بصری‌سازی توزیع احتمال‌ها.
    """

    def __init__(self) -> None:
        pass

    def visualize_distribution(self, probabilities: np.ndarray, bins: int = 20,
                               figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        نمودار هیستوگرام توزیع احتمال‌ها را بر اساس آرایه ورودی رسم می‌کند.
        با افزودن جزئیات مانند خط میانگین، برچسب‌های دقیق محور و تنظیمات grid.

        :param probabilities: آرایه numpy شامل احتمال‌ها.
        :param bins: تعداد بخش‌های هیستوگرام (پیش‌فرض 20).
        :param figsize: اندازه شکل نمودار (پیش‌فرض (10, 6)).
        """
        plt.figure(figsize=figsize)

        # رسم هیستوگرام
        n, bins, patches = plt.hist(probabilities, bins=bins, edgecolor='black',
                                    alpha=0.7, color='skyblue')

        # محاسبه و نمایش خط میانگین
        mean_val = np.mean(probabilities)
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2,
                    label=f'Mean = {mean_val:.2f}')

        # افزودن جزئیات به نمودار
        plt.title("Distribution of Default Probabilities", fontsize=16)
        plt.xlabel("Probability", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()


class ParsianLoan(Base):
    __tablename__ = "parsian_loan"

    # ستون‌ها مطابق دیتابیس parsian_loan
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


class LoanRepository:
    """
    ریپازیتوری برای واکشی داده از دیتابیس parsian_loan با SQLAlchemy.
    """

    def __init__(self):
        # فرض کنید Connection String را در متغیرمحیطی DB_CONNECTION_STRING می‌گذاریم.
        self.db_conn_str = os.getenv("DB_CONNECTION_STRING", "mysql+pymysql://root:pass@localhost:3306/ln")
        self.engine = create_engine(self.db_conn_str)
        SessionLocal = sessionmaker(bind=self.engine)
        self.session = SessionLocal()

    def fetch_loans(self, limit=10_000):
        """
        واکشی حداکثر `limit` رکورد از جدول parsian_loan.
        داده‌ها در قالب یک DataFrame برگردانده می‌شوند.
        """
        loans = self.session.query(ParsianLoan).limit(limit).all()
        if not loans:
            logging.warning("هیچ داده‌ای از پایگاه داده دریافت نشد.")
            return pd.DataFrame()

        columns = [c.name for c in ParsianLoan.__table__.columns]
        data = {}
        for col in columns:
            data[col] = [getattr(loan, col) for loan in loans]
        df = pd.DataFrame(data)
        logging.info(f"✅ {len(df)} رکورد از دیتابیس دریافت شد (parsian_loan).")
        return df


###########################################
# گام دوم: برآورد احتمال نکول (Default Probability)
###########################################

class ParsianDefaultProbabilityModel:
    """
    یک کلاس برای پیاده‌سازی گام دوم در pseudocodeorg:
    - آموزش یک مدل (مثلاً LightGBM) بر روی داده‌های آموزش
    - محاسبه احتمال نکول برای داده‌های آزمون
    """

    def __init__(self, model_type="lightgbm", n_estimators=100, learning_rate=0.05, random_state=42, **kwargs):
        """
        پارامترها:
          - model_type: نوع مدل (lightgbm, xgboost یا هر مدل دیگری)
          - n_estimators, learning_rate: هایپراستعداد برای مدل
          - random_state: برای بازتولید پذیری
          - kwargs: سایر پارامترهای دلخواه مدل
        """

        self.model_type = model_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.default_probabilities_ = None  # احتمال نکول برای نمونه‌های test

        self.kwargs = kwargs

    def fit_model(self, x_train, y_train):
        """
        آموزش مدل بر اساس نوع انتخابی:
        در این مثال، پیش‌فرض LightGBM است.
        """
        if self.model_type.lower() == "lightgbm":
            logging.info("🔵 در حال آموزش مدل LightGBM...")
            self.model = LGBMClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                random_state=self.random_state, **self.kwargs)
        else:
            raise ValueError("فعلاً فقط مدل lightgbm پشتیبانی می‌شود (برای مثال).")

        self.model.fit(x_train, y_train)
        logging.info("✅ آموزش مدل به پایان رسید.")

    def predict_default_probability(self, x_test):
        """
        محاسبه احتمال نکول برای نمونه‌های تست.
        خروجی: یک آرایه NumPy از مقادیر احتمال (بین ۰ و ۱)
        """
        if not self.model:
            raise ValueError("ابتدا باید مدل را آموزش دهید (fit_model).")

        logging.info("🔵 در حال محاسبه احتمال نکول نمونه‌های آزمون...")
        probs = self.model.predict_proba(x_test)
        # چون خروجی predict_proba دو ستون [Prob_of_Class_0, Prob_of_Class_1] است،
        # احتمال نکول معمولاً ستون دوم (کلاس ۱) است.
        self.default_probabilities_ = probs[:, 1]
        return self.default_probabilities_

    def get_model(self):
        """ برگرداندن شیء مدل آموزش‌دیده. """
        return self.model


# ------------------------------------------------------------
# گام اول: Preprocessing Manager و Preprocessor
# ------------------------------------------------------------
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


class LoanPreprocessor:
    """
    کلاس مسئول عملیات پیش‌پردازش:
    - تبدیل برچسب (status) به نکول/غیرنکول (۰ یا ۱)
    - تبدیل انواع ستون‌ها به عددی یا زمان
    - حذف ویژگی‌های با همبستگی بالا
    - انتخاب ویژگی با کمک LGBM + RFECV
    - ایمپیوت داده (SimpleImputer)
    """

    def __init__(self, imputation_strategy="mean"):
        self.imputer = SimpleImputer(strategy=imputation_strategy)

    def convert_labels(self, df, label_column="status"):
        logging.info(f"[LoanPreprocessor] تبدیل برچسب: {label_column}")
        if label_column not in df.columns:
            raise ValueError(f"ستون {label_column} در داده وجود ندارد.")

        # فرض بر این است که مقادیر {"مشكوك الوصول", "معوق", "سررسيد گذشته"} => 1
        default_statuses = {"مشكوك الوصول", "معوق", "سررسيد گذشته"}
        df[label_column] = df[label_column].apply(lambda x: 1 if x in default_statuses else 0)
        return df

    def convert_dataframe_columns(self, df):
        """
        تلاش برای تبدیل ستون‌های تاریخ به timestamp و ستون‌های object به numeric.
        در غیر این صورت، از LabelEncoder استفاده می‌شود.
        """
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.datetime64):
                df[col] = pd.to_datetime(df[col]).apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
        return df

    def remove_highly_correlated_features(self, data, threshold=0.9, class_column=None):
        new_data = data.copy()
        numeric_cols = new_data.select_dtypes(include=[np.number]).columns.tolist()
        if class_column and class_column in numeric_cols:
            numeric_cols.remove(class_column)

        protected_columns = ["approval_amount", "interest_amount"]  # ستون‌هایی که نباید حذف شوند
        numeric_cols = [c for c in numeric_cols if c not in protected_columns]

        corr_matrix = new_data[numeric_cols].corr()
        attributes_to_remove = set()
        for i in range(len(numeric_cols) - 1):
            for j in range(i + 1, len(numeric_cols)):
                col_i = numeric_cols[i]
                col_j = numeric_cols[j]
                corr_value = corr_matrix.loc[col_i, col_j]
                if abs(corr_value) > threshold:
                    logging.info(f"حذف ویژگی با همبستگی بالا: {col_i} - {col_j} => {corr_value:.2f}, حذف {col_j}")
                    attributes_to_remove.add(col_j)

        for col_r in attributes_to_remove:
            if col_r in new_data.columns:
                new_data.drop(columns=[col_r], inplace=True)

        return new_data

    def select_features(self, X, y):
        lgbm_estimator = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        rfecv = RFECV(estimator=lgbm_estimator, step=1, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
        rfecv.fit(X, y)
        selected_features = list(X.columns[rfecv.support_])
        for col in protected_columns:
            if col in X.columns and col not in selected_features:
                selected_features.append(col)
        not_selected_features = [col for col in X.columns if col not in selected_features]
        logging.info("ویژگی‌های انتخاب شده: " + ", ".join(selected_features))
        logging.info("ویژگی‌های انتخاب نشده: " + ", ".join(not_selected_features))
        return X.loc[:, selected_features]


class ParsianPreprocessingManager:
    """
    یک کلاس جامع برای گام اول (Preprocessing) از pseudocodeorg.
    """

    def __init__(self, repository, limit_records=10000, label_column="status", imputation_strategy="mean",
            need_2_remove_highly_correlated_features=False, correlation_threshold=0.9, do_balance=True, test_size=0.2,
            random_state=42):
        self.repository = repository
        self.limit_records = limit_records
        self.label_column = label_column
        self.imputation_strategy = imputation_strategy
        self.correlation_threshold = correlation_threshold
        self.need_2_remove_highly_correlated_features = need_2_remove_highly_correlated_features
        self.do_balance = do_balance
        self.test_size = test_size
        self.random_state = random_state

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.original_df = None
        self.preprocessor = None

    def step1_process_data(self):
        """
        1) واکشی داده از parsian_loan
        2) تبدیل برچسب نکول => 0/1
        3) تبدیل انواع ستون => عددی
        4) حذف ویژگی‌های با همبستگی بالا
        5) ایمپیوت
        6) تفکیک X,y
        7) train_test_split
        8) انتخاب ویژگی
        9) متعادل‌سازی SMOTE (در صورت نیاز)

        خروجی: (x_train, y_train, x_test, y_test, original_df)
        """
        logging.info("🔵 [Step1] شروع آماده‌سازی داده‌ها (Preprocessing).")

        df = self.repository.fetch_loans(limit=self.limit_records)
        if df.empty:
            logging.error("هیچ داده‌ای دریافت نشد. فرآیند پایان یافت.")
            return None, None, None, None, None
        logging.info(f"تعداد رکورد دریافتی: {len(df)}")

        self.preprocessor = LoanPreprocessor(imputation_strategy=self.imputation_strategy)

        # برچسب نکول
        df = self.preprocessor.convert_labels(df, label_column=self.label_column)

        # تبدیل انواع ستون
        df = self.preprocessor.convert_dataframe_columns(df)

        # مثال برای حذف برخی ستون‌ها
        drop_columns = ["create_date"]  # در صورت نیاز می‌توان تغییر داد
        for col_d in drop_columns:
            if col_d in df.columns:
                df.drop(columns=[col_d], inplace=True, errors="ignore")

        # حذف ویژگی‌های با همبستگی بالا
        if self.need_2_remove_highly_correlated_features:
            df = self.preprocessor.remove_highly_correlated_features(df, threshold=self.correlation_threshold,
                class_column=self.label_column)

        # ایمپیوت
        df_imputed = pd.DataFrame(self.preprocessor.imputer.fit_transform(df), columns=df.columns)

        # تفکیک X,y
        X = df_imputed.drop(columns=[self.label_column])
        y = df_imputed[self.label_column]

        # تقسیم آموزش/آزمون
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
            random_state=self.random_state)

        # انتخاب ویژگی
        x_train_selected = self.preprocessor.select_features(x_train, y_train)
        x_test_selected = x_test[x_train_selected.columns]

        # متعادل‌سازی SMOTE
        if self.do_balance:
            logging.info("🔵 اعمال SMOTE برای بالانس کلاس‌ها")
            x_train_bal, y_train_bal = SMOTE(random_state=self.random_state).fit_resample(x_train_selected, y_train)
        else:
            logging.info("🔵 عدم استفاده از SMOTE")
            x_train_bal, y_train_bal = x_train_selected, y_train

        # ذخیره خروجی
        self.x_train = x_train_bal
        self.y_train = y_train_bal
        self.x_test = x_test_selected
        self.y_test = y_test
        self.original_df = df.copy()

        logging.info("✅ [Step1] مرحله اول پیش‌پردازش به پایان رسید.")
        return (self.x_train, self.y_train, self.x_test, self.y_test, self.original_df)


###########################################
# گام سوم: محاسبه ماتریس زیان (Cost Computation)
###########################################
class ParsianCostMatrix:
    """
    در این کلاس، بر اساس داده‌های جریان نقدی (مثلاً approval_amount, interest_amount)
    هزینه‌های تصمیم‌گیری برای هر نمونه محاسبه می‌شود.

    فرض می‌کنیم dataframe تست (x_test) را داشته باشیم که ستون‌های مالی در آن وجود دارد.
    مثلاً approval_amount, interest_amount.

    سپس چهار نوع زیان اصلی را تعریف می‌کنیم:
      λ_PP, λ_NN (معمولاً 0)
      λ_PN (زیان پذیرش اشتباه)
      λ_NP (زیان رد اشتباه)

    کاربر می‌تواند بسته به نیازش این فرمول‌ها را شخصی‌سازی کند.
    """

    def __init__(self, df_test: pd.DataFrame, approval_col="approval_amount", interest_col="interest_amount"):
        """
        پارامترها:
          - df_test: داده‌های آزمون، شامل ستون‌های مالی لازم
          - approval_col: نام ستونی که اصل وام را نگه می‌دارد (approval_amount)
          - interest_col: نام ستونی که سود بالقوه یا مبلغ بهره را نشان می‌دهد (interest_amount)
        """
        self.df_test = df_test.reset_index(drop=True)  # اگر لازم است ایندکس ریست شود
        self.approval_col = approval_col
        self.interest_col = interest_col

        # در این دیکشنری هزینه‌های تصمیم را برای هر نمونه نگه می‌داریم
        # مثال: cost_matrix[i] = {
        #    "PP": val, "PN": val, "BP": val, "BN": val, "NP": val, "NN": val
        # }
        self.cost_matrix = []

    def compute_costs(self):
        """
        محاسبه زیان برای هر رکورد در df_test.
        در این مثال، فرمول‌ها را بدین صورت تعریف می‌کنیم:
          λ_PP = 0
          λ_NN = 0
          λ_PN = interest_amount
          λ_NP = approval_amount + interest_amount
          (اگر بخواهیم λ_BP و λ_BN هم داشته باشیم، می‌توانیم اضافه کنیم)
        """
        if self.approval_col not in self.df_test.columns or self.interest_col not in self.df_test.columns:
            raise ValueError("ستون‌های مالی برای محاسبه زیان موجود نیست.")

        for i in range(len(self.df_test)):
            principal = float(self.df_test.loc[i, self.approval_col] or 0.0)
            interest = float(self.df_test.loc[i, self.interest_col] or 0.0)

            cost_pp = 0.0
            cost_nn = 0.0
            cost_pn = interest  # پذیرش اشتباه نمونه‌ای که واقعاً غیرنکول است
            cost_np = principal + interest  # رد اشتباه نمونه‌ای که واقعاً نکول است

            # اگر بخواهیم هزینه تصمیم مرزی اضافه کنیم:
            # cost_bp = ...
            # cost_bn = ...

            self.cost_matrix.append({"PP": cost_pp, "NN": cost_nn, "PN": cost_pn, "NP": cost_np# "BP": cost_bp,
                # "BN": cost_bn
            })

    def get_cost_for_sample(self, index: int):
        """
        گرفتن هزینه تصمیم‌ها برای رکورد iام.
        خروجی یک دیکشنری با کلیدهای PP, PN, NP, NN (و در صورت وجود BP, BN)
        """
        return self.cost_matrix[index]

    def get_all_costs(self):
        """ برمی‌گرداند کل cost_matrix به صورت لیستی از دیکشنری‌ها. """
        return self.cost_matrix


###########################################
# گام چهارم: بهینه‌سازی چندهدفه آستانه‌ها با NSGA-II (pymoo)
###########################################
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize


class ParsianThresholdNSGA2:
    """
    یک کلاس برای پیاده‌سازی گام چهارم تصمیم‌گیری سه‌طرفه به صورت چندهدفه:
      - هدف اول: کمینه‌کردن هزینه تصمیم‌گیری
      - هدف دوم: کمینه‌کردن اندازه ناحیه مرزی (BND)
      با استفاده از الگوریتم NSGA-II از کتابخانه pymoo.

    ورودی‌ها:
      - probabilities_test: آرایه احتمال نکول نمونه‌های تست
      - cost_matrix: آرایه یا لیستی از دیکشنری هزینه برای هر رکورد
           مثال: cost_matrix[i] = {"PP": cost_if_true_and_decide_positive,
                                   "PN": cost_if_false_and_decide_positive,
                                   "NP": cost_if_true_and_decide_negative,
                                   "NN": cost_if_false_and_decide_negative,
                                   ...}
      - true_labels: آرایه برچسب واقعی (۰ یا ۱)
      - pop_size: اندازه جمعیت برای NSGA2
      - n_gen: تعداد نسل (iteration) برای NSGA2
    """

    def __init__(self, probabilities_test: np.ndarray, cost_matrix: list, true_labels: np.ndarray, pop_size=50,
            n_gen=100, step_bnd=False):
        """
        پارامترها:
         - step_bnd: اگر True باشد، objective دوم را تعداد نمونه‌های BND در نظر می‌گیریم
                     اگر False باشد، نسبت نمونه‌های BND به کل را در نظر می‌گیریم
                     (هر دو رویکرد ممکن است.)
        """
        self.probabilities_test = probabilities_test  # احتمال نکول
        self.cost_matrix = cost_matrix  # زیان هر رکورد
        self.true_labels = true_labels
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.step_bnd = step_bnd

        self.best_solutions = None  # راه‌حل‌های پارتو به‌دست‌آمده
        self.front_costs = None  # مقدار اهداف در پارتو
        self.problem_instance = None

    def _decision_cost_for_sample(self, i, alpha, beta):
        """
        محاسبه هزینه تصمیم برای نمونه iام.
        سه حالت:
         - اگر p_i >= alpha => POS
         - اگر p_i <= beta => NEG
         - در غیر این صورت => BND (اینجا هزینه را می‌توان سفارشی کرد.)
        """
        p = self.probabilities_test[i]
        y_true = self.true_labels[i]  # 0 یا 1
        costs = self.cost_matrix[i]

        if p >= alpha:
            # تصمیم => نکول
            return costs["PP"] if y_true == 1 else costs["PN"]
        elif p <= beta:
            # تصمیم => غیرنکول
            return costs["NP"] if y_true == 1 else costs["NN"]
        else:
            # مرزی
            # اگر بخواهید هزینه مرزی را هم اضافه کنید، باید آن را در cost_matrix[i] گنجانید
            # مثلاً costs["BP"], costs["BN"] ...
            # در اینجا برای سادگی 0 در نظر می‌گیریم
            return 0.0

    def _boundary_count_for_solution(self, alpha, beta):
        """
        تعداد (یا نسبت) نمونه‌هایی که در ناحیه مرزی قرار می‌گیرند:
         یعنی sample i که p_i ∈ (beta, alpha)
        """
        p = self.probabilities_test
        bnd_mask = (p > beta) & (p < alpha)
        bnd_count = np.sum(bnd_mask)
        if self.step_bnd:
            return bnd_count  # تعداد نمونه‌های مرزی
        else:
            return bnd_count / len(p)  # نسبت نمونه‌های مرزی به کل

    class ThresholdOptimizationProblem(Problem):
        """
        کلاس مسئله pymoo برای NSGA-II.
        n_var=2 => (alpha, beta)
        n_obj=2 => هدف اول: هزینه، هدف دوم: اندازه ناحیه مرزی
        n_constr=1 => alpha >= beta  => beta - alpha <= 0
        xl=[0,0], xu=[1,1] => alpha,beta ∈ [0,1]
        """

        def __init__(self, outer, ):
            """
            - outer: یک اشاره به کلاس بیرونی ParsianThresholdNSGA2
                     تا بتوانیم از داده‌های probabilities_test و ... استفاده کنیم.
            """
            super().__init__(n_var=2, n_obj=2, n_constr=1, xl=np.array([0.0, 0.0]), xu=np.array([1.0, 1.0]),
                type_var=np.double)
            self.outer = outer  # ارجاع به کلاس بیرونی

        def _evaluate(self, X, out, *args, **kwargs):
            """
            X آرایه‌ای شکل (N, 2) است که در هر سطر:
              X[i,0] = alpha
              X[i,1] = beta
            باید 2 هدف محاسبه شود: total_cost, boundary_size
            همچنین یک محدودیت: alpha >= beta => beta - alpha <= 0
            """
            n_solutions = X.shape[0]
            f1 = np.zeros(n_solutions)  # هزینه
            f2 = np.zeros(n_solutions)  # اندازه مرزی

            for i_sol in range(n_solutions):
                alpha = X[i_sol, 0]
                beta = X[i_sol, 1]

                # محاسبه هزینه کل
                total_cost = 0.0
                for i_sample in range(len(self.outer.probabilities_test)):
                    c = self.outer._decision_cost_for_sample(i_sample, alpha, beta)
                    total_cost += c

                # محاسبه اندازه مرزی
                boundary_size = self.outer._boundary_count_for_solution(alpha, beta)

                f1[i_sol] = total_cost
                f2[i_sol] = boundary_size

            # محدودیت: alpha >= beta => (beta - alpha) <= 0
            g = np.zeros((n_solutions, 1))
            g[:, 0] = X[:, 1] - X[:, 0]  # beta - alpha => باید <= 0

            out["F"] = np.column_stack([f1, f2])
            out["G"] = g

    def optimize(self):
        """
        اجرای الگوریتم NSGA-II برای کمینه‌کردن [cost, boundary_size]
        با محدودیت alpha >= beta.
        """
        logging.info("🔵 شروع بهینه‌سازی چندهدفه آستانه‌ها با NSGA-II...")

        # ساخت نمونه مسئله
        self.problem_instance = self.ThresholdOptimizationProblem(self)

        # الگوریتم NSGA2
        algo = NSGA2(pop_size=self.pop_size)

        # اجرای بهینه‌سازی
        res = minimize(self.problem_instance, algo, ("n_gen", self.n_gen), seed=42, verbose=False)

        self.front_costs = res.F  # هدف‌های راه‌حل‌های پارتو
        self.best_solutions = res.X  # خود راه‌حل‌ها (alpha,beta) در پارتو
        logging.info("✅ NSGA-II به اتمام رسید. تعداد راه‌حل‌های پارتو: {}".format(len(self.front_costs)))

    def get_pareto_front(self):
        """
        برمی‌گرداند (solutions, objectives) = (self.best_solutions, self.front_costs)
        جایی که هر solutions[i] = [alpha_i, beta_i]
              each objectives[i] = [cost_i, boundary_i]
        """
        return self.best_solutions, self.front_costs

    def get_final_solution(self):
        """
        از میان راه‌حل‌های پارتو، راه‌حل نهایی (α, β) را برمی‌گرداند که تعداد نمونه‌های ناحیه مرزی را به حداقل رسانده است.
        خروجی: (final_solution, final_objectives)
          final_solution: آرایه [alpha, beta]
          final_objectives: آرایه [total_cost, boundary_size] مربوط به راه‌حل نهایی
        """
        solutions, objectives = self.get_pareto_front()
        # انتخاب راه‌حل با کمترین مقدار objective دوم (boundary_size)
        best_index = np.argmin(objectives[:, 1])
        final_solution = solutions[best_index]
        final_objectives = objectives[best_index]
        return final_solution, final_objectives


###########################################
# گام پنجم: تقسیم نمونه‌ها به POS/NEG/BND (Three-Way Decision)
###########################################

class ParsianThreeWayDecision:
    """
    در این گام، با داشتن احتمالات نکول p و آستانه‌های alpha و beta،
    هر نمونه را به یکی از سه دسته POS/NEG/BND تخصیص می‌دهیم.

    اگر:
      p_i >= alpha  =>  POS
      p_i <= beta   =>  NEG
      otherwise     =>  BND
    """

    def __init__(self, probabilities_test: np.ndarray, alpha: float, beta: float):
        """
        پارامترها:
          - probabilities_test: آرایه احتمال نکول برای داده‌های تست
          - alpha, beta: آستانه‌های تصمیم سه‌طرفه
        """
        self.probabilities_test = probabilities_test
        self.alpha = alpha
        self.beta = beta
        self.decisions = None  # لیبل نهایی هر نمونه: POS=1, NEG=0, BND=-1 (مثلاً)

    def apply_three_way_decision(self):
        """
        با توجه به alpha و beta، روی probabilities_test اجرا کرده
        و سه دسته‌بندی را بازمی‌گرداند.
        """
        n_samples = len(self.probabilities_test)
        decisions = np.zeros(n_samples, dtype=int)  # 0 => NEG, 1 => POS, -1 => BND

        for i in range(n_samples):
            p_i = self.probabilities_test[i]
            if p_i >= self.alpha:
                decisions[i] = 1  # POS
            elif p_i <= self.beta:
                decisions[i] = 0  # NEG
            else:
                decisions[i] = -1  # BND

        self.decisions = decisions
        return decisions

    def get_decisions(self):
        """
        اگر از قبل apply_three_way_decision را صدا زده باشیم،
        تصمیم نهایی را برمی‌گردانیم.
        خروجی یک آرایه با مقادیر {0=NEG, 1=POS, -1=BND}.
        """
        return self.decisions

    def get_decision_counts(self):
        """
        تعداد نمونه‌های هر دسته (POS=1, NEG=0, BND=-1) را محاسبه و برمی‌گرداند.
        خروجی: دیکشنری به شکل {1: تعداد POS, 0: تعداد NEG, -1: تعداد BND}
        """
        if self.decisions is None:
            self.apply_three_way_decision()
        unique, counts = np.unique(self.decisions, return_counts=True)
        return dict(zip(unique, counts))


###########################################
# گام ششم: تصمیم‌گیری نهایی روی نمونه‌های BND
#          (مثلاً با استکینگ یا مدل کمکی دیگر)
###########################################
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class ParsianBNDResolver:
    """
    این کلاس نمونه‌هایی را که در گام پنجم در ناحیه BND واقع شده‌اند،
    شناسایی و با یک مدل اضافی (مثلاً استکینگ) تصمیم قطعی (POS یا NEG) می‌گیرد.
    """

    def __init__(self, x_train_all: pd.DataFrame, y_train_all: pd.Series, model_type="stacking"):
        """
        پارامترها:
          - x_train_all, y_train_all: داده‌های آموزش اصلی (یا داده‌های مرزی خاص؟)
          - model_type: نوع مدلی که می‌خواهیم برای تشخیص نمونه‌های مرزی به‌کار بریم
                       (اینجا مثلاً "stacking" یا "bagging" یا ...)
        """
        self.x_train_all = x_train_all
        self.y_train_all = y_train_all
        self.model_type = model_type
        self.classifier = None

    def fit_bnd_model(self):
        """
        آموزش مدل استکینگ یا هر مدل دیگر بر روی کل دادهٔ آموزش.
        توجه: می‌توانیم بنا به نیاز، فقط نمونه‌های دشوار یا بخشی از دیتاست را استفاده کنیم.
        """
        if self.model_type.lower() == "stacking":
            # چندین مدل پایه + متا
            base_estimators = [("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
                ("xgb", XGBClassifier(eval_metric="logloss", random_state=42))]
            meta_estimator = LogisticRegression(max_iter=1000, random_state=42)
            self.classifier = StackingClassifier(estimators=base_estimators, final_estimator=meta_estimator, cv=5,
                n_jobs=-1)
        else:
            # می‌توانید روش‌های دیگر را اضافه کنید
            raise ValueError("فقط مدل 'stacking' پیاده شده است.")

        logging.info("🔵 در حال آموزش مدل BNDResolver (استکینگ)...")
        self.classifier.fit(self.x_train_all, self.y_train_all)
        logging.info("✅ آموزش مدل BNDResolver کامل شد.")

    def resolve_bnd_samples(self, x_test: pd.DataFrame, decisions_final: np.ndarray):
        """
        پارامترها:
         - x_test: داده‌های تست کامل
         - decisions_final: برچسب تصمیم گام پنجم (POS=1, NEG=0, BND=-1)
        خروجی:
         - decisions_updated: آرایه‌ی تصمیم نهایی که BNDها را نیز به POS یا NEG تبدیل می‌کند.
        """
        bnd_indices = np.where(decisions_final == -1)[0]
        logging.info(f"🔵 تعداد نمونه‌های BND: {len(bnd_indices)}")

        if len(bnd_indices) == 0:
            logging.info("هیچ نمونه مرزی وجود ندارد. تغییری اعمال نمی‌شود.")
            return decisions_final  # همان قبلی

        # استخراج نمونه‌های BND از x_test
        x_test_bnd = x_test.iloc[bnd_indices]
        # پیش‌بینی مدل ثانویه
        y_pred_bnd = self.classifier.predict(x_test_bnd)

        # به‌روزرسانی تصمیم نهایی
        decisions_updated = decisions_final.copy()
        for idx, pred in zip(bnd_indices, y_pred_bnd):
            decisions_updated[idx] = pred  # pred=0 => NEG, pred=1 => POS
        return decisions_updated


###########################################
# گام هفتم: ارزیابی نهایی و گزارش (Final Evaluation)
###########################################
import numpy as np
from sklearn.metrics import roc_auc_score


class ParsianFinalEvaluator:
    """
    در این کلاس، برچسب‌های نهایی تصمیم (بعد از گام ششم) را با برچسب واقعی
    مقایسه می‌کنیم و معیارهای عملکرد را محاسبه و گزارش می‌دهیم.
    """

    def __init__(self, true_labels: np.ndarray, final_decisions: np.ndarray, probabilities_test: np.ndarray = None,
            cost_matrix: list = None):
        """
        پارامترها:
          - true_labels: آرایه برچسب‌های واقعی تست (۰ یا ۱)
          - final_decisions: آرایه تصمیم‌های نهایی (۰=NEG, ۱=POS)
          - probabilities_test: اگر می‌خواهیم AUC یا معیارهای احتمالاتی را حساب کنیم
          - cost_matrix: اگر بخواهیم هزینه تصمیم‌گیری را نیز محاسبه نماییم. (مثلاً خروجی گام سوم)
        """
        self.true_labels = true_labels
        self.final_decisions = final_decisions
        self.probabilities_test = probabilities_test
        self.cost_matrix = cost_matrix

    def evaluate_metrics(self):
        """
        محاسبه معیارهای اصلی نظیر:
         - Balanced Accuracy
         - Precision, Recall, F1
         - AUC (در صورت وجود probabilities_test)
         - هزینه تصمیم (در صورت وجود cost_matrix)
        خروجی: دیکشنری شامل مقادیر معیارها
        """

        if len(self.true_labels) != len(self.final_decisions):
            raise ValueError("طول برچسب‌های واقعی با تصمیم‌های نهایی همخوانی ندارد.")

        # محاسبه confusion matrix
        cm = confusion_matrix(self.true_labels, self.final_decisions)
        # cm ساختار [[TN, FP], [FN, TP]]
        TN, FP, FN, TP = cm.ravel()

        # محاسبه Balanced Accuracy
        # = 0.5 * (TP/(TP+FN) + TN/(TN+FP))
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        b_acc = 0.5 * (sensitivity + specificity)

        # Precision, Recall, F1
        precision = precision_score(self.true_labels, self.final_decisions, zero_division=0)
        recall = recall_score(self.true_labels, self.final_decisions, zero_division=0)
        f1 = f1_score(self.true_labels, self.final_decisions, zero_division=0)

        # AUC
        auc_val = None
        if self.probabilities_test is not None:
            # اگر probabilities_test خروجی predict_proba (ستون کلاس ۱) باشد:
            try:
                auc_val = roc_auc_score(self.true_labels, self.probabilities_test)
            except Exception:
                auc_val = None

        # محاسبه هزینه تصمیم، در صورتی که cost_matrix داشته باشیم
        # فرض بر این است که cost_matrix[i] = {"PP", "PN", "NP", "NN"}
        # و final_decisions[i] = 1 => POS, 0 => NEG
        # true_labels[i] = 1 => نمونه واقعی نکول, 0 => غیرنکول
        total_cost = None
        if self.cost_matrix is not None:
            if len(self.cost_matrix) != len(self.true_labels):
                logging.warning("طول cost_matrix با دادهٔ تست همخوانی ندارد؛ هزینه محاسبه نمی‌شود.")
            else:
                total_cost_calc = 0.0
                for i in range(len(self.true_labels)):
                    y_true = self.true_labels[i]
                    y_pred = self.final_decisions[i]
                    costs = self.cost_matrix[i]
                    # اگر y_pred=1 و y_true=1 => PP
                    # اگر y_pred=1 و y_true=0 => PN
                    # اگر y_pred=0 و y_true=1 => NP
                    # اگر y_pred=0 و y_true=0 => NN
                    if y_pred == 1 and y_true == 1:
                        total_cost_calc += costs["PP"]
                    elif y_pred == 1 and y_true == 0:
                        total_cost_calc += costs["PN"]
                    elif y_pred == 0 and y_true == 1:
                        total_cost_calc += costs["NP"]
                    elif y_pred == 0 and y_true == 0:
                        total_cost_calc += costs["NN"]
                total_cost = total_cost_calc

        # محاسبه GM به صورت sqrt((TP/(TP+FN)) * (TN/(TN+FP)))
        gm = np.sqrt((TP / (TP + FN)) * (TN / (TN + FP))) if (TP + FN) != 0 and (TN + FP) != 0 else 0

        metrics_dict = {"ModelName": "Proposed-3WD", "TN": TN, "FP": FP, "FN": FN, "TP": TP, "BalancedAccuracy": b_acc,
            "Precision": precision, "Recall": recall, "F1": f1, "GM": gm, "AUC": auc_val, "TotalCost": total_cost}
        return metrics_dict


from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from math import sqrt
import logging


class ParsianMethodComparison:
    """
    در این گام، مجموعه‌ای از مدل‌های رقیب (Baseline) را بر روی داده‌های آموزش
    آموزش داده و سپس بر روی داده‌های تست ارزیابی می‌کنیم تا با روش پیشنهادی
    (سه‌طرفه) مقایسه شوند.

    معیارهای ارزیابی:
      - TP, TN, FP, FN از ماتریس سردرگمی
      - BalancedAccuracy = 0.5 * (TP/(TP+FN) + TN/(TN+FP))
      - Precision, Recall, F1 (یا FM با β=1)
      - GM = sqrt( (TP/(TP+FN)) * (TN/(TN+FP)) )
      - AUC (در صورت وجود احتمال پیش‌بینی)
      - TotalCost (در صورت وجود cost_matrix)
    """

    def __init__(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series,
            cost_matrix: list = None, model_comparisons: dict = None):
        """
        پارامترها:
         - x_train, y_train: داده‌های آموزش
         - x_test, y_test: داده‌های تست
         - cost_matrix: در صورت نیاز برای محاسبه هزینه تصمیم (مانند گام ۳)
         - model_comparisons: دیکشنری { ModelName: model_object } برای مدل‌های رقیب
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.cost_matrix = cost_matrix

        if model_comparisons is None:
            from sklearn.naive_bayes import GaussianNB
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.neural_network import MLPClassifier
            from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
                RandomForestClassifier, BaggingClassifier
            from lightgbm import LGBMClassifier
            from xgboost import XGBClassifier
            from sklearn.ensemble import StackingClassifier
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            self.model_comparisons = {"Bayes": GaussianNB(), "KNN": KNeighborsClassifier(),
                "LR": LogisticRegression(max_iter=10_000), "NN": MLPClassifier(max_iter=300),
                "AdaBoost": AdaBoostClassifier(algorithm="SAMME"), "ERT": ExtraTreesClassifier(),
                "GBDT": GradientBoostingClassifier(), "LGBM": LGBMClassifier(verbose=-1),
                "RF": RandomForestClassifier(), "XGB": XGBClassifier(eval_metric='logloss', verbosity=0),
                "Stacking": StackingClassifier(
                    estimators=[('lda', LinearDiscriminantAnalysis()), ('knn', KNeighborsClassifier())],
                    final_estimator=RandomForestClassifier()),
                "Bagging": BaggingClassifier(estimator=ExtraTreesClassifier(n_estimators=100, random_state=42),
                    n_estimators=10, random_state=42)}
        else:
            self.model_comparisons = model_comparisons

        self.comparison_table = None

    def _compute_metrics(self, y_true, y_pred, y_prob=None, cost_matrix=None):
        """
        محاسبه معیارهای عملکرد: TP, TN, FP, FN, BalancedAccuracy, Precision, Recall, F1, GM, AUC و TotalCost.
        """
        cm = confusion_matrix(y_true, y_pred)
        TN, FP, FN, TP = cm.ravel()

        # Balanced Accuracy
        b_acc = 0.5 * ((TP / (TP + FN) if (TP + FN) > 0 else 0) + (TN / (TN + FP) if (TN + FP) > 0 else 0))

        # Precision و Recall
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        # F1 (که همان FM با β=1 است)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # GM
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        gm = sqrt(sensitivity * specificity)

        # AUC
        auc_val = None
        if y_prob is not None:
            from sklearn.metrics import roc_auc_score
            try:
                auc_val = roc_auc_score(y_true, y_prob)
            except Exception:
                auc_val = None

        # محاسبه هزینه تصمیم (TotalCost)
        total_cost = None
        if cost_matrix is not None and len(cost_matrix) == len(y_true):
            tc = 0.0
            for i in range(len(y_true)):
                yi = y_true[i]
                yp = y_pred[i]
                costs = cost_matrix[i]
                if yi == 1 and yp == 1:
                    tc += costs["PP"]
                elif yi == 0 and yp == 1:
                    tc += costs["PN"]
                elif yi == 1 and yp == 0:
                    tc += costs["NP"]
                elif yi == 0 and yp == 0:
                    tc += costs["NN"]
            total_cost = tc

        return {"TP": TP, "TN": TN, "FP": FP, "FN": FN, "BalancedAccuracy": b_acc, "Precision": precision,
            "Recall": recall, "F1": f1, "GM": gm, "AUC": auc_val, "TotalCost": total_cost}

    def run_comparison(self):
        """
        برای هر مدل در مدل‌های رقیب:
          - آموزش روی x_train, y_train
          - پیش‌بینی روی x_test (y_pred و در صورت امکان y_prob)
          - محاسبه معیارها
          - ثبت نتایج در یک DataFrame
        خروجی: DataFrame شامل نتایج مقایسه
        """
        logging.info("🔵 شروع مقایسه با سایر روش‌ها (گام ۹) ...")
        results_list = []
        for model_name, model_obj in self.model_comparisons.items():
            model_obj.fit(self.x_train, self.y_train)

            y_pred = model_obj.predict(self.x_test)

            y_prob = None
            try:
                prob_mat = model_obj.predict_proba(self.x_test)
                y_prob = prob_mat[:, 1]
            except Exception:
                y_prob = None

            metrics = self._compute_metrics(y_true=self.y_test.values, y_pred=y_pred, y_prob=y_prob,
                cost_matrix=self.cost_matrix)
            metrics["ModelName"] = model_name
            results_list.append(metrics)

        df_results = pd.DataFrame(results_list)
        df_results.sort_values(by="BalancedAccuracy", ascending=False, inplace=True)
        self.comparison_table = df_results.reset_index(drop=True)
        logging.info("✅ مقایسه مدل‌های رقیب پایان یافت.")
        return self.comparison_table

    def add_proposed_method_results(self, proposed_method_metrics: dict):
        """
        افزودن نتایج روش پیشنهادی به جدول مقایسه.
        proposed_method_metrics باید دیکشنری با کلیدهای:
        ModelName, TP, TN, FP, FN, BalancedAccuracy, Precision, Recall, F1, GM, AUC, TotalCost
        باشد.
        """
        self.comparison_table = pd.concat([self.comparison_table, pd.DataFrame([proposed_method_metrics])],
            ignore_index=True)
        self.comparison_table.sort_values(by="BalancedAccuracy", ascending=False, inplace=True)
        self.comparison_table.reset_index(drop=True, inplace=True)
        logging.info("🔵 نتایج روش پیشنهادی هم به جدول مقایسه اضافه شد.")

    def show_final_comparison(self):
        """
        نمایش جدول نهایی مقایسه در لاگ و بازگرداندن آن.
        تنظیمات پانداز به گونه‌ای تغییر می‌کند که همه ستون‌ها بدون ellipsis نمایش داده شوند.
        """
        logging.info("🔸 جدول مقایسه مدل‌ها:")
        logging.warning("\n" + str(self.comparison_table))
        return self.comparison_table


###########################################
# تست کل فرآیند (در صورت اجرای مستقیم این فایل)
###########################################
if __name__ == "__main__":
    os.environ["LOKY_MAX_CPU_COUNT"] = "8"

    logging.basicConfig(level=logging.INFO)

    # ساخت آبجکت مخزن داده (LoanRepository)
    repo = LoanRepository()

    # ایجاد مدیر پیش‌پردازش (ParsianPreprocessingManager)
    prep_manager = ParsianPreprocessingManager(repository=repo, limit_records=5000, label_column="status",
        imputation_strategy="mean", need_2_remove_highly_correlated_features=False, correlation_threshold=0.9,
        do_balance=True, test_size=0.2, random_state=42)

    x_train, y_train, x_test, y_test, original_df = prep_manager.step1_process_data()
    if x_train is None:
        logging.error("گام اول ناموفق بود.")
        exit(1)

    # 2) اجرای گام دوم: آموزش مدل و محاسبه احتمال نکول
    default_model = ParsianDefaultProbabilityModel(model_type="lightgbm", n_estimators=100, learning_rate=0.05,
        random_state=42)
    default_model.fit_model(x_train, y_train)
    probabilities_test = default_model.predict_default_probability(x_test)

    visualizer = Plot()
    visualizer.visualize_distribution(probabilities_test)

    logging.info(f"احتمال نکول برای اولین 5 نمونه: {probabilities_test[:5]}")
    logging.info("گام دوم (برآورد احتمال نکول) با موفقیت انجام شد.")

    # 3) گام سوم: محاسبه ماتریس زیان
    # فرض می‌کنیم x_test دارای ستون‌های approval_amount و interest_amount است.
    cost_calc = ParsianCostMatrix(df_test=x_test, approval_col="approval_amount", interest_col="interest_amount")
    cost_calc.compute_costs()
    all_costs = cost_calc.get_all_costs()

    # 4) گام چهارم: بهینه‌سازی چندهدفه آستانه‌ها با NSGA-II
    from numpy import array

    threshold_nsgaii = ParsianThresholdNSGA2(probabilities_test=probabilities_test, cost_matrix=all_costs,
        true_labels=y_test.values,  # یا array(y_test)
        pop_size=50, n_gen=100, step_bnd=False)
    threshold_nsgaii.optimize()

    solutions, objectives = threshold_nsgaii.get_pareto_front()
    logging.info("🔹 راه‌حل‌های پارتو (alpha,beta) و مقدار اهداف (cost,boundary):")
    for i, sol in enumerate(solutions):
        alpha, beta = sol
        cost_val, boundary_val = objectives[i]
        logging.info(f"  alpha={alpha:.3f}, beta={beta:.3f} => cost={cost_val:.2f}, boundary={boundary_val:.3f}")

    # انتخاب راه‌حل نهایی از میان راه‌حل‌های پارتو بر اساس کمترین مقدار objective دوم (boundary_size)
    final_solution, final_objectives = threshold_nsgaii.get_final_solution()
    best_alpha, best_beta = final_solution[0], final_solution[1]
    logging.warning(
        f"🔹 the best is: alpha={best_alpha:.3f}, beta={best_beta:.3f} => cost={final_objectives[0]:.2f}, boundary={final_objectives[1]:.3f}")
    logging.info("گام چهارم (NSGA-II چندهدفه) با موفقیت انجام شد.")

    threeway = ParsianThreeWayDecision(probabilities_test, best_alpha, best_beta)
    decisions_final = threeway.apply_three_way_decision()
    logging.info(f"Decision counts: POS: {threeway.get_decision_counts().get(1, 0)} samples,"
                 f" NEG: {threeway.get_decision_counts().get(0, 0)} samples,"
                 f" BND: {threeway.get_decision_counts().get(-1, 0)} samples")

    # 6) گام ششم: تصمیم‌گیری روی BNDها
    bnd_resolver = ParsianBNDResolver(x_train_all=x_train, y_train_all=y_train, model_type="stacking")
    bnd_resolver.fit_bnd_model()

    # اعمال مدل استکینگ روی نمونه‌های مرزی
    decisions_updated = bnd_resolver.resolve_bnd_samples(x_test, decisions_final)

    logging.info("🔹 برچسب‌های نهایی پس از گام ششم:")
    logging.info(
        f" count POS={np.sum(decisions_updated == 1)}, NEG={np.sum(decisions_updated == 0)}, BND={np.sum(decisions_updated == -1)}")

    # 7) گام هفتم: Evaluation نهایی
    final_eval = ParsianFinalEvaluator(true_labels=y_test.values, final_decisions=decisions_updated,
        probabilities_test=probabilities_test,  # اگر AUC بخواهیم
        cost_matrix=all_costs  # اگر هزینه بخواهیم
    )
    results = final_eval.evaluate_metrics()
    logging.info("🔹 نتایج نهایی مدل:")
    for k, v in results.items():
        logging.info(f"  {k}: {v}")

    comparator = ParsianMethodComparison(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        cost_matrix=all_costs,  # اگر می‌خواهید هزینه هم محاسبه شود
        model_comparisons=None  # اگر None بگذارید، چند مدل پایه به صورت پیش‌فرض دارد
    )
    comparison_df = comparator.run_comparison()
    logging.error("\nنتایج مدل‌های رقیب:\n" + str(comparison_df))

    comparator.add_proposed_method_results(proposed_method_metrics=results)

    final_comparison = comparator.show_final_comparison()
    logging.info("🔹 گام نهم (مقایسه با سایر روش‌ها) با موفقیت انجام شد.")
