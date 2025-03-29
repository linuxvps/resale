import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from colorlog import ColoredFormatter
# ------------------------------------------------------------
# بخش مربوط به SQLAlchemy برای اتصال به دیتابیس و تعریف انتیتی
# ------------------------------------------------------------
from sqlalchemy import create_engine, Column, Integer, BigInteger, String, Date, DateTime, Numeric, Float
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
protected_columns = ['approval_amount', 'interest_amount']

formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


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

    def fetch_loans(self, limit=10000):
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

    def __init__(
        self,
        model_type="lightgbm",
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        **kwargs
    ):
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
            self.model = LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                **self.kwargs
            )
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

    def __init__(
            self,
            repository,
            limit_records=10000,
            label_column="status",
            imputation_strategy="mean",
            need_2_remove_highly_correlated_features=False,
            correlation_threshold=0.9,
            do_balance=True,
            test_size=0.2,
            random_state=42
    ):
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
            df = self.preprocessor.remove_highly_correlated_features(
                df,
                threshold=self.correlation_threshold,
                class_column=self.label_column
            )

        # ایمپیوت
        df_imputed = pd.DataFrame(self.preprocessor.imputer.fit_transform(df), columns=df.columns)

        # تفکیک X,y
        X = df_imputed.drop(columns=[self.label_column])
        y = df_imputed[self.label_column]

        # تقسیم آموزش/آزمون
        x_train, x_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

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

            self.cost_matrix.append({
                "PP": cost_pp,
                "NN": cost_nn,
                "PN": cost_pn,
                "NP": cost_np
                # "BP": cost_bp,
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
# تست کل فرآیند (در صورت اجرای مستقیم این فایل)
###########################################
if __name__ == "__main__":
    os.environ["LOKY_MAX_CPU_COUNT"] = "8"

    logging.basicConfig(level=logging.INFO)

    # ساخت آبجکت مخزن داده (LoanRepository)
    repo = LoanRepository()

    # ایجاد مدیر پیش‌پردازش (ParsianPreprocessingManager)
    prep_manager = ParsianPreprocessingManager(
        repository=repo,
        limit_records=5000,
        label_column="status",
        imputation_strategy="mean",
        need_2_remove_highly_correlated_features=False,
        correlation_threshold=0.9,
        do_balance=True,
        test_size=0.2,
        random_state=42
    )

    x_train, y_train, x_test, y_test, original_df = prep_manager.step1_process_data()
    if x_train is None:
        logging.error("گام اول ناموفق بود.")
        exit(1)

    # 2) اجرای گام دوم: آموزش مدل و محاسبه احتمال نکول
    default_model = ParsianDefaultProbabilityModel(
        model_type="lightgbm",
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    )
    default_model.fit_model(x_train, y_train)
    probabilities_test = default_model.predict_default_probability(x_test)

    logging.info(f"احتمال نکول برای اولین 5 نمونه: {probabilities_test[:5]}")
    logging.info("گام دوم (برآورد احتمال نکول) با موفقیت انجام شد.")

    # 3) گام سوم: محاسبه ماتریس زیان
    # فرض می‌کنیم x_test دارای ستون‌های approval_amount و interest_amount است.
    cost_calc = ParsianCostMatrix(df_test=x_test, approval_col="approval_amount", interest_col="interest_amount")
    cost_calc.compute_costs()
    all_costs = cost_calc.get_all_costs()

    logging.info(f"نمونه‌ای از cost_matrix[0]: {all_costs[0]}")
    logging.info("گام سوم (محاسبه ماتریس زیان) با موفقیت انجام شد.")
