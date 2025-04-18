import logging
import os

import networkx as nx
import pandas as pd
from colorlog import ColoredFormatter
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
# ------------------------------------------------------------
# بخش مربوط به SQLAlchemy برای اتصال به دیتابیس و تعریف انتیتی
# ------------------------------------------------------------
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
protected_columns = ['LOAN_AMOUNT', 'CURRENT_LOAN_RATES']

formatter = ColoredFormatter("%(log_color)s%(asctime)s - %(levelname)s - %(message)s", datefmt=None, reset=True,
                             log_colors={'DEBUG': 'cyan', 'INFO': 'white', 'WARNING': 'yellow', 'ERROR': 'red',
                                         'CRITICAL': 'bold_red', })

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
import seaborn as sns


class Plot:
    """
    یک کلاس عمومی برای بصری‌سازی توزیع احتمال‌ها.
    """

    def __init__(self) -> None:
        pass

    def plot_default_prob_hist(self,           # ← اضافه شدن self
                               probs,
                               u, v,
                               bins=100,
                               figsize=(12, 6),
                               log_y=True,
                               title='Distribution of Default Probabilities with Thresholds (u, v)'):
        """
        رسم هیستوگرام احتمال نکول به‌همراه خطوط آستانهٔ u و v
        """
        plt.figure(figsize=figsize)

        n, bins_edges, _ = plt.hist(probs,
                                    bins=bins,
                                    color='skyblue',
                                    edgecolor='black',
                                    alpha=0.7)

        mean_val = np.mean(probs)
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean = {mean_val:.2f}')
        plt.axvline(u,        color='green',  linewidth=2, label=f'u (POS) = {u:.3f}')
        plt.axvline(v,        color='orange', linewidth=2, label=f'v (NEG) = {v:.3f}')

        if log_y:
            plt.yscale('log')

        plt.title(title)
        plt.xlabel('Probability')
        plt.ylabel('Frequency' + (' (log scale)' if log_y else ''))
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def draw_preprocessing_flowchart(self, output_path="flowchart_standardize_select.png"):
        G = nx.DiGraph()

        G.add_node("Extract Data\n(استخراج داده‌ها)")
        G.add_node("Data Cleaning\n(پاکسازی داده‌ها)")
        G.add_node("Convert to Standard Format\n(تبدیل داده‌ها به قالب استاندارد)")
        G.add_node("Handle Missing & Invalid Values\n(حذف رکوردهای ناقص/ناصحیح)")
        G.add_node("Standardize Numeric & Date Columns\n(استانداردسازی داده‌های عددی و تاریخی)")
        G.add_node("Correlation Analysis\n(تحلیل همبستگی)")
        G.add_node("Remove Redundant Features\n(حذف ویژگی‌های تکراری)")
        G.add_node("Feature Selection\n(انتخاب ویژگی‌های کلیدی)")
        G.add_node("Preprocessed Data Ready for Modeling\n(داده‌های آماده مدل‌سازی)")

        G.add_edges_from([
            ("Extract Data\n(استخراج داده‌ها)", "Data Cleaning\n(پاکسازی داده‌ها)"),
            ("Data Cleaning\n(پاکسازی داده‌ها)", "Convert to Standard Format\n(تبدیل داده‌ها به قالب استاندارد)"),
            ("Convert to Standard Format\n(تبدیل داده‌ها به قالب استاندارد)",
             "Handle Missing & Invalid Values\n(حذف رکوردهای ناقص/ناصحیح)"),
            ("Handle Missing & Invalid Values\n(حذف رکوردهای ناقص/ناصحیح)",
             "Standardize Numeric & Date Columns\n(استانداردسازی داده‌های عددی و تاریخی)"),
            ("Standardize Numeric & Date Columns\n(استانداردسازی داده‌های عددی و تاریخی)",
             "Correlation Analysis\n(تحلیل همبستگی)"),
            ("Correlation Analysis\n(تحلیل همبستگی)", "Remove Redundant Features\n(حذف ویژگی‌های تکراری)"),
            ("Remove Redundant Features\n(حذف ویژگی‌های تکراری)", "Feature Selection\n(انتخاب ویژگی‌های کلیدی)"),
            ("Feature Selection\n(انتخاب ویژگی‌های کلیدی)",
             "Preprocessed Data Ready for Modeling\n(داده‌های آماده مدل‌سازی)")
        ])

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3500, font_size=10,
                font_weight='bold', arrows=True, arrowstyle='->', arrowsize=20)
        plt.title("Flowchart: Standardization & Feature Selection Process", fontsize=14, fontweight='bold')

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot1(self, probabilities: np.ndarray, bins: int = 100, figsize: Tuple[int, int] = (10, 6),
              xlim: Tuple[float, float] = None) -> None:
        """
        نمودار هیستوگرام توزیع احتمال‌ها را بر اساس آرایه ورودی رسم می‌کند.
        با افزودن جزئیات مانند خط میانگین، برچسب‌های دقیق محور و تنظیمات grid.

        :param probabilities: آرایه numpy شامل احتمال‌ها.
        :param bins: تعداد بخش‌های هیستوگرام (پیش‌فرض 100).
        :param figsize: اندازه شکل نمودار (پیش‌فرض (10, 6)).
        :param xlim: محدوده محور افقی به صورت (min, max). اگر None باشد، به صورت خودکار تنظیم می‌شود.
        """
        plt.figure(figsize=figsize)

        # رسم هیستوگرام
        n, bins, patches = plt.hist(probabilities, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')

        # محاسبه و نمایش خط میانگین
        mean_val = np.mean(probabilities)
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {mean_val:.2f}')

        # تنظیم محدوده محور افقی در صورت نیاز
        if xlim is not None:
            plt.xlim(xlim)

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

    def plot_label_count(self, label_counts: pd.Series) -> None:
        plt.figure(figsize=(10, 6))

        # تبدیل ایندکس‌ها به عددی
        label_counts.index = label_counts.index.astype(int)

        # تبدیل سری به DataFrame برای استفاده از hue
        label_df = pd.DataFrame({'Labels': label_counts.index, 'Frequency': label_counts.values})

        sns.barplot(x='Labels', y='Frequency', data=label_df, hue='Labels', dodge=False, palette=['#4CAF50', '#FF6F61'],
                    legend=False)

        plt.title('Label Distribution After Conversion', fontsize=18)
        plt.xlabel('Labels (0: Non-Default, 1: Default)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.xticks(ticks=[0, 1], labels=['Non-Default (0)', 'Default (1)'])
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_with_thresholds(self,
                             probabilities: np.ndarray,
                             u: float,            # ← قبلاً alpha بود
                             v: float,            # ← قبلاً beta بود
                             bins: int = 100,
                             figsize: Tuple[int, int] = (12, 6),
                             xlim: Tuple[float, float] = None) -> None:
        """
        رسم هیستوگرام احتمال نکول به‌همراه خطوط u و v (آستانه‌های بهینهٔ جهانی).
        """
        plt.figure(figsize=figsize)

        # ۱) هیستوگرام احتمال‌ها
        n, bins_array, patches = plt.hist(probabilities,
                                          bins=bins,
                                          edgecolor='black',
                                          alpha=0.7,
                                          color='skyblue')

        # ۲) خط میانگین
        mean_val = np.mean(probabilities)
        plt.axvline(mean_val,
                    color='red',
                    linestyle='dashed',
                    linewidth=2,
                    label=f'Mean = {mean_val:.2f}')

        # ۳) خطوط u و v  (به‌ترتیب تصمیمِ POS و NEGِ سراسری)
        plt.axvline(u,
                    color='green',
                    linestyle='-',
                    linewidth=3,
                    label=f'u (POS) = {u:.3f}')
        plt.axvline(v,
                    color='orange',
                    linestyle='-',
                    linewidth=3,
                    label=f'v (NEG) = {v:.3f}')

        # جزئیات نمودار
        plt.title("Distribution of Default Probabilities with Global Thresholds (u, v)",
                  fontsize=16)
        plt.xlabel("Probability", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)

        if xlim:
            plt.xlim(xlim)

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, model, feature_names, top_n=20):
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False).head(top_n)

        plt.figure(figsize=(14, 12))  # افزایش ارتفاع نمودار

        sns.barplot(
            x='Importance',
            y='Feature',
            data=feature_importance,
            palette='viridis',
            hue='Feature',
            dodge=False
        )

        plt.title('Feature Importance (Top {})'.format(top_n), fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.yticks(fontsize=10)  # کاهش اندازه فونت ویژگی‌ها
        plt.legend([], [], frameon=False)  # حذف legend اضافی
        plt.tight_layout()  # تنظیم خودکار حاشیه‌ها
        plt.show()


    def plot_pca(self, X: pd.DataFrame, n_components: int = 2):
        """
        اجرای PCA و نمایش درصد واریانس توضیح داده شده توسط مؤلفه‌ها
        :param X: داده‌ها (به شکل DataFrame)
        :param n_components: تعداد مؤلفه‌های اصلی برای نمایش
        """
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()

        plt.figure(figsize=(10, 6))
        plt.bar(range(1, n_components + 1), explained_variance * 100, alpha=0.5, align='center',
                label='Individual Explained Variance')
        plt.step(range(1, n_components + 1), cumulative_variance * 100, where='mid',
                 label='Cumulative Explained Variance')
        plt.xlabel('Principal Components', fontsize=14)
        plt.ylabel('Percentage of Variance Explained', fontsize=14)
        plt.title('Explained Variance by Principal Components', fontsize=16)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        print(f"مقدار واریانس توضیح داده شده توسط مؤلفه‌ها: {explained_variance}")
        print(f"واریانس تجمعی (Cumulative Variance): {cumulative_variance[-1]}")

    def explained_variance(self, x_train, n_components=10):
        pca = PCA(n_components=n_components)
        pca.fit(x_train)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # نمایش نسبت واریانس و واریانس تجمعی
        print(f"Explained Variance Ratio: {explained_variance_ratio}")
        print(f"Cumulative Variance: {cumulative_variance}")

        # رسم نمودار
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, n_components + 1), explained_variance_ratio * 100, alpha=0.6, color='skyblue',
                label='Individual Explained Variance')
        plt.plot(range(1, n_components + 1), cumulative_variance * 100, color='blue', marker='o',
                 label='Cumulative Explained Variance')
        plt.title('Explained Variance by Principal Components')
        plt.xlabel('Principal Components')
        plt.ylabel('Percentage of Variance Explained')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_pca_2d(self, x_train):
        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(x_train)
        plt.figure(figsize=(10, 6))
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c='blue', alpha=0.5)
        plt.title('2D PCA Plot')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.show()

    def plot_pca_3d(self, x_train):
        pca = PCA(n_components=3)
        x_pca = pca.fit_transform(x_train)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c='blue', alpha=0.5)
        ax.set_title('3D PCA Plot')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.show()

    def plot_tsne(self, x_train):
        tsne = TSNE(n_components=2, random_state=42)
        x_tsne = tsne.fit_transform(x_train)

        plt.figure(figsize=(10, 6))
        plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c='blue', alpha=0.5)
        plt.title('t-SNE Plot')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True)
        plt.show()

    def plot_pareto_front(self, front_costs):
        """
        نمایش نمودار پارتو از خروجی NSGA-II (cost vs boundary size)
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(front_costs[:, 0], front_costs[:, 1], c='red', label='Pareto Front')
        plt.xlabel('Total Cost', fontsize=14)
        plt.ylabel('Boundary Size', fontsize=14)
        plt.title('Pareto Front - NSGA-II', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        رسم ماتریس سردرگمی مدل نهایی (Confusion Matrix)
        """
        cm = confusion_matrix(y_true, y_pred)
        labels = ['NEG', 'POS']
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix - Final Model", fontsize=14)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.tight_layout()
        plt.show()


from sqlalchemy import Column, BigInteger, Integer, Numeric, DateTime, Date, String, CHAR, Float
from datetime import datetime


class ParsianLoan(Base):
    __tablename__ = "parsian_loan_2"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    advance_pay = Column(Numeric(28, 8), nullable=True)
    advance_pay_to_remain_non_cash = Column(Numeric(28, 8), nullable=True)
    advance_pay_to_total_cash = Column(Numeric(28, 8), nullable=True)
    approval_amount = Column(Numeric(28, 8), nullable=True)
    bank_share_cash_amount = Column(Numeric(28, 8), nullable=True)
    bank_share_non_cash_amount = Column(Numeric(28, 8), nullable=True)
    branch_code = Column(Integer, nullable=False)
    branchname = Column(String(100, collation='utf8mb4_unicode_ci'), nullable=True)
    charge = Column(Numeric(28, 8), nullable=True)
    loan_file_numberr = Column(BigInteger, nullable=True)
    client_id = Column(Integer, nullable=True)
    commission_amount_remain = Column(Numeric(28, 8), nullable=True)
    contract = Column(String(255, collation='utf8mb4_unicode_ci'), nullable=True)
    create_date = Column(Date, nullable=True)
    customer_obligation_amount = Column(Numeric(28, 8), nullable=True)
    customer_share_cash_amount = Column(Numeric(28, 8), nullable=True)
    customer_share_non_cash_amount = Column(Numeric(28, 8), nullable=True)
    discount = Column(Numeric(28, 8), nullable=True)
    due_date = Column(Date, nullable=True)
    finalized_loan_amount = Column(Numeric(28, 8), nullable=True)
    first_over_due = Column(Date, nullable=True)
    first_passed = Column(Date, nullable=True)
    first_payment_date_in_du = Column(Date, nullable=True)
    frequency = Column(Integer, nullable=True)
    inc_commission_amount = Column(Numeric(28, 8), nullable=True)
    insert_sysdate = Column(DateTime(6), nullable=False, default=datetime.utcnow)
    installment_number_remain = Column(Integer, nullable=True)
    interest_amount = Column(Numeric(28, 8), nullable=True)
    interest_rate = Column(Numeric(19, 2), nullable=True)
    interest_sum = Column(Numeric(28, 8), nullable=True)
    is_installment = Column(CHAR, nullable=True)
    loan_duration_day = Column(Integer, nullable=True)
    loan_file_number = Column(BigInteger, nullable=True)
    long_title = Column(String(255, collation='utf8mb4_unicode_ci'), nullable=True)
    obligation_penalty = Column(Numeric(28, 8), nullable=True)
    passed_date = Column(Date, nullable=True)
    penalty = Column(Numeric(28, 8), nullable=True)
    penalty_interest = Column(Numeric(28, 8), nullable=True)
    principal_sum = Column(Numeric(28, 8), nullable=True)
    receivable_installment_number = Column(Integer, nullable=True)
    sit_distribute_phases = Column(Integer, nullable=True)
    sit_duration = Column(Integer, nullable=True)
    sit_duration_day = Column(Integer, nullable=True)
    sit_fast_receive_percent = Column(Float, nullable=True)
    sit_flag = Column(CHAR, nullable=True)
    status = Column(String(255, collation='utf8mb4_unicode_ci'), nullable=True)
    title = Column(String(255, collation='utf8mb4_unicode_ci'), nullable=True)
    to_due_date = Column(Numeric(28, 8), nullable=True)
    to_end_of_month = Column(Numeric(28, 8), nullable=True)
    total_payment_up_to_now = Column(Numeric(28, 8), nullable=True)
    total_repayment_up_to_now = Column(Numeric(28, 8), nullable=True)

    def __repr__(self):
        return f"<ParsianLoan(id={self.id})>"


class LoanDetail(Base):
    __tablename__ = "MY_TABLE"

    ID = Column(BigInteger, primary_key=True, autoincrement=False)
    LOAN_FILE_NUMBER = Column(BigInteger, nullable=True)
    LOAN_AMOUNT = Column(Numeric(65, 2), nullable=True)
    TOTAL_DEBT_IN_TOMAN = Column(Numeric(65, 2), nullable=True)
    CURRENT_LOAN_RATES = Column(Numeric(65, 2), nullable=True)
    LOAN_PURPOSE = Column(String(255), nullable=True)
    CONTRACT_DUE_DATE = Column(Date, nullable=True)
    INSTALLMENT_LOAN_AWARD_DATE = Column(Date, nullable=True)
    FIRST_PAYMENT_DATE_IN_DU = Column(Date, nullable=True)
    GRANT_DATE = Column(Date, nullable=True)
    APPLICATION_TYPE = Column(CHAR, nullable=True)
    LOAN_STATUS = Column(String(255), nullable=True)
    TOTAL_INSTALLMENT_AMOUNT = Column(Numeric(65, 2), nullable=True)
    NUM_OF_INSTALLMENTS = Column(BigInteger, nullable=True)
    FIRST_INSTALLMENT_DUE = Column(Date, nullable=True)
    LAST_INSTALLMENT_DUE = Column(Date, nullable=True)
    DEFAULT_COUNT = Column(BigInteger, nullable=True)
    COMPANY_TYPE = Column(BigInteger, nullable=True)
    POSTAL_CODE = Column(String(20), nullable=True)
    CITY_CODE = Column(String(20), nullable=True)
    REGION = Column(String(255), nullable=True)
    PROVINCE = Column(String(255), nullable=True)
    APPROXIMATE_INCOME_IN_TOMAN = Column(Numeric(65, 2), nullable=True)
    ANNUAL_TURNOVER_IN_TOMAN = Column(Numeric(65, 2), nullable=True)


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

    def fetch_loans_in_chunks(self, excluded_columns,chunk_size=100000):
        total_rows = self.session.query(LoanDetail).count()
        offset = 0
        dataframes = []
        while offset < total_rows:
            loans_chunk = (self.session.query(LoanDetail)
                           .order_by(LoanDetail.LOAN_FILE_NUMBER.desc())
                           .offset(offset)
                           .limit(chunk_size)
                           .all())
            if not loans_chunk:
                break
            # ستون‌هایی که نیاز به دریافت نداریم
            all_columns = list(LoanDetail.__table__.columns.keys())
            selected_columns = [col for col in all_columns if col not in excluded_columns]
            data = {col: [getattr(loan, col) for loan in loans_chunk] for col in selected_columns}
            df_chunk = pd.DataFrame(data)
            dataframes.append(df_chunk)
            offset += chunk_size
            logging.info(f"دریافت {min(offset, total_rows)} از {total_rows} رکورد")
        if dataframes:
            return pd.concat(dataframes, ignore_index=True)
        else:
            return pd.DataFrame()



    def fetch_loans(self,excluded_columns, limit=10_000):
        """
        واکشی حداکثر `limit` رکورد از جدول parsian_loan.
        داده‌ها در قالب یک DataFrame برگردانده می‌شوند.
        """
        # دریافت لیست تمام ستون‌های موجود در جدول
        all_columns = [column.name for column in LoanDetail.__table__.columns]
        # انتخاب ستون‌هایی که در لیست excluded وجود ندارند
        selected_columns = [col for col in all_columns if col not in excluded_columns]

        # اجرای کوئری با انتخاب فقط ستون‌های مورد نظر

        loans = (self.session.query(*[getattr(LoanDetail, col) for col in selected_columns]).order_by(
            LoanDetail.LOAN_FILE_NUMBER.desc()).limit(limit).all())

        if not loans:
            logging.warning("هیچ داده‌ای از پایگاه داده دریافت نشد.")
            return pd.DataFrame()

        # تبدیل داده‌ها به DataFrame
        data = {col: [getattr(loan, col) for loan in loans] for col in selected_columns}
        df = pd.DataFrame(data)
        logging.info(f"✅ {len(df)} رکورد از دیتابیس دریافت شد (LoanDetail).")
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
        Plot().plot_feature_importance(default_model.model, x_train.columns)

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
from sklearn.preprocessing import StandardScaler


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
        self.scaler = StandardScaler()  # برای استانداردسازی ثابت

    def standardize_numeric_columns(self, df: pd.DataFrame, exclude_cols: list = None,
                                    fit: bool = True) -> pd.DataFrame:
        """
        ستون‌های عددی را به کمک Z‑score استاندارد می‌کند.
        :param df: دیتافریم ورودی
        :param exclude_cols: لیستی از ستون‌هایی که نباید استاندارد شوند (مثل برچسب)
        :param fit: اگر True باشد، scaler را با داده‌های df می‌فیت می‌کند؛
                    در غیر این صورت فقط transform انجام می‌دهد.
        :return: دیتافریم استانداردشده
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if exclude_cols:
            numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        return df


    def convert_labels(self, df, label_column="status"):
        logging.info(f"[LoanPreprocessor] تبدیل برچسب: {label_column}")
        if label_column not in df.columns:
            raise ValueError(f"ستون {label_column} در داده وجود ندارد.")

        logger.warning(df[label_column].value_counts())

        # فرض بر این است که مقادیر {"مشكوك الوصول", "معوق", "سررسيد گذشته"} => 1
        default_statuses = {"مشكوك الوصول", "معوق", "سررسيد گذشته", "سررسيد" , "باطل شده" , "درخواست رد شد"}
        df[label_column] = df[label_column].apply(lambda x: 1 if x in default_statuses else 0)
        # لاگ گرفتن از توزیع داده‌ها
        label_counts = df[label_column].value_counts()
        logging.warning(label_counts.to_string())

        Plot().plot_label_count(label_counts)
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

    def summary_stats_for_df(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        این تابع برای هر ستون موجود در DataFrame، آماری شامل:
        تعداد یکتا، تعداد داده‌های گمشده (NaN)، مینیموم، ماکسیموم، دامنه (ماکسیموم - مینیموم)،
        میانگین، واریانس و انحراف معیار را محاسبه می‌کند.

        نکات:
        - برای ستون‌های عددی، تمامی آماره‌ها به صورت دقیق محاسبه می‌شوند.
        - برای ستون‌های تاریخ (datetime)، مینیموم، ماکسیموم، دامنه و میانگین (به عنوان تاریخ میانگین) محاسبه می‌شود.
          اما واریانس و انحراف معیار به دلیل عدم تناسب معنایی با تاریخ، به عنوان None برگردانده می‌شوند.
        - برای سایر نوع داده‌ها، تنها مینیموم و ماکسیموم (و در صورت امکان دامنه) محاسبه شده و آماره‌های عددی برابر None خواهند بود.
        """
        stats_rows = []

        for col in df.columns:
            # محاسبه تعداد یکتا (بدون احتساب NaN)
            unique_count = df[col].nunique(dropna=True)
            # محاسبه تعداد مقادیر گمشده
            missing_count = df[col].isna().sum()

            # بررسی نوع داده ستون برای پردازش مناسب
            try:
                # اگر ستون عددی باشد
                if np.issubdtype(df[col].dtype, np.number):
                    col_min = df[col].min(skipna=True)
                    col_max = df[col].max(skipna=True)
                    col_range = col_max - col_min if (col_min is not None and col_max is not None) else None
                    col_mean = df[col].mean(skipna=True)
                    col_var = df[col].var(skipna=True)
                    col_std = df[col].std(skipna=True)

                # اگر ستون از نوع تاریخ (datetime) باشد
                elif np.issubdtype(df[col].dtype, np.datetime64):
                    col_min = df[col].min(skipna=True)
                    col_max = df[col].max(skipna=True)
                    # دامنه به عنوان اختلاف زمان بین بیشینه و کمینه محاسبه می‌شود
                    col_range = col_max - col_min if (col_min is not None and col_max is not None) else None
                    # میانگین تاریخ محاسبه می‌شود؛ واریانس و انحراف معیار به عنوان None تعیین می‌شوند
                    col_mean = df[col].mean(skipna=True)
                    col_var = None
                    col_std = None

                # برای ستون‌های غیر عددی و غیر تاریخ
                else:
                    col_min = df[col].min()
                    col_max = df[col].max()
                    # بررسی اینکه آیا مقادیر مینیموم و ماکسیموم قابل محاسبه هستند و در صورت امکان دامنه محاسبه می‌شود
                    try:
                        col_range = col_max - col_min
                    except Exception:
                        col_range = None
                    col_mean, col_var, col_std = None, None, None

            except Exception as e:
                col_min, col_max, col_range, col_mean, col_var, col_std = None, None, None, None, None, None

            stats_rows.append({
                "متغیر": col,
                "تعداد یکتا": unique_count,
                "گمشده": missing_count,
                "مینیموم": col_min,
                "ماکسیموم": col_max,
                "دامنه": col_range,
                "میانگین": col_mean,
                "واریانس": col_var,
                "انحراف معیار": col_std
            })

        stats_df = pd.DataFrame(stats_rows)
        return stats_df


class ParsianPreprocessingManager:
    """
    یک کلاس جامع برای گام اول (Preprocessing) از pseudocodeorg.
    """

    def __init__(self, repository, limit_records=10000, label_column="status", imputation_strategy="mean",
                 need_2_remove_highly_correlated_features=False, correlation_threshold=0.9, do_balance=True,
                 test_size=0.2, random_state=42):
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
        excluded_columns = [LoanDetail.REGION.key,LoanDetail.ID.key,LoanDetail.COMPANY_TYPE.key]
        # اگر تعداد رکوردها بسیار زیاد باشد، از روش chunk استفاده می‌کنیم
        if self.limit_records > 50_000:
            df = self.repository.fetch_loans_in_chunks(excluded_columns,chunk_size=100000)
        else:
            df = self.repository.fetch_loans(excluded_columns,limit=self.limit_records)

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

        summary_stats_for_df = self.preprocessor.summary_stats_for_df(df)
        logging.error(summary_stats_for_df)
        # ایمپیوت
        # حذف ستون‌هایی که همه مقدارشون NaN هست
        all_nan_cols = df.columns[df.isna().all()].tolist()
        if all_nan_cols:
            logging.warning(f"حذف ستون‌های همه NaN: {all_nan_cols}")
            df.drop(columns=all_nan_cols, inplace=True)

        df_imputed = pd.DataFrame(self.preprocessor.imputer.fit_transform(df), columns=df.columns)

        X = df_imputed.drop(columns=[self.label_column])
        y = df_imputed[self.label_column].astype(int)

        X = self.preprocessor.standardize_numeric_columns(X, exclude_cols=[self.label_column], fit=True)

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
# گام سوم: محاسبه ماتریس زیان (loss Computation)
###########################################
class ParsianLossMatrix:
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
        λ_PP = λ_NN = 0
        λ_PN = interest          (زیان پذیرش اشتباهِ غیرنکول)
        λ_NP = principal + interest   (زیان ردِ اشتباهِ نکول)
        λ_BP = u·λ_NP , λ_BN = v·λ_PN  ⇐ داخل NSGA-II محاسبه می‌شود
        """
        if self.approval_col not in self.df_test.columns or self.interest_col not in self.df_test.columns:
            raise ValueError("ستون‌های مالی برای محاسبه زیان موجود نیست.")

        self.cost_matrix.clear()
        for i in range(len(self.df_test)):
            principal = float(self.df_test.loc[i, self.approval_col] or 0.0)
            interest = float(self.df_test.loc[i, self.interest_col] or 0.0)

            self.cost_matrix.append({
                "PP": 0.0,
                "NN": 0.0,
                "PN": interest,
                "NP": principal + interest  # نه ضرب!  جمع طبق مقاله
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

    def _decision_cost_for_sample(self, i: int, u: float, v: float) -> float:
        """
        فرمول کاملِ مقاله برای محاسبهٔ هزینهٔ تصمیم یک رکورد.
        ➊ ابتدا α_i و β_i را با u , v و ضرایب λ محاسبه می‌کنیم.
        ➋ سپس تصمیم (POS / NEG / BND) را پیدا کرده و هزینه را برمی‌گردانیم.
        """
        p_i = self.probabilities_test[i]
        y_i = self.true_labels[i]  # 0 یا 1
        lam = self.cost_matrix[i]  # {'PP','PN','NP','NN'}

        # ---- آستانه‌های نمونه‑ای (α_i , β_i) همان معادلات مقاله ----
        alpha_i = (lam["PN"] - v * lam["PN"]) / ((lam["PN"] - v * lam["PN"]) + (u * lam["NP"]))
        beta_i = (v * lam["PN"]) / ((v * lam["PN"]) + (lam["NP"] - u * lam["NP"]))

        # ---- تصمیم سه‑گانه ----
        if p_i >= alpha_i:  # POS
            return lam["PP"] if y_i == 1 else lam["PN"]
        elif p_i <= beta_i:  # NEG
            return lam["NP"] if y_i == 1 else lam["NN"]
        else:  # BND  (هزینهٔ مرزی طبق مقاله λ_BP , λ_BN)
            bp_cost = 0.25 * lam["NP"]  # λ_BP = 0.25 λ_NP
            bn_cost = 0.25 * lam["PN"]  # λ_BN = 0.25 λ_PN
            return bp_cost if y_i == 1 else bn_cost

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
            اکنون برای هر راه‌حل (u,v) آستانه‌های αᵢ,βᵢ را «برای تک‌تک رکوردها» می‌سازیم.
            λ_BP = u·λ_NP , λ_BN = v·λ_PN  (فرمول 4 مقاله)
            αᵢ , βᵢ طبق فرمول‌های (2) و (3) مقاله محاسبه می‌شود.
            objective1 = مجموع زیانِ سه‌‑راهه
            objective2 = Σ(αᵢ − βᵢ)  (اندازه فضای مرزی)
            constraint : u+v ≤ 1
            """
            n_sol = X.shape[0]
            f1 = np.zeros(n_sol)  # total cost
            f2 = np.zeros(n_sol)  # boundary size
            g = np.zeros((n_sol, 1))  # u+v -1 ≤ 0

            p = self.outer.probabilities_test
            y_true = self.outer.true_labels
            costs = self.outer.cost_matrix

            for k in range(n_sol):
                u, v = X[k]
                tot_cost, bnd_size = 0.0, 0.0

                for i in range(len(p)):
                    lam = costs[i]
                    lam_BP = u * lam["NP"]
                    lam_BN = v * lam["PN"]

                    alpha = (lam["PN"] - lam_BN) / ((lam["PN"] - lam_BN) + (lam_BP - lam["PP"]))
                    beta = (lam_BN - lam["NN"]) / ((lam_BN - lam["NN"]) + (lam["NP"] - lam_BP))

                    # decision + cost
                    if p[i] >= alpha:  # POS
                        tot_cost += lam["PP"] if y_true[i] == 1 else lam["PN"]
                    elif p[i] <= beta:  # NEG
                        tot_cost += lam["NP"] if y_true[i] == 1 else lam["NN"]
                    else:  # BND
                        tot_cost += lam_BP if y_true[i] == 1 else lam_BN
                        bnd_size += (alpha - beta)

                f1[k] = tot_cost
                f2[k] = bnd_size
                g[k, 0] = u + v - 1.0  # همان قیود (6) مقاله

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
        راه‌حل نهایی همان عضوی از جبههٔ پارتو است که:
           •  نسبت BND ≤ 5٪ کل داده
           •  بیشترین Balanced‑Accuracy را دارد
           •  (در صورت تساوی) کمینهٔ Total‑Cost را دارد
        """
        if self.best_solutions is None:
            raise RuntimeError("ابتدا optimize را اجرا کنید.")

        best_idx, best_balacc, best_cost = None, -1.0, np.inf

        for idx, (u, v) in enumerate(self.best_solutions):
            preds = []
            for i in range(len(self.probabilities_test)):
                p_i = self.probabilities_test[i]
                lam = self.cost_matrix[i]
                alpha = (lam["PN"] - v * lam["PN"]) / ((lam["PN"] - v * lam["PN"]) + (u * lam["NP"]))
                beta = (v * lam["PN"]) / ((v * lam["PN"]) + (lam["NP"] - u * lam["NP"]))
                preds.append(1 if p_i >= alpha else (0 if p_i <= beta else -1))

            preds_arr = np.array(preds)
            bnd_ratio = np.mean(preds_arr == -1)
            if bnd_ratio > 0.05:  # شرط مقاله
                continue

            preds_arr[preds_arr == -1] = 0  # برآورد سریع برای BalAcc
            cm = confusion_matrix(self.true_labels, preds_arr)
            TN, FP, FN, TP = cm.ravel()
            balacc = 0.5 * ((TP / (TP + FN + 1e-9)) + (TN / (TN + FP + 1e-9)))

            if (balacc > best_balacc) or (np.isclose(balacc, best_balacc) and self.front_costs[idx, 0] < best_cost):
                best_idx, best_balacc, best_cost = idx, balacc, self.front_costs[idx, 0]

        return self.best_solutions[best_idx], self.front_costs[best_idx]


###########################################
# گام پنجم: تقسیم نمونه‌ها به POS/NEG/BND (Three-Way Decision)
###########################################

class ParsianThreeWayDecision:
    def __init__(self, probabilities_test: np.ndarray, cost_matrix: list,
                 alpha_beta_pair: Tuple[float,float]):
        self.prob = probabilities_test
        self.cost = cost_matrix
        self.u, self.v = alpha_beta_pair   # همان (u*,v*)
        self.decisions = None

    def _alpha_beta_i(self, lam):
        lam_BP = self.u * lam["NP"]
        lam_BN = self.v * lam["PN"]
        α = (lam["PN"] - lam_BN) / ((lam["PN"]-lam_BN) + (lam_BP - lam["PP"]))
        β = (lam_BN - lam["NN"]) / ((lam_BN - lam["NN"]) + (lam["NP"] - lam_BP))
        return α, β

    def apply_three_way_decision(self):
        dec = np.zeros(len(self.prob), dtype=int)
        for i, p_i in enumerate(self.prob):
            α, β = self._alpha_beta_i(self.cost[i])
            if p_i >= α:      dec[i] = 1
            elif p_i <= β:    dec[i] = 0
            else:             dec[i] = -1
        self.decisions = dec
        return dec

    # ---------- شمارش برچسب‌ها --------------
    def get_decision_counts(self):
        """
        برگشت دیکشنری {1:POS , 0:NEG , -1:BND}
        """
        if self.decisions is None:
            self.apply_three_way_decision()
        uniq, cnt = np.unique(self.decisions, return_counts=True)
        return dict(zip(uniq, cnt))

###########################################
# گام ششم: تصمیم‌گیری نهایی روی نمونه‌های BND
#          (مثلاً با استکینگ یا مدل کمکی دیگر)
###########################################
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class ParsianBNDResolver:
    """
    این کلاس نمونه‌هایی را که در گام پنجم در ناحیه BND واقع شده‌اند،
    شناسایی و با یک مدل اضافی، تصمیم قطعی (POS یا NEG) می‌گیرد.

    در این پژوهش، به منظور تعیین تصمیم‌های نهایی برای نمونه‌های مرزی،
    از دو رویکرد یادگیری جمعی به صورت مدل‌های Stacking و Bagging استفاده شده است.
    این انتخاب به پژوهشگر امکان می‌دهد تا در شرایط مختلف داده،
    بهترین روش تکمیلی جهت کاهش عدم قطعیت در تصمیم‌گیری را به کار گیرد.
    """

    def __init__(self, x_train_all: pd.DataFrame, y_train_all: pd.Series, model_type="bagging"):
        """
        پارامترها:
          - x_train_all, y_train_all: داده‌های آموزش اصلی یا داده‌های مرزی
          - model_type: نوع مدلی که می‌خواهیم برای تشخیص نمونه‌های مرزی به‌کار بریم
                       (برای مثال "stacking" یا "bagging")
        """
        self.x_train_all = x_train_all
        self.y_train_all = y_train_all
        self.model_type = model_type
        self.classifier = None

    def fit_bnd_model(self):
        """
        در این بخش، بر اساس نوع انتخاب‌شده (Stacking یا Bagging)، مدل تکمیلی جهت
        تصمیم‌گیری برای نمونه‌های مرزی آموزش داده می‌شود.

        - در صورت انتخاب مدل Stacking، چندین مدل پایه (مانند Random Forest و XGBoost)
          به همراه یک مدل متا (مانند Logistic Regression) به‌صورت یک چارچوب یادگیری
          چندلایه آموزش می‌داده می‌شوند.
        - در صورت انتخاب مدل Bagging، از یک مدل پایه مانند درخت تصمیم به‌عنوان
          الگوریتم یادگیری جمعی استفاده شده و با بهره‌گیری از رویکرد Bagging، توان پیش‌بینی
          در نمونه‌های مرزی ارتقا می‌یابد.

        این رویکرد یادگیری جمعی با هدف بهبود قابلیت تعمیم مدل و کاهش خطاهای ناشی از
        تصمیم‌گیری در نواحی نامطمئن (Boundary) به‌کار گرفته شده است.
        """
        if self.model_type.lower() == "stacking":
            # تنظیم چارچوب مدل Stacking: چندین مدل پایه به همراه یک متا مدل
            base_estimators = [("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
                               ("xgb", XGBClassifier(eval_metric="logloss", random_state=42))]
            meta_estimator = LogisticRegression(max_iter=1000, random_state=42)
            self.classifier = StackingClassifier(estimators=base_estimators, final_estimator=meta_estimator, cv=5,
                                                 n_jobs=-1)
        elif self.model_type.lower() == "bagging":
            base_estimator = DecisionTreeClassifier(
                criterion="gini",
                max_depth=None,  # اجازهٔ رشد کامل
                min_samples_leaf=2,  # جلوگیری از over‑fitting ریز
                class_weight="balanced",  # جبران کلاس اقلیت
                random_state=42
            )

            # ۲) BaggingClassifier با ۲۰۰ درخت، بوت‌استرپ هم روی نمونه و هم روی ویژگی‌ها
            self.classifier = BaggingClassifier(
                estimator=base_estimator,
                n_estimators=200,  # تعداد کیف‌های زیادتر
                max_samples=0.8,  # ۸۰٪ نمونه‌ها در هر کیف
                max_features=0.8,  # ۸۰٪ ویژگی‌ها در هر کیف
                bootstrap=True,
                bootstrap_features=True,  # بوت‌استرپ ویژگی‌ها برای متنوع‌سازی بیشتر
                oob_score=True,  # برآورد خطای خارج‌از-کیف
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
        else:
            raise ValueError("فعلاً فقط مدل‌های 'stacking' و 'bagging' پشتیبانی می‌شوند.")

        logging.info(f"🔵 در حال آموزش مدل BNDResolver ({self.model_type.capitalize()})...")
        self.classifier.fit(self.x_train_all, self.y_train_all)
        logging.info("✅ آموزش مدل BNDResolver کامل شد.")

    def resolve_bnd_samples(self, x_test: pd.DataFrame, decisions_final: np.ndarray):
        """
        این متد به‌منظور به‌روزرسانی تصمیم‌های نهایی برای نمونه‌هایی که در ناحیه مرزی (BND)
        قرار گرفته‌اند، اجرا می‌شود. از مدل آموزش‌دیده جهت تعیین طبقه (POS یا NEG) برای
        نمونه‌های مرزی استفاده می‌شود.

        ورودی‌ها:
          - x_test: داده‌های آزمون کامل
          - decisions_final: برچسب‌های اولیه تصمیم (POS=1, NEG=0, BND=-1)

        خروجی:
          - decisions_updated: آرایه نهایی تصمیم‌ها که نمونه‌های مرزی نیز به یکی از دو کلاس
            قطعی (POS یا NEG) تخصیص یافته‌اند.
        """
        bnd_indices = np.where(decisions_final == -1)[0]
        logging.info(f"🔵 تعداد نمونه‌های BND: {len(bnd_indices)}")

        if len(bnd_indices) == 0:
            logging.info("هیچ نمونه مرزی وجود ندارد. تغییری اعمال نمی‌شود.")
            return decisions_final

        # استخراج نمونه‌های مربوط به منطقه مرزی از مجموعه داده‌های آزمون
        x_test_bnd = x_test.iloc[bnd_indices]
        # پیش‌بینی طبقه نهایی برای این نمونه‌ها با استفاده از مدل تکمیلی
        y_pred_bnd = self.classifier.predict(x_test_bnd)

        # به‌روزرسانی برچسب‌های تصمیم نهایی
        decisions_updated = decisions_final.copy()
        for idx, pred in zip(bnd_indices, y_pred_bnd):
            decisions_updated[idx] = pred  # تعیین نهایی به ازای هر نمونه (0 یا 1)
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
                        "Precision": precision, "Recall": recall, "F1": f1, "GM": gm, "AUC": auc_val,
                        "TotalCost": total_cost}
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
                                      "RF": RandomForestClassifier(),
                                      "XGB": XGBClassifier(eval_metric='logloss', verbosity=0),
                                      "Stacking": StackingClassifier(estimators=[('lda', LinearDiscriminantAnalysis()),
                                                                                 ('knn', KNeighborsClassifier())],
                                                                     final_estimator=RandomForestClassifier()),
                                      "Bagging": BaggingClassifier(
                                          estimator=ExtraTreesClassifier(n_estimators=100, random_state=42),
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
    visualizer = Plot()
    logging.basicConfig(level=logging.INFO)

    # ساخت آبجکت مخزن داده (LoanRepository)
    repo = LoanRepository()

    # ایجاد مدیر پیش‌پردازش (ParsianPreprocessingManager)
    prep_manager = ParsianPreprocessingManager(repository=repo, limit_records=49_000, label_column="LOAN_STATUS",
                                               imputation_strategy="mean",
                                               need_2_remove_highly_correlated_features=False,
                                               correlation_threshold=0.95, do_balance=True, test_size=0.2,
                                               random_state=42)

    x_train, y_train, x_test, y_test, original_df = prep_manager.step1_process_data()
    if x_train is None:
        logging.error("گام اول ناموفق بود.")
        exit(1)

    # visualizer.explained_variance(x_train)
    # visualizer.plot_pca_2d(x_train)
    # visualizer.plot_pca_3d(x_train)
    # visualizer.plot_tsne(x_train)
    # visualizer.draw_preprocessing_flowchart()

    # 2) اجرای گام دوم: آموزش مدل و محاسبه احتمال نکول
    default_model = ParsianDefaultProbabilityModel(model_type="lightgbm", n_estimators=100, learning_rate=0.05,
                                                   random_state=42)
    default_model.fit_model(x_train, y_train)
    probabilities_test = default_model.predict_default_probability(x_test)

    visualizer.plot1(probabilities_test)

    logging.info(f"احتمال نکول برای اولین 5 نمونه: {probabilities_test[:5]}")
    logging.info("گام دوم (برآورد احتمال نکول) با موفقیت انجام شد.")

    # 3) گام سوم: محاسبه ماتریس زیان
    # فرض می‌کنیم x_test دارای ستون‌های approval_amount و interest_amount است.
    cost_calc = ParsianLossMatrix(df_test=x_test, approval_col="LOAN_AMOUNT", interest_col="CURRENT_LOAN_RATES")
    cost_calc.compute_costs()
    all_costs = cost_calc.get_all_costs()

    # 4) گام چهارم: بهینه‌سازی چندهدفه آستانه‌ها با NSGA-II
    from numpy import array

    # ------------------------------------------------------------------
    # 4) گام چهارم: بهینه‌سازی چندهدفه آستانه‌ها با NSGA‑II  (u*, v*)
    # ------------------------------------------------------------------
    threshold_nsgaii = ParsianThresholdNSGA2(
        probabilities_test=probabilities_test,
        cost_matrix=all_costs,
        true_labels=y_test.values,  # یا np.array(y_test)
        pop_size=50,
        n_gen=100,
        step_bnd=False
    )
    threshold_nsgaii.optimize()

    solutions, objectives = threshold_nsgaii.get_pareto_front()
    visualizer.plot_pareto_front(objectives)

    logging.info("🔹 راه‌حل‌های پارتو (u,v) و مقدار اهداف (cost, boundary):")
    for (u, v), (cost_val, bnd_val) in zip(solutions, objectives):
        logging.info(f"  u={u:.3f}, v={v:.3f}  →  cost={cost_val:,.2f},  boundary={bnd_val:.3f}")

    # انتخاب جفتِ (u*, v*) با کمترین اندازهٔ ناحیهٔ مرزی
    (best_u, best_v), best_obj = threshold_nsgaii.get_final_solution()
    logging.warning(
        f"🔹 بهترین جفت ضریب‌ها: u*={best_u:.3f}, v*={best_v:.3f}  →  "
        f"cost={best_obj[0]:,.2f},  boundary={best_obj[1]:.3f}"
    )

    logger.warning("11111111111111111111111111111111111111111111")
    logger.warning(f"best_u: {best_u}, best_v: {best_v}")
    logger.warning("22222222222222222222222222222222222222222222")

    visualizer.plot_with_thresholds(probabilities_test, u=best_u, v=best_v)

    visualizer.plot_default_prob_hist(probabilities_test,best_u,best_v)

    logging.info("گام چهارم (NSGA‑II چندهدفه) با موفقیت به پایان رسید.")

    # ------------------------------------------------------------------
    # 5) گام پنجم: اعمال تصمیم سه‌راهه با استفاده از (u*, v*)
    # ------------------------------------------------------------------
    threeway = ParsianThreeWayDecision(
        probabilities_test=probabilities_test,
        cost_matrix=all_costs,
        alpha_beta_pair=(best_u, best_v)  # (u*, v*)
    )
    decisions_final = threeway.apply_three_way_decision()

    cnts = threeway.get_decision_counts()
    logging.warning(
        f"Decision counts  →  POS: {cnts.get(1, 0)}   NEG: {cnts.get(0, 0)}   BND: {cnts.get(-1, 0)}"
    )

    # ------------------------------------------------------------------
    # 6) گام ششم: تعیین تکلیف نمونه‌های ناحیهٔ مرزی با مدل کمکی
    # ------------------------------------------------------------------
    bnd_resolver = ParsianBNDResolver(
        x_train_all=x_train,
        y_train_all=y_train,
        model_type="bagging"  # یا "stacking"
    )
    bnd_resolver.fit_bnd_model()

    decisions_updated = bnd_resolver.resolve_bnd_samples(x_test, decisions_final)

    logging.info("🔹 برچسب‌های نهایی پس از گام ششم:")
    logging.error(
        f"   POS={np.sum(decisions_updated == 1)}, "
        f"NEG={np.sum(decisions_updated == 0)}, "
        f"BND={np.sum(decisions_updated == -1)}"
    )

    # اعمال مدل روی نمونه‌های مرزی
    decisions_updated = bnd_resolver.resolve_bnd_samples(x_test, decisions_final)

    logging.info("🔹 برچسب‌های نهایی پس از گام ششم:")
    logging.error(
        f" count POS={np.sum(decisions_updated == 1)}, NEG={np.sum(decisions_updated == 0)}, BND={np.sum(decisions_updated == -1)}")

    # 7) گام هفتم: Evaluation نهایی
    final_eval = ParsianFinalEvaluator(true_labels=y_test.values, final_decisions=decisions_updated,
                                       probabilities_test=probabilities_test,  # اگر AUC بخواهیم
                                       cost_matrix=all_costs  # اگر هزینه بخواهیم
                                       )
    results = final_eval.evaluate_metrics()
    visualizer.plot_confusion_matrix(y_test.values, decisions_updated)

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
