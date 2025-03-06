import pandas as pd
from sklearn.impute import SimpleImputer
from typing import Tuple
import numpy as np
from sklearn.preprocessing import LabelEncoder


class LoanPreprocessor:
    """کلاس مسئول پیش‌پردازش داده‌های وام."""

    def __init__(self, imputation_strategy: str = "mean"):
        """
        مقداردهی اولیه پیش‌پردازشگر با امکان تنظیم استراتژی جایگزینی داده‌های گمشده.
        مقدارهای معتبر برای imputation_strategy:
        - "mean" (میانگین)
        - "median" (میانه)
        - "most_frequent" (مد پرتکرار)
        """
        self.imputer = SimpleImputer(strategy=imputation_strategy)

    # ------------------------ تبدیل نوع داده‌ها ------------------------
    def convert_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        تمام ستون‌های DataFrame را بررسی می‌کند؛ اگر ستون به نوع datetime باشد، آن را به تعداد روزهای سپری‌شده از تاریخ مرجع (2000-01-01) تبدیل می‌کند.
        اگر ستون دارای نوع object باشد، ابتدا تلاش می‌کند آن را به عدد تبدیل کند؛ در صورت عدم موفقیت، از LabelEncoder برای تبدیل داده‌های دسته‌بندی استفاده می‌کند.
        """
        for col in df.columns:
            # تبدیل ستون‌های تاریخ به تعداد روزها
            if np.issubdtype(df[col].dtype, np.datetime64):
                df[col] = (pd.to_datetime(df[col]) - pd.Timestamp("2000-01-01")).dt.days
            elif df[col].dtype == 'object':
                try:
                    # تلاش برای تبدیل به عدد؛ در صورت موفقیت، مقدارهای قابل تبدیل باقی می‌مانند
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    # در صورت عدم موفقیت، از LabelEncoder استفاده می‌کنیم
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
        return df

    def encode_labels(self, column: pd.Series) -> pd.Series:
        """استفاده از `LabelEncoder` برای تبدیل داده‌های متنی به عددی."""
        le = LabelEncoder()
        return le.fit_transform(column.astype(str))

    def convert_labels(self, df: pd.DataFrame, label_column: str = "status") -> pd.DataFrame:
        """
        تبدیل مقدارهای کیفی در ستون وضعیت (status) به مقدارهای عددی ۰ و ۱.
        :param df: دیتافریم اصلی
        :param label_column: نام ستون برچسب
        :return: دیتافریم با مقادیر عددی شده در ستون برچسب
        """
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

    def preprocess(self, df: pd.DataFrame, label_column: str = "status") -> Tuple[pd.DataFrame, pd.Series]:
        """
        اجرای فرآیندهای پیش‌پردازش روی داده‌ها و آماده‌سازی X و y.
        :param df: دیتافریم ورودی
        :param label_column: نام ستون برچسب
        :return: ویژگی‌های X و برچسب‌های y
        """
        df = self.convert_labels(df, label_column)

        # تبدیل تمامی ستون‌های غیرعددی به داده‌های عددی
        df = self.convert_dataframe_columns(df)
        df.drop(columns=["create_date"], errors="ignore", inplace=True)

        # جایگزینی داده‌های گمشده
        df_imputed = pd.DataFrame(self.imputer.fit_transform(df), columns=df.columns)

        # جداسازی ویژگی‌ها و برچسب‌ها
        X = df_imputed.drop(columns=[label_column])
        y = df_imputed[label_column]

        return X, y