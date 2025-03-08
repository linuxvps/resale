from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# protected_columns=None
protected_columns=['approval_amount', 'interest_amount']

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

        df = self.remove_highly_correlated_features(df, threshold=0.9, class_column=label_column)


        # جایگزینی داده‌های گمشده
        df_imputed = pd.DataFrame(self.imputer.fit_transform(df), columns=df.columns)

        # جداسازی ویژگی‌ها و برچسب‌ها
        X = df_imputed.drop(columns=[label_column])
        y = df_imputed[label_column]

        return X, y

    def remove_highly_correlated_features(self, data, threshold, class_column=None):
        """
        این تابع ویژگی‌هایی که همبستگی بالایی با یکدیگر دارند را حذف می‌کند.
        ورودی‌ها:
          data: دیتافریم ورودی
          threshold: آستانه همبستگی
          class_column: نام ستون کلاس (اختیاری)
          protected_columns: لیستی از ستون‌هایی که نباید حذف شوند (اختیاری)
        خروجی:
          دیتافریم جدید با حذف ویژگی‌های همبسته.
        """

        new_data = data.copy()
        numeric_cols = new_data.select_dtypes(include=[np.number]).columns.tolist()

        # حذف ستون کلاس از بررسی در صورت وجود
        if class_column and class_column in numeric_cols:
            numeric_cols.remove(class_column)

        # حذف ستون‌های محافظت‌شده از لیست پردازش
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

        # چاپ ماتریس همبستگی به صورت جدول
        print("ماتریس همبستگی:")
        print(corr_matrix.to_string())

        return new_data