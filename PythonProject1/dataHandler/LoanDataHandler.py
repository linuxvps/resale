from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple

from processor.LoanPreprocessor import LoanPreprocessor
from repository.LoanRepository import LoanRepository


class LoanDataHandler:
    """مدیریت دریافت، پردازش و تقسیم داده‌های وام به مجموعه‌های آموزش و تست."""

    def __init__(self, repository: LoanRepository, preprocessor: LoanPreprocessor):
        """
        مقداردهی اولیه با دریافت وابستگی‌های repository و preprocessor.
        :param repository: شیء مربوط به دریافت داده از دیتابیس
        :param preprocessor: شیء مربوط به پردازش داده‌ها
        """
        self.repository = repository
        self.preprocessor = preprocessor

    def load_and_process_data(self, limit_records: int = 10000) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        دریافت داده‌ها از دیتابیس، پردازش و تقسیم آن‌ها به مجموعه‌های آموزش و تست.
        :param limit_records: حداکثر تعداد رکوردها برای پردازش
        :return: مجموعه‌های آموزشی و تست (X_train, y_train, X_test, y_test)
        """
        df = self.repository.fetch_loans(limit_records)
        X, y = self.preprocessor.preprocess(df)

        # تقسیم داده‌ها به آموزش و تست
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, y_train, X_test, y_test
