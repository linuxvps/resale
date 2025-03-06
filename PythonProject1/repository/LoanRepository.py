from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
import pandas as pd
from ParsianLoan import ParsianLoan  # مدل پایگاه داده را ایمپورت کنید

# مقدار URL پایگاه داده را متناسب با پروژه تغییر دهید
DATABASE_URL = "mysql+pymysql://root:pass@localhost:3306/ln"

class LoanRepository:
    """مدیریت عملیات خواندن داده‌ها از پایگاه داده."""

    def __init__(self):
        """
        مقداردهی اولیه ریپازیتوری بدون نیاز به ارسال session.
        """
        self.session = self.create_database_session()

    @staticmethod
    def create_database_session() -> Session:
        """
        ایجاد سشن SQLAlchemy برای ارتباط با پایگاه داده.
        :return: شیء Session برای ارتباط با پایگاه داده
        """
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        return SessionLocal()  # ایجاد یک session معتبر

    def fetch_loans(self, limit: int = 10000) -> pd.DataFrame:
        """
        دریافت داده‌ها از دیتابیس و تبدیل آن‌ها به DataFrame.
        :param limit: تعداد رکوردهایی که باید دریافت شوند
        :return: DataFrame شامل داده‌های وام
        """
        loans = self.session.query(ParsianLoan).limit(limit).all()
        if not loans:
            print("⚠️ هیچ داده‌ای از پایگاه داده دریافت نشد.")
            return pd.DataFrame()  # بازگشت دیتافریم خالی

        df = pd.DataFrame([loan.__dict__ for loan in loans])
        df.drop(columns=["_sa_instance_state"], inplace=True, errors="ignore")
        print(f"✅ {len(df)} رکورد از دیتابیس دریافت شد.")
        return df
