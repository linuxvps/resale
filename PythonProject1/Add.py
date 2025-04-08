import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# برقراری اتصال به دیتابیس
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "mysql+pymysql://root:pass@localhost:3306/ln")
engine = create_engine(DB_CONNECTION_STRING)
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

# تعداد رکوردهای تستی مورد نظر
NUM_RECORDS = 10000

# تابع کمکی برای تولید تاریخ تصادفی بین دو سال
def random_date(start_year=2000, end_year=2020):
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    delta = (end_date - start_date).days
    random_days = random.randint(0, delta)
    return (start_date + timedelta(days=random_days)).date()

# لیست‌های از پیش تعریف‌شده برای مقادیر متنی
branchnames = ["نوربخش", "برج تجارت", "میرداماد", "میدان", "سبزوار", "دولت", "بلوار امیر"]
contracts = ["تسويه", "دوره بازپرداخت", "مضاربه", "فروش اقساطی"]
long_titles = ["مضاربه بازرگانی", "فروش اقساطی", "مضاربه", "تسويه"]
statuses = ["خاتمه عادي", "فعال", "باطل شده", "سررسيد شده", "ضمانت نامه صادر شده",
            "سررسيد گذشته", "با طل شده", "معوق", "ابطال مصوبه ضمانت نامه (قبل از صدور)",
            "مشكوك الوصول", "منقضي شده", "پرداخت شده كامل", "وثيقه ناقص",
            "درخواست رد شد", "اعطاء كامل", "درخواست رد شده", "جاري", "اعطاء ناقص"]
is_installment_options = ["Y", "N"]
sit_flag_options = ["Y", "N"]

# تولید داده‌های تصادفی برای هر ستون طبق ساختار جدول parsian_loan_2
data = {
    "advance_pay": [round(random.uniform(0, 10000000), 8) for _ in range(NUM_RECORDS)],
    "advance_pay_to_remain_non_cash": [round(random.uniform(0, 10000000), 8) for _ in range(NUM_RECORDS)],
    "advance_pay_to_total_cash": [round(random.uniform(0, 10000000), 8) for _ in range(NUM_RECORDS)],
    "approval_amount": [round(random.uniform(10000000, 5000000000), 8) for _ in range(NUM_RECORDS)],
    "bank_share_cash_amount": [round(random.uniform(0, 5000000000), 8) for _ in range(NUM_RECORDS)],
    "bank_share_non_cash_amount": [round(random.uniform(0, 5000000000), 8) for _ in range(NUM_RECORDS)],
    "branch_code": [random.randint(1000, 3000) for _ in range(NUM_RECORDS)],
    "branchname": [random.choice(branchnames) for _ in range(NUM_RECORDS)],
    "charge": [round(random.uniform(0, 1000000), 8) for _ in range(NUM_RECORDS)],
    "loan_file_numberr": [random.randint(58000000000000, 59000000000000) for _ in range(NUM_RECORDS)],
    "client_id": [random.randint(100000, 1000000) for _ in range(NUM_RECORDS)],
    "commission_amount_remain": [round(random.uniform(0, 1000000), 8) for _ in range(NUM_RECORDS)],
    "contract": [random.choice(contracts) for _ in range(NUM_RECORDS)],
    "create_date": [random_date() for _ in range(NUM_RECORDS)],
    "customer_obligation_amount": [round(random.uniform(0, 10000000), 8) for _ in range(NUM_RECORDS)],
    "customer_share_cash_amount": [round(random.uniform(0, 10000000), 8) for _ in range(NUM_RECORDS)],
    "customer_share_non_cash_amount": [round(random.uniform(0, 10000000), 8) for _ in range(NUM_RECORDS)],
    "discount": [round(random.uniform(0, 500000), 8) for _ in range(NUM_RECORDS)],
    "due_date": [random_date() for _ in range(NUM_RECORDS)],
    "finalized_loan_amount": [round(random.uniform(10000000, 5000000000), 8) for _ in range(NUM_RECORDS)],
    "first_over_due": [random_date() if random.choice([True, False]) else None for _ in range(NUM_RECORDS)],
    "first_passed": [random_date() if random.choice([True, False]) else None for _ in range(NUM_RECORDS)],
    "first_payment_date_in_du": [random_date() if random.choice([True, False]) else None for _ in range(NUM_RECORDS)],
    "frequency": [random.randint(1, 12) for _ in range(NUM_RECORDS)],
    "inc_commission_amount": [round(random.uniform(0, 1000000), 8) for _ in range(NUM_RECORDS)],
    "insert_sysdate": [datetime.now() for _ in range(NUM_RECORDS)],
    "installment_number_remain": [random.randint(0, 100) for _ in range(NUM_RECORDS)],
    "interest_amount": [round(random.uniform(0, 5000000), 8) for _ in range(NUM_RECORDS)],
    "interest_rate": [round(random.uniform(0, 30), 2) for _ in range(NUM_RECORDS)],
    "interest_sum": [round(random.uniform(0, 5000000), 8) for _ in range(NUM_RECORDS)],
    "is_installment": [random.choice(is_installment_options) for _ in range(NUM_RECORDS)],
    "loan_duration_day": [random.randint(30, 365) for _ in range(NUM_RECORDS)],
    "loan_file_number": [random.randint(58000000000000, 59000000000000) for _ in range(NUM_RECORDS)],
    "long_title": [random.choice(long_titles) for _ in range(NUM_RECORDS)],
    "obligation_penalty": [round(random.uniform(0, 1000000), 8) for _ in range(NUM_RECORDS)],
    "passed_date": [random_date() if random.choice([True, False]) else None for _ in range(NUM_RECORDS)],
    "penalty": [round(random.uniform(0, 500000), 8) for _ in range(NUM_RECORDS)],
    "penalty_interest": [round(random.uniform(0, 500000), 8) for _ in range(NUM_RECORDS)],
    "principal_sum": [round(random.uniform(0, 5000000000), 8) for _ in range(NUM_RECORDS)],
    "receivable_installment_number": [random.randint(0, 50) for _ in range(NUM_RECORDS)],
    "sit_distribute_phases": [random.randint(0, 5) for _ in range(NUM_RECORDS)],
    "sit_duration": [random.randint(0, 1000) for _ in range(NUM_RECORDS)],
    "sit_duration_day": [random.randint(0, 1000) for _ in range(NUM_RECORDS)],
    "sit_fast_receive_percent": [round(random.uniform(0, 100), 8) for _ in range(NUM_RECORDS)],
    "sit_flag": [random.choice(sit_flag_options) for _ in range(NUM_RECORDS)],
    "status": [random.choice(statuses) for _ in range(NUM_RECORDS)],
    "title": [random.choice(contracts) for _ in range(NUM_RECORDS)],  # استفاده از لیست قراردادها برای عنوان
    "to_due_date": [round(random.uniform(0, 5000000000), 8) for _ in range(NUM_RECORDS)],
    "to_end_of_month": [round(random.uniform(0, 5000000000), 8) for _ in range(NUM_RECORDS)],
    "total_payment_up_to_now": [round(random.uniform(0, 5000000000), 8) for _ in range(NUM_RECORDS)],
    "total_repayment_up_to_now": [round(random.uniform(0, 5000000000), 8) for _ in range(NUM_RECORDS)],
    "is_synthetic": [0 for _ in range(NUM_RECORDS)]  # مقدار 0 برای داده‌های تستی واقعی
}

# ساخت DataFrame نهایی
df_test = pd.DataFrame(data)

# درج داده‌های تولیدشده در جدول parsian_loan_2
df_test.to_sql("parsian_loan_2", con=engine, if_exists="append", index=False)

print(f"تعداد {len(df_test)} رکورد تستی به جدول parsian_loan_2 اضافه شد.")
