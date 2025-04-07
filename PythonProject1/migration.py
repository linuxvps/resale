# """
# این اسکریپت با نام make_synthetic_parsian_loan2.py ذخیره شود.
# وظیفه:
#   - از جدول parsian_loan داده‌ها به‌صورت طبقه‌بندی‌شده بر اساس status گرفته می‌شود،
#     به‌طوری که درصد رکوردهای هر status همانند داده‌های اصلی حفظ شود.
#   - مجموعاً ۱۰۰,۰۰۰ رکورد (نمونه نهایی) به‌دست می‌آید؛ سپس در هر دسته به‌صورت سیستماتیک
#     تعداد معینی (۱۰٪ از رکوردهای انتخابی آن دسته) رکورد به داده‌های ساختگی تبدیل می‌شوند.
#     در داده‌های ساختگی، مقدار ستون‌های approval_amount و interest_amount به بازه میانی تغییر می‌یابد
#     و ستون is_synthetic به ۱ تنظیم می‌شود، در حالی که بقیه رکوردها به عنوان واقعی (is_synthetic = 0)
#     باقی می‌مانند.
#   - داده‌های نهایی (واقعی + ساختگی) در جدول parsian_loan_2 درج می‌شوند.
#
# فرضیات:
#   - در دیتابیس "ln" جدولی به نام parsian_loan وجود دارد.
#   - توزیع اولیه رکوردها بر اساس status طبق گزارش زیر است:
#       خاتمه عادي,426401
#       فعال,320337
#       باطل شده,124465
#       سررسيد شده,85933
#       ضمانت نامه صادر شده,76121
#       سررسيد گذشته,23633
#       با طل شده,17810
#       معوق,6615
#       ابطال مصوبه ضمانت نامه (قبل از صدور),6247
#       مشكوك الوصول,4171
#       منقضي شده,418
#       پرداخت شده كامل,333
#       وثيقه ناقص,254
#       درخواست رد شد,134
#       اعطاء كامل,102
#       درخواست رد شده,50
#       جاري,30
#       اعطاء ناقص,10
#     که مجموعاً ۱,۰۹۳,۰۶۴ رکورد می‌باشد.
#
# """
#
# import os
# import logging
# import random
# import math
# import numpy as np
# import pandas as pd
# from sqlalchemy import create_engine, text
# from sqlalchemy.orm import sessionmaker
#
# # پیکربندی لاگر
# logging.basicConfig(level=logging.INFO)
#
# # پارامترهای اولیه
# DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "mysql+pymysql://root:pass@localhost:3306/ln")
# MAX_RECORDS = 100_000      # تعداد رکورد نهایی مورد نظر
# FAKE_PERCENT = 0.10        # درصد داده ساختگی در هر دسته
# NEW_TABLE_NAME = "parsian_loan_3"
# EXTRA_COLUMN_NAME = "is_synthetic"
#
# # بازه‌های تنظیم‌شده برای تغییر مقادیر کلیدی در داده ساختگی
# APPROVAL_MIN, APPROVAL_MAX = 50_000_000, 100_000_000
# INTEREST_MIN, INTEREST_MAX = 500_000, 3_000_000
#
# # اتصال به دیتابیس
# engine = create_engine(DB_CONNECTION_STRING)
# SessionLocal = sessionmaker(bind=engine)
# session = SessionLocal()
#
# # ایجاد جدول جدید parsian_loan_2 به صورت کپی از ساختار parsian_loan و اضافه کردن ستون is_synthetic
# CREATE_TABLE_SQL = f"""
# CREATE TABLE IF NOT EXISTS {NEW_TABLE_NAME}
# LIKE parsian_loan;
# """
# ADD_COLUMN_SQL = f"""
# ALTER TABLE {NEW_TABLE_NAME}
# ADD COLUMN {EXTRA_COLUMN_NAME} TINYINT DEFAULT 0;
# """
# with engine.connect() as conn:
#     conn.execute(text(CREATE_TABLE_SQL))
#     try:
#         conn.execute(text(ADD_COLUMN_SQL))
#     except Exception as e:
#         logging.warning(f"ستون {EXTRA_COLUMN_NAME} احتمالاً وجود دارد یا خطایی رخ داده: {e}")
#
# # گرفتن توزیع رکوردها بر اساس status از جدول parsian_loan
# distribution_sql = """
# SELECT status, COUNT(*) as cnt
# FROM parsian_loan
# GROUP BY status
# ORDER BY cnt DESC;
# """
# df_distribution = pd.read_sql(distribution_sql, engine)
# total_count = df_distribution['cnt'].sum()
# logging.info(f"تعداد کل رکوردهای جدول اصلی: {total_count}")
#
# # محاسبه تعداد رکورد نمونه نهایی برای هر دسته به‌طوری که مجموعاً MAX_RECORDS رکورد شود
# sample_sizes = {}
# for _, row in df_distribution.iterrows():
#     status_val = row['status']
#     cnt = row['cnt']
#     sample_num = round(MAX_RECORDS * (cnt / total_count))
#     sample_sizes[status_val] = sample_num
#
# # تنظیم مجدد در صورت اختلاف ناشی از گرد کردن (اختلاف را به دسته با بیشترین تعداد اختصاص می‌دهیم)
# sample_total = sum(sample_sizes.values())
# if sample_total != MAX_RECORDS:
#     diff = MAX_RECORDS - sample_total
#     main_status = df_distribution.iloc[0]['status']
#     sample_sizes[main_status] += diff
#     logging.info(f"تنظیم مجدد تعداد نمونه در {main_status} به دلیل اختلاف گردکردن: اضافه {diff} رکورد")
#
# logging.info("تعداد رکوردهای نمونه انتخاب شده برای هر دسته:")
# for status_val, num in sample_sizes.items():
#     logging.info(f"  {status_val}: {num} رکورد")
#
# # لیست نهایی برای ذخیره داده‌های نمونه
# final_dataframes = []
#
# # برای هر دسته، ابتدا رکوردهای مورد نظر را به صورت تصادفی (با حفظ نسبت اصلی) انتخاب می‌کنیم
# # سپس الگوریتم سیستماتیک برای انتخاب رکوردهای ساختگی اجرا می‌شود
# for status_val, sample_num in sample_sizes.items():
#     query = f"""
#     SELECT *
#     FROM parsian_loan
#     WHERE status = :status_val
#     ORDER BY RAND()
#     LIMIT {sample_num}
#     """
#     df_group = pd.read_sql(text(query), engine, params={"status_val": status_val})
#
#     if len(df_group) < sample_num:
#         logging.warning(f"تعداد رکوردهای دریافت شده برای {status_val} کمتر از انتظار است: {len(df_group)} رکورد دریافت شد.")
#
#     # تعداد رکوردهای ساختگی در این دسته
#     num_fake = math.ceil(len(df_group) * FAKE_PERCENT)
#     logging.info(f"برای دسته '{status_val}' از {len(df_group)} رکورد، {num_fake} رکورد به صورت ساختگی تغییر داده خواهد شد.")
#
#     # الگوریتم سیستماتیک: با فاصله مساوی انتخاب رکوردها به این صورت که
#     # فاصله = تعداد رکوردها تقسیم بر تعداد رکوردهای ساختگی
#     # اندیس انتخاب شده = int(round(i * فاصله + فاصله/2)) برای i از 0 تا num_fake-1
#     if num_fake > 0:
#         n = len(df_group)
#         step = n / num_fake
#         for i in range(num_fake):
#             pos = int(round(i * step + step/2))
#             if pos >= n:
#                 pos = n - 1
#             row_index = df_group.index[pos]
#             # تغییر مقدار ستون‌های کلیدی به بازه میانی
#             if "approval_amount" in df_group.columns:
#                 df_group.at[row_index, "approval_amount"] = random.uniform(APPROVAL_MIN, APPROVAL_MAX)
#             if "interest_amount" in df_group.columns:
#                 df_group.at[row_index, "interest_amount"] = random.uniform(INTEREST_MIN, INTEREST_MAX)
#             # تنظیم ستون is_synthetic به ۱ برای داده ساختگی
#             df_group.at[row_index, EXTRA_COLUMN_NAME] = 1
#     # بقیه رکوردها به عنوان واقعی (is_synthetic = 0) علامت‌گذاری می‌شوند
#     df_group.loc[df_group[EXTRA_COLUMN_NAME].isnull(), EXTRA_COLUMN_NAME] = 0
#
#     final_dataframes.append(df_group)
#
# # ادغام تمامی داده‌های انتخاب شده از دسته‌های مختلف
# df_final = pd.concat(final_dataframes, ignore_index=True)
# logging.info(f"کل رکورد نهایی پس از ادغام: {len(df_final)}")
#
# # پاکسازی جدول جدید parsian_loan_2 در صورت وجود داده‌های قدیمی
# with engine.connect() as conn:
#     conn.execute(text(f"TRUNCATE TABLE {NEW_TABLE_NAME}"))
#
# logging.info(f"در حال درج رکوردها در {NEW_TABLE_NAME} ...")
# df_final.to_sql(NEW_TABLE_NAME, con=engine, if_exists='append', index=False)
# logging.info(f"✅ {len(df_final)} رکورد (واقعی + ساختگی) در جدول {NEW_TABLE_NAME} درج شد.")



import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# برقراری اتصال به دیتابیس
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "mysql+pymysql://root:pass@localhost:3306/ln")
engine = create_engine(DB_CONNECTION_STRING)
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

NEW_TABLE_NAME = "parsian_loan_4"

# ایجاد جدول جدید parsian_loan_4 با ساختار مشابه parsian_loan و پاکسازی داده‌های قبلی
CREATE_TABLE_SQL = f"CREATE TABLE IF NOT EXISTS {NEW_TABLE_NAME} LIKE parsian_loan;"
with engine.connect() as conn:
    conn.execute(text(CREATE_TABLE_SQL))
    conn.execute(text(f"TRUNCATE TABLE {NEW_TABLE_NAME}"))

# تابع کمکی برای تولید تاریخ تصادفی
def random_date(start_year=2000, end_year=2020):
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return (start_date + timedelta(days=random_days)).date()

# تعداد رکوردهای تستی مورد نظر
num_records = 10000

# دیکشنری وزن‌های وضعیت‌ها بر اساس داده‌های ارائه‌شده
status_weights = {
    "خاتمه عادي": 426401,
    "فعال": 320337,
    "باطل شده": 124465,
    "سررسيد شده": 85933,
    "ضمانت نامه صادر شده": 76121,
    "سررسيد گذشته": 23633,
    "با طل شده": 17810,
    "معوق": 6615,
    "ابطال مصوبه ضمانت نامه (قبل از صدور)": 6247,
    "مشكوك الوصول": 4171,
    "منقضي شده": 418,
    "پرداخت شده كامل": 333,
    "وثيقه ناقص": 254,
    "درخواست رد شد": 134,
    "اعطاء كامل": 102,
    "درخواست رد شده": 50,
    "جاري": 30,
    "اعطاء ناقص": 10
}

# محاسبه احتمال هر وضعیت بر اساس وزن‌های داده‌شده
status_list = list(status_weights.keys())
weights = list(status_weights.values())
total_weight = sum(weights)
probabilities = [w/total_weight for w in weights]
statuses = np.random.choice(status_list, size=num_records, p=probabilities)

# تولید داده‌های تصادفی برای سایر ستون‌ها
data = {
    "advance_pay": [round(random.uniform(0, 10000000), 8) for _ in range(num_records)],
    "advance_pay_to_remain_non_cash": [round(random.uniform(0, 10000000), 8) for _ in range(num_records)],
    "advance_pay_to_total_cash": [round(random.uniform(0, 10000000), 8) for _ in range(num_records)],
    "approval_amount": [round(random.uniform(10000000, 5000000000), 8) for _ in range(num_records)],
    "bank_share_cash_amount": [round(random.uniform(0, 5000000000), 8) for _ in range(num_records)],
    "bank_share_non_cash_amount": [round(random.uniform(0, 5000000000), 8) for _ in range(num_records)],
    "branch_code": [random.randint(1000, 3000) for _ in range(num_records)],
    "branchname": [random.choice(["نوربخش", "برج تجارت", "میرداماد", "میدان", "سبزوار", "دولت", "بلوار امیر"]) for _ in range(num_records)],
    "charge": [round(random.uniform(0, 1000000), 8) for _ in range(num_records)],
    "loan_file_numberr": [random.randint(58000000000000, 59000000000000) for _ in range(num_records)],
    "client_id": [random.randint(100000, 1000000) for _ in range(num_records)],
    "commission_amount_remain": [round(random.uniform(0, 1000000), 8) for _ in range(num_records)],
    "contract": [random.choice(["تسويه", "دوره بازپرداخت", "مضاربه", "فروش اقساطی"]) for _ in range(num_records)],
    "create_date": [random_date() for _ in range(num_records)],
    "customer_obligation_amount": [round(random.uniform(0, 10000000), 8) for _ in range(num_records)],
    "customer_share_cash_amount": [round(random.uniform(0, 10000000), 8) for _ in range(num_records)],
    "customer_share_non_cash_amount": [round(random.uniform(0, 10000000), 8) for _ in range(num_records)],
    "discount": [round(random.uniform(0, 500000), 8) for _ in range(num_records)],
    "due_date": [random_date() for _ in range(num_records)],
    "finalized_loan_amount": [round(random.uniform(10000000, 5000000000), 8) for _ in range(num_records)],
    "first_over_due": [random_date() if random.choice([True, False]) else None for _ in range(num_records)],
    "first_passed": [random_date() if random.choice([True, False]) else None for _ in range(num_records)],
    "first_payment_date_in_du": [random_date() if random.choice([True, False]) else None for _ in range(num_records)],
    "frequency": [random.randint(1, 12) for _ in range(num_records)],
    "inc_commission_amount": [round(random.uniform(0, 1000000), 8) for _ in range(num_records)],
    "insert_sysdate": [datetime.now() for _ in range(num_records)],
    "installment_number_remain": [random.randint(0, 100) for _ in range(num_records)],
    "interest_amount": [round(random.uniform(0, 5000000), 8) for _ in range(num_records)],
    "interest_rate": [round(random.uniform(0, 30), 2) for _ in range(num_records)],
    "interest_sum": [round(random.uniform(0, 5000000), 8) for _ in range(num_records)],
    "is_installment": [random.choice(["Y", "N"]) for _ in range(num_records)],
    "loan_duration_day": [random.randint(30, 365) for _ in range(num_records)],
    "loan_file_number": [random.randint(58000000000000, 59000000000000) for _ in range(num_records)],
    "long_title": [random.choice(["مضاربه بازرگانی", "فروش اقساطی", "مضاربه", "تسويه"]) for _ in range(num_records)],
    "obligation_penalty": [round(random.uniform(0, 1000000), 8) for _ in range(num_records)],
    "passed_date": [random_date() if random.choice([True, False]) else None for _ in range(num_records)],
    "penalty": [round(random.uniform(0, 500000), 8) for _ in range(num_records)],
    "penalty_interest": [round(random.uniform(0, 500000), 8) for _ in range(num_records)],
    "principal_sum": [round(random.uniform(0, 5000000000), 8) for _ in range(num_records)],
    "receivable_installment_number": [random.randint(0, 50) for _ in range(num_records)],
    "sit_distribute_phases": [random.randint(0, 5) for _ in range(num_records)],
    "sit_duration": [random.randint(0, 1000) for _ in range(num_records)],
    "sit_duration_day": [random.randint(0, 1000) for _ in range(num_records)],
    "sit_fast_receive_percent": [round(random.uniform(0, 100), 8) for _ in range(num_records)],
    "sit_flag": [random.choice(["Y", "N"]) for _ in range(num_records)],
    "status": statuses,
    "title": [random.choice(["تسويه", "مضاربه بازرگانی داخلی", "فروش اقساطی خدمات"]) for _ in range(num_records)],
    "to_due_date": [round(random.uniform(0, 5000000000), 8) for _ in range(num_records)],
    "to_end_of_month": [round(random.uniform(0, 5000000000), 8) for _ in range(num_records)],
    "total_payment_up_to_now": [round(random.uniform(0, 5000000000), 8) for _ in range(num_records)],
    "total_repayment_up_to_now": [round(random.uniform(0, 5000000000), 8) for _ in range(num_records)]
}

df_test = pd.DataFrame(data)
df_test.to_sql(NEW_TABLE_NAME, con=engine, if_exists='append', index=False)
print(f"تعداد {len(df_test)} رکورد تستی در جدول {NEW_TABLE_NAME} درج شد.")
