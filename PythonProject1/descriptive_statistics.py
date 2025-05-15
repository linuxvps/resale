# -*- coding: utf-8 -*-
"""
descriptive_statistics_enhanced.py

این اسکریپت آمار توصیفی متغیرهای عددی را محاسبه می‌کند و به‌جای Min/Max،
از Range (بازه) و CV (ضریب تغییرات) استفاده می‌کند تا نسبت به پرت‌ها
مستحکم‌تر باشد. خروجی را در descriptive_statistics_enhanced.csv ذخیره می‌کند.

مسیر فایل اکسل هاردکد شده است—در صورت نیاز آن را تغییر دهید.
"""

import sys
from pathlib import Path
import pandas as pd

def main():
    # مسیر هاردکد فایل اکسل
    data_path = Path(r'C:\Users\nima\data\ln_loans_1000.xlsx')
    if not data_path.exists():
        print(f"ERROR: File not found: {data_path}")
        sys.exit(1)

    # بارگذاری داده و حذف ستون‌های کاملاً خالی
    df = pd.read_excel(data_path).dropna(axis=1, how='all')

    # انتخاب فقط ستون‌های عددی
    df_num = df.select_dtypes(include='number')

    # ۱) محاسبه میانگین، انحراف معیار و میانه
    stats = df_num.agg(['mean', 'std', 'median']).T
    stats.index.name = 'Variable'
    stats.columns = ['Mean', 'StdDev', 'Median']

    # ۲) محاسبه Range و CV
    # Range = حداکثر منهای حداقل، CV = StdDev / Mean
    rng = df_num.max() - df_num.min()
    stats['Range'] = rng.values
    stats['CV']    = stats['StdDev'] / stats['Mean']

    # ۳) مرتب‌سازی ستون‌ها
    stats = stats[['Mean', 'StdDev', 'CV', 'Median', 'Range']]

    # ۴) چاپ جدول
    print("\nEnhanced Descriptive Statistics:\n")
    print(stats.to_string(float_format="{:,.2f}".format))

    # ۵) ذخیره در CSV
    out_file = Path('results/statistics/descriptive_statistics.csv')
    stats.to_csv(out_file, index=True)
    print(f"\nSaved enhanced descriptive statistics to: {out_file}")

if __name__ == '__main__':
    main()
