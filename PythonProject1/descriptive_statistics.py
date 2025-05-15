# -*- coding: utf-8 -*-
"""
descriptive_statistics_advanced.py

این اسکریپت آمار توصیفی پیشرفته متغیرهای عددی را محاسبه می‌کند:
- Mean, StdDev, CV, Median, Range
- IQR (Interquartile Range)
- Skewness (چولگی)
- Kurtosis (کشیدگی دم)
- Missing% (درصد مقادیر گمشده)
- Unique (تعداد مقادیر یکتا)

و خروجی را در descriptive_statistics_advanced.csv ذخیره می‌کند.

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

    # محاسبه مقدماتی
    stats = df_num.agg(['mean', 'std', 'median']).T
    stats.index.name = 'Variable'
    stats.columns = ['Mean', 'StdDev', 'Median']

    # محاسبه سایر متریک‌ها
    stats['CV']     = stats['StdDev'] / stats['Mean']                    # ضریب تغییرات
    stats['Min']    = df_num.min()
    stats['Max']    = df_num.max()
    stats['Range']  = stats['Max'] - stats['Min']
    q75 = df_num.quantile(0.75)
    q25 = df_num.quantile(0.25)
    stats['IQR']    = q75 - q25                                          # بازه چارکی
    stats['Skew']   = df_num.skew()                                      # چولگی
    stats['Kurt']   = df_num.kurtosis()                                  # کشیدگی دم
    stats['Missing%'] = df_num.isna().mean() * 100                       # درصد گمشده
    stats['Unique'] = df_num.nunique()                                   # تعداد مقادیر یکتا

    # مرتب‌سازی ستون‌ها
    cols = ['Mean','StdDev','CV','Median','Min','Max','Range','IQR','Skew','Kurt','Missing%','Unique']
    cols = ['Mean','StdDev','CV','Median','IQR','Missing%','Unique']
    stats = stats[cols]

    # چاپ
    print("\nAdvanced Descriptive Statistics:\n")
    print(stats.to_string(float_format="{:,.2f}".format))

    # ۵) ذخیره در CSV
    out_file = Path('results/statistics/descriptive_statistics.csv')
    stats.to_csv(out_file, index=True)
    print(f"\nSaved advanced descriptive statistics to: {out_file}")

if __name__ == '__main__':
    main()

