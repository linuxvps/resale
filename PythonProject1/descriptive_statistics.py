# -*- coding: utf-8 -*-
"""
descriptive_statistics_advanced.py

محاسبه آمار توصیفی پیشرفته متغیرهای عددی:
Mean, StdDev, CV, Median, Min, Max, Range, IQR, Skew, Kurt, Missing%, Unique
و خروجی در descriptive_statistics_advanced.csv با حداکثر ۲ رقم اعشار.

مسیر فایل اکسل هاردکد شده است—در صورت نیاز آن را تغییر دهید.
"""

import sys
from pathlib import Path
import pandas as pd

def main():
    # مسیر هاردکد فایل اکسل
    data_path = Path(r'C:\Users\nima\data\Saeed_un - Copy.xlsx')
    if not data_path.exists():
        print(f"ERROR: File not found: {data_path}")
        sys.exit(1)

    # بارگذاری و حذف ستون‌های کاملاً خالی
    df = pd.read_excel(data_path).dropna(axis=1, how='all')

    # انتخاب فقط ستون‌های عددی
    df_num = df.select_dtypes(include='number')

    # ۱) محاسبات پایه
    stats = df_num.agg(['mean', 'std', 'median']).T
    stats.index.name = 'Variable'
    stats.columns = ['Mean', 'StdDev', 'Median']

    # ۲) محاسبه سایر متریک‌ها
    stats['CV']       = stats['StdDev'] / stats['Mean']
    stats['Min']      = df_num.min()
    stats['Max']      = df_num.max()
    stats['Range']    = stats['Max'] - stats['Min']
    q75 = df_num.quantile(0.75)
    q25 = df_num.quantile(0.25)
    stats['IQR']      = q75 - q25
    stats['Skew']     = df_num.skew()
    stats['Kurt']     = df_num.kurtosis()
    stats['Missing%'] = df_num.isna().mean() * 100
    stats['Unique']   = df_num.nunique()

    # ۳) مرتب‌سازی ستون‌ها
    cols = ['Mean','StdDev','CV','Median','Min','Max','Range','IQR',
            'Skew','Kurt','Missing%','Unique']
    cols = ['Mean','StdDev','CV','Median','Missing%','Unique']
    stats = stats[cols]

    # ۴) گرد کردن همهٔ مقادیر تا ۲ رقم اعشار
    stats = stats.round(2)

    # ۵) چاپ جدول
    print("\nAdvanced Descriptive Statistics (2 decimals max):\n")
    print(stats.to_string())

    # ۶) ذخیره در CSV
    out_file = Path('results/statistics/descriptive_statistics.csv')
    stats.to_csv(out_file, index=True)
    print(f"\nSaved enhanced descriptive statistics to: {out_file}")

if __name__ == '__main__':
    main()
