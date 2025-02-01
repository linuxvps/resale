package com.example.demo;

import lombok.RequiredArgsConstructor;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.poi.xwpf.usermodel.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.FileOutputStream;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;

@Service
@RequiredArgsConstructor
public class CreditRiskService {

    @Autowired
    private LendingClubRepository lendingClubRepository;

    // تعریف نقشه برای ترجمه نام ستون‌ها به فارسی
    public static final Map<String, String> COLUMN_NAME_TRANSLATIONS = new HashMap<>();
    static {
        COLUMN_NAME_TRANSLATIONS.put("loan_amnt", "مبلغ وام");
        COLUMN_NAME_TRANSLATIONS.put("int_rate", "نرخ بهره");
        COLUMN_NAME_TRANSLATIONS.put("installment", "قسط ماهانه");
        COLUMN_NAME_TRANSLATIONS.put("annual_inc", "درآمد سالانه");
        COLUMN_NAME_TRANSLATIONS.put("dti", "نسبت بدهی به درآمد");
        COLUMN_NAME_TRANSLATIONS.put("delinq_2yrs", "تأخیر بیش از 2 سال");
        COLUMN_NAME_TRANSLATIONS.put("revol_bal", "مانده اعتباری");
        COLUMN_NAME_TRANSLATIONS.put("total_acc", "تعداد حساب‌ها");
        COLUMN_NAME_TRANSLATIONS.put("open_acc", "حساب‌های فعال");
        COLUMN_NAME_TRANSLATIONS.put("pub_rec", "سوابق عمومی");
    }

    public Map<String, Object> getColumnStatistics(String columnName, String condition) {
        List<Double> values = lendingClubRepository.findColumnByCondition(columnName, condition);
        Map<String, Object> stats = calculateStatistics(values);
        stats.put("columnName",COLUMN_NAME_TRANSLATIONS.get(columnName));
        System.out.println(stats);
        return stats;
    }


    public List<Map<String, Object>> calcStatForList() {
        List<Map<String, Object>> results = new ArrayList<>();

        // خواندن ستون‌ها از نقشه ترجمه
        for (String columnName : COLUMN_NAME_TRANSLATIONS.keySet()) {
            Map<String, Object> stats = getColumnStatistics(columnName, "1=1");
            stats.put("نام ستون", COLUMN_NAME_TRANSLATIONS.get(columnName)); // افزودن نام فارسی ستون
            results.add(stats);
        }
        return results;
    }
    public Map<String, Object> calculateStatistics(List<Double> data) {
        DescriptiveStatistics stats = new DescriptiveStatistics();
        for (Double value : data) {
            if (value != null) {
                stats.addValue(value);
            }
        }

        double max = stats.getMax();
        double min = stats.getMin();
        long count = stats.getN();
        double mean = stats.getMean();
        double stdDev = stats.getStandardDeviation();
        double variance = stats.getVariance();
        double skew = stats.getSkewness();
        double kurt = stats.getKurtosis();

        Map<String, Object> result = new HashMap<>();
        result.put("بیشینه", max);
        result.put("کمینه", min);
        result.put("تعداد مشاهده", count);
        result.put("میانگین", mean);
        result.put("انحراف معیار", stdDev);
        result.put("واریانس", variance);
        result.put("چولگی", skew);
        result.put("کشیدگی", kurt);

        return result;

    }

    public void createWordReport(List<Map<String, Object>> results, String fileName) {
        try (XWPFDocument document = new XWPFDocument(); FileOutputStream out = new FileOutputStream(fileName)) {
            DecimalFormat df = new DecimalFormat("#0"); // حذف ویرگول از اعداد

            // عنوان گزارش
            XWPFParagraph title = document.createParagraph();
            title.setAlignment(ParagraphAlignment.CENTER);
            XWPFRun titleRun = title.createRun();
            titleRun.setText("گزارش آمار ریسک اعتباری");
            titleRun.setBold(true);
            titleRun.setFontSize(18);
            titleRun.setFontFamily("B Nazanin");
            titleRun.addBreak();

            // ایجاد جدول
            XWPFTable table = document.createTable();
            table.setWidth("100%");
            table.setInsideHBorder(XWPFTable.XWPFBorderType.SINGLE, 2, 0, "000000");
            table.setInsideVBorder(XWPFTable.XWPFBorderType.SINGLE, 2, 0, "000000");

            // ایجاد سطر عنوان‌ها
            XWPFTableRow headerRow = table.getRow(0);
            String[] headers = {"متغیر", "میانگین", "بیشینه", "کمینه", "تعداد مشاهده", "انحراف معیار", "واریانس", "چولگی", "کشیدگی"};
            for (int i = 0; i < headers.length; i++) {
                if (i == 0) {
                    headerRow.getCell(0).setText(headers[0]);
                } else {
                    headerRow.addNewTableCell().setText(headers[i]);
                }
            }

            // تنظیم استایل سطر عنوان
            for (XWPFTableCell cell : headerRow.getTableCells()) {
                cell.setVerticalAlignment(XWPFTableCell.XWPFVertAlign.CENTER);
                XWPFParagraph para = cell.getParagraphs().get(0);
                para.setAlignment(ParagraphAlignment.CENTER);
                XWPFRun run = para.createRun();
                run.setBold(true);
                run.setFontSize(14);
                run.setFontFamily("B Nazanin");

            }

            // افزودن داده‌ها به جدول
            for (Map<String, Object> stats : results) {
                XWPFTableRow row = table.createRow();

                // استفاده از نام ستون فارسی
                String columnName = stats.getOrDefault("نام ستون", "نامشخص").toString();
                row.getCell(0).setText(columnName);

                row.getCell(1).setText(stats.getOrDefault("تعداد مشاهده", "0").toString());
                row.getCell(2).setText(df.format(Double.parseDouble(stats.getOrDefault("بیشینه", "0").toString())));
                row.getCell(3).setText(df.format(Double.parseDouble(stats.getOrDefault("کمینه", "0").toString())));
                row.getCell(4).setText(df.format(Double.parseDouble(stats.getOrDefault("میانگین", "0").toString())));
                row.getCell(5).setText(df.format(Double.parseDouble(stats.getOrDefault("انحراف معیار", "0").toString())));
                row.getCell(6).setText(df.format(Double.parseDouble(stats.getOrDefault("واریانس", "0").toString())));
                row.getCell(7).setText(df.format(Double.parseDouble(stats.getOrDefault("چولگی", "0").toString())));
                row.getCell(8).setText(df.format(Double.parseDouble(stats.getOrDefault("کشیدگی", "0").toString())));

                // تنظیم چینش داده‌ها
                for (XWPFTableCell cell : row.getTableCells()) {
                    cell.setVerticalAlignment(XWPFTableCell.XWPFVertAlign.CENTER);
                    XWPFParagraph para = cell.getParagraphs().get(0);
                    para.setAlignment(ParagraphAlignment.CENTER);
                    XWPFRun run = para.createRun();
                    run.setFontSize(12);
                    run.setFontFamily("B Nazanin");

                }
            }

            // ذخیره فایل
            document.write(out);
            System.out.println("✅ گزارش Word ایجاد شد: " + fileName);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}
