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
        stats.put("columnName", COLUMN_NAME_TRANSLATIONS.get(columnName));
        return stats;
    }

    public List<Map<String, Object>> calcStatForList() {
        List<Map<String, Object>> results = new ArrayList<>();

        for (String columnName : COLUMN_NAME_TRANSLATIONS.keySet()) {
            Map<String, Object> stats = getColumnStatistics(columnName, "1=1");
            stats.put("نام ستون", COLUMN_NAME_TRANSLATIONS.get(columnName));
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

        Map<String, Object> result = new HashMap<>();
        result.put("بیشینه", stats.getMax());
        result.put("کمینه", stats.getMin());
        result.put("تعداد مشاهده", stats.getN());
        result.put("میانگین", stats.getMean());
        result.put("انحراف معیار", stats.getStandardDeviation());
        result.put("واریانس", stats.getVariance());
        result.put("چولگی", stats.getSkewness());
        result.put("کشیدگی", stats.getKurtosis());

        return result;
    }

    public void createWordReport(List<Map<String, Object>> results, String fileName) {
        try (XWPFDocument document = new XWPFDocument(); FileOutputStream out = new FileOutputStream(fileName)) {
            DecimalFormat df = new DecimalFormat("#,##0.##"); // افزودن جداکننده هزارگان و حذف نمایش علمی

            XWPFParagraph title = document.createParagraph();
            title.setAlignment(ParagraphAlignment.CENTER);
            XWPFRun titleRun = title.createRun();
            titleRun.setText("گزارش آمار ریسک اعتباری");
            titleRun.setBold(true);
            titleRun.setFontSize(18);
            titleRun.setFontFamily("B Nazanin");
            titleRun.addBreak();

            XWPFTable table = document.createTable();
            table.setWidth("100%");
            table.setInsideHBorder(XWPFTable.XWPFBorderType.SINGLE, 2, 0, "000000");
            table.setInsideVBorder(XWPFTable.XWPFBorderType.SINGLE, 2, 0, "000000");

            XWPFTableRow headerRow = table.getRow(0);
            String[] headers = {"متغیر", "میانگین", "بیشینه", "کمینه", "تعداد مشاهده", "انحراف معیار", "واریانس", "چولگی", "کشیدگی"};
            for (int i = 0; i < headers.length; i++) {
                if (i == 0) {
                    headerRow.getCell(0).setText(headers[0]);
                } else {
                    headerRow.addNewTableCell().setText(headers[i]);
                }
            }

            for (XWPFTableCell cell : headerRow.getTableCells()) {
                cell.setVerticalAlignment(XWPFTableCell.XWPFVertAlign.CENTER);
                XWPFParagraph para = cell.getParagraphs().get(0);
                para.setAlignment(ParagraphAlignment.CENTER);
                XWPFRun run = para.createRun();
                run.setBold(true);
                run.setFontSize(14);
                run.setFontFamily("B Nazanin");
            }

            for (Map<String, Object> stats : results) {
                XWPFTableRow row = table.createRow();

                String columnName = stats.getOrDefault("نام ستون", "نامشخص").toString();
                row.getCell(0).setText(columnName);

                row.getCell(1).setText(  formatIfLarge( df.format(stats.getOrDefault("میانگین", 0))));
                row.getCell(2).setText(  formatIfLarge( df.format(stats.getOrDefault("بیشینه", 0))));
                row.getCell(3).setText(  formatIfLarge( df.format(stats.getOrDefault("کمینه", 0))));
                row.getCell(4).setText(  formatIfLarge( stats.getOrDefault("تعداد مشاهده", 0).toString()));
                row.getCell(5).setText(  formatIfLarge( df.format(stats.getOrDefault("انحراف معیار", 0))));
                row.getCell(6).setText(  formatIfLarge( df.format(stats.getOrDefault("واریانس", 0))));
                row.getCell(7).setText(  formatIfLarge( df.format(stats.getOrDefault("چولگی", 0))));
                row.getCell(8).setText(  formatIfLarge( df.format(stats.getOrDefault("کشیدگی", 0))));

                for (XWPFTableCell cell : row.getTableCells()) {
                    cell.setVerticalAlignment(XWPFTableCell.XWPFVertAlign.CENTER);
                    XWPFParagraph para = cell.getParagraphs().get(0);
                    para.setAlignment(ParagraphAlignment.CENTER);
                    XWPFRun run = para.createRun();
                    run.setFontSize(12);
                    run.setFontFamily("B Nazanin");
                }
            }

            document.write(out);
            System.out.println("✅ گزارش Word ایجاد شد: " + fileName);

        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    public static String formatIfLarge(String numberStr) {
        final int MAX_DIGITS = 6; // حداکثر تعداد ارقام مجاز

        try {
            numberStr = numberStr.replace(",", "");

            // تبدیل رشته به عدد اعشاری
            double number = Double.parseDouble(numberStr);

            // جدا کردن قسمت صحیح عدد
            String integerPart = numberStr.contains(".") ? numberStr.substring(0, numberStr.indexOf(".")) : numberStr;

            // حذف علامت منفی در صورت وجود
            integerPart = integerPart.replace("-", "");

            // بررسی تعداد ارقام قسمت صحیح
            if (integerPart.length() > MAX_DIGITS) {
                // فرمت علمی برای اعداد بزرگ
                DecimalFormat scientificFormat = new DecimalFormat("0.##E0");
                return scientificFormat.format(number);
            } else {
                return numberStr;
            }
        } catch (NumberFormatException e) {
            return "Invalid number format";
        }
    }


}
