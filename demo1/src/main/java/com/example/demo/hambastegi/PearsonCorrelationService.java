package com.example.demo.hambastegi;

import com.example.demo.repository.lendingclub.LendingClubRepository;
import lombok.RequiredArgsConstructor;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.poi.xwpf.usermodel.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.*;

@Service
@RequiredArgsConstructor
public class PearsonCorrelationService {

    @Autowired
    private LendingClubRepository lendingClubRepository;

    static List<String> selectedColumns = new ArrayList<>();

    static {
        selectedColumns.add("annual_inc");
        selectedColumns.add("loan_amnt");
    }

//    static {
//        columnPairsConfig.put("annual_inc", "loan_amnt"); // درآمد سالانه و مبلغ وام
//        columnPairs.put("int_rate", "dti");         // نرخ بهره و نسبت بدهی به درآمد
//        columnPairs.put("installment", "revol_bal"); // قسط ماهانه و مانده اعتباری
//        columnPairs.put("open_acc", "total_acc");   // حساب‌های فعال و تعداد کل حساب‌ها
//        columnPairs.put("pub_rec", "delinq_2yrs");  // سوابق عمومی و تأخیر بیش از ۲ سال
//    }

    public List<ColumnPair> findMultipleColumnPairs() {
        List<ColumnPair> columnPairsAndPearsonCorrelation = new ArrayList<>();

        for (int i = 0; i < selectedColumns.size(); i++) {
            for (int j = i + 1; j < selectedColumns.size(); j++) {
                String column1 = selectedColumns.get(i);
                String column2 = selectedColumns.get(j);

                ColumnPair pair = lendingClubRepository.findTwoColumns(column1, column2);
                pair.setPearsonCorrelation(calculatePearsonCorrelation(pair));
                pair.setColumn1Name(column1);
                pair.setColumn2Name(column2);
                columnPairsAndPearsonCorrelation.add(pair);
            }
        }

        return columnPairsAndPearsonCorrelation;
    }

    private double calculatePearsonCorrelation(ColumnPair columnPair) {
        double[] array1 = columnPair.getColumn1Values().stream().mapToDouble(Double::doubleValue).toArray();
        double[] array2 = columnPair.getColumn2Values().stream().mapToDouble(Double::doubleValue).toArray();
        PearsonsCorrelation correlation = new PearsonsCorrelation();
        return correlation.correlation(array1, array2);
    }

    public void generateCorrelationMatrixReport(List<ColumnPair> columnPairs) {
        try (XWPFDocument document = new XWPFDocument()) {
            // اضافه کردن عنوان
            XWPFParagraph title = document.createParagraph();
            title.setAlignment(ParagraphAlignment.CENTER);
            XWPFRun titleRun = title.createRun();
            titleRun.setText("ماتریس همبستگی بین متغیرهای مالی");
            titleRun.setBold(true);
            titleRun.setFontSize(14);
            titleRun.addBreak();

            // ایجاد جدول در فایل Word
            int size = selectedColumns.size() + 1;
            XWPFTable table = document.createTable(size, size);
            table.setWidth("100%");

            // اضافه کردن عنوان ستون‌ها و سطرها
            table.getRow(0).getCell(0).setText("متغیرها");
            for (int i = 0; i < selectedColumns.size(); i++) {
                String variableName = selectedColumns.get(i);
                table.getRow(0).getCell(i + 1).setText(variableName);
                table.getRow(i + 1).getCell(0).setText(variableName);
            }

            // اضافه کردن مقادیر همبستگی پیرسون به جدول
            for (int i = 0; i < selectedColumns.size(); i++) {
                for (int j = 0; j < selectedColumns.size(); j++) {
                    double correlation;
                    if (i == j) {
                        correlation = 1.0; // همبستگی یکسان برای متغیرهای مشابه
                    } else {
                        correlation = findCorrelation(columnPairs, selectedColumns.get(i), selectedColumns.get(j));
                    }
                    table.getRow(i + 1).getCell(j + 1).setText(String.format("%.2f", correlation));
                }
            }

            // ذخیره فایل Word
            String time = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss").format(new Date());
            String fileName = "correlation_matrix_" + time + ".docx";
            try (FileOutputStream out = new FileOutputStream(fileName)) {
                document.write(out);
            }

            System.out.println("✅ گزارش ماتریس همبستگی ایجاد شد: " + fileName);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private double findCorrelation(List<ColumnPair> columnPairs, String col1, String col2) {
        return columnPairs.stream()
                .filter(pair -> (pair.getColumn1Name().equals(col1) && pair.getColumn2Name().equals(col2)) ||
                        (pair.getColumn1Name().equals(col2) && pair.getColumn2Name().equals(col1)))
                .map(ColumnPair::getPearsonCorrelation)
                .findFirst()
                .orElse(0.0);
    }

    public void calcAndCreateReportOfPearsonsCorrelation() {
        List<ColumnPair> multipleColumnPairs = findMultipleColumnPairs();
        generateCorrelationMatrixReport(multipleColumnPairs);
    }
}
