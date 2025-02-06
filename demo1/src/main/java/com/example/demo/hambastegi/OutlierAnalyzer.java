package com.example.demo.hambastegi;

import com.example.demo.LendingClub;
import com.example.demo.repository.LendingClubRepository;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.data.statistics.HistogramDataset;
import org.jfree.data.statistics.HistogramType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import jakarta.transaction.Transactional;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class OutlierAnalyzer {

    @Autowired
    private LendingClubRepository lendingClubRepository;

    @Transactional
    public void analyzeAndSoftDeleteOutliers(String columnName) {
        List<Object[]> recordsWhereFieldIsNotNull = lendingClubRepository.findRecordsWhereFieldIsNotNull(Collections.singletonList(columnName));

        if (recordsWhereFieldIsNotNull == null || recordsWhereFieldIsNotNull.isEmpty()) {
            throw new IllegalArgumentException("داده‌ای برای این ستون یافت نشد: " + columnName);
        }

        List<Double> values = recordsWhereFieldIsNotNull.stream()
                .map(record -> (record[1] instanceof Number) ? ((Number) record[1]).doubleValue() : null)
                .filter(value -> value != null)
                .collect(Collectors.toList());

        if (values.isEmpty()) {
            System.out.println("⚠️ هیچ مقدار معتبری برای پردازش در ستون " + columnName + " یافت نشد.");
            return;
        }

        DescriptiveStatistics stats = new DescriptiveStatistics();
        values.forEach(stats::addValue);

        double q1 = stats.getPercentile(25);
        double q3 = stats.getPercentile(75);
        double iqr = q3 - q1;

        double lowerBound = q1 - 1.5 * iqr;
        double upperBound = q3 + 1.5 * iqr;

        List<Long> outlierIds = recordsWhereFieldIsNotNull.stream()
                .filter(record -> {
                    Double value = (record[1] instanceof Number) ? ((Number) record[1]).doubleValue() : null;
                    return value != null && (value < lowerBound || value > upperBound);
                })
                .map(record -> ((Number) record[0]).longValue())
                .collect(Collectors.toList());

        System.out.println("✅ ستون: " + columnName);
        System.out.println("📊 میانگین: " + stats.getMean());
        System.out.println("📊 انحراف معیار: " + stats.getStandardDeviation());
        System.out.println("📊 چولگی: " + stats.getSkewness());
        System.out.println("📊 کشیدگی: " + stats.getKurtosis());
        System.out.println("🔍 محدوده مجاز داده‌ها: [" + lowerBound + " , " + upperBound + "]");
        System.out.println("📉 تعداد داده‌های قبل از پردازش: " + recordsWhereFieldIsNotNull.size());
        System.out.println("❌ تعداد داده‌های پرت که باید حذف شوند: " + outlierIds.size());

        lendingClubRepository.updateIsDeletedForIds(outlierIds);

        System.out.println("✅ حذف منطقی داده‌های پرت در ستون " + columnName + " انجام شد.");

        plotHistogram(values, columnName + " (قبل از حذف داده‌های پرت)");

        List<Double> filteredValues = values.stream()
                .filter(value -> value >= lowerBound && value <= upperBound)
                .collect(Collectors.toList());
        plotHistogram(filteredValues, columnName + " (بعد از حذف داده‌های پرت)");
    }
    public void plotHistogram(List<Double> data, String title) {
        double[] dataArray = data.stream().mapToDouble(Double::doubleValue).toArray();

        HistogramDataset dataset = new HistogramDataset();
        dataset.setType(HistogramType.RELATIVE_FREQUENCY);
        dataset.addSeries("Frequency", dataArray, 20);

        JFreeChart histogram = ChartFactory.createHistogram(
                title,
                "مقدار",
                "فراوانی",
                dataset
        );

        try {
            File file = new File(title.replace(" ", "_") + ".png");
            ChartUtils.saveChartAsPNG(file, histogram, 800, 600);
            System.out.println("📊 نمودار ذخیره شد: " + file.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}
