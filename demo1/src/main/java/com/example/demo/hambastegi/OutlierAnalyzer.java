package com.example.demo.hambastegi;

import com.example.demo.repository.LendingClubRepository;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.data.statistics.HistogramDataset;
import org.jfree.data.statistics.HistogramType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class OutlierAnalyzer {

    @Autowired
    private LendingClubRepository lendingClubRepository;

    public void calcOutlierAnalyzer() {
        List<String> selectedColumns = selectColumnsForOutlierDetection();
        List<OutlierAnalysisResult> analysisResults = analyzeOutliers(selectedColumns);
        plotOutlierResults(analysisResults);
    }

    private List<String> selectColumnsForOutlierDetection() {
        List<String> allColumns = List.of("annual_inc", "dti", "loan_amnt", "int_rate", "revol_bal", "open_acc", "total_acc");
        Map<String, Double> varianceMap = new HashMap<>();

        for (String column : allColumns) {
            List<Object[]> records = lendingClubRepository.findRecordsWhereFieldIsNotNull(Collections.singletonList(column));
            List<Double> values = records.stream()
                    .map(record -> (record[1] instanceof Number) ? ((Number) record[1]).doubleValue() : null)
                    .filter(Objects::nonNull)
                    .toList();

            if (values.isEmpty()) continue;

            DescriptiveStatistics stats = new DescriptiveStatistics();
            values.forEach(stats::addValue);
            varianceMap.put(column, stats.getVariance());
        }

        return varianceMap.entrySet().stream()
                .sorted((e1, e2) -> Double.compare(e2.getValue(), e1.getValue()))
                .limit(5)
                .map(Map.Entry::getKey)
                .toList();
    }


    public List<OutlierAnalysisResult> analyzeOutliers(List<String> columnNames) {
        List<OutlierAnalysisResult> results = new ArrayList<>();
        for (String column : columnNames) {
            OutlierAnalysisResult result = analyzeAndSoftDeleteOutliers(column);
            if (result != null) {
                results.add(result);
            }
        }
        return results;
    }

    public OutlierAnalysisResult analyzeAndSoftDeleteOutliers(String columnName) {
        List<Object[]> recordsWhereFieldIsNotNull = lendingClubRepository.findRecordsWhereFieldIsNotNull(Collections.singletonList(columnName));

        if (recordsWhereFieldIsNotNull == null || recordsWhereFieldIsNotNull.isEmpty()) {
            throw new IllegalArgumentException("داده‌ای برای این ستون یافت نشد: " + columnName);
        }

        List<Double> valuesBefore = recordsWhereFieldIsNotNull.stream()
                .map(record -> (record[1] instanceof Number) ? ((Number) record[1]).doubleValue() : null)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());

        if (valuesBefore.isEmpty()) {
            return null;
        }

        DescriptiveStatistics stats = new DescriptiveStatistics();
        valuesBefore.forEach(stats::addValue);

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

        lendingClubRepository.updateIsDeletedForIds(outlierIds);

        List<Double> valuesAfter = valuesBefore.stream()
                .filter(value -> value >= lowerBound && value <= upperBound)
                .collect(Collectors.toList());

        return new OutlierAnalysisResult(columnName, valuesBefore, valuesAfter,
                stats.getMean(), stats.getStandardDeviation(), stats.getSkewness(), stats.getKurtosis(),
                lowerBound, upperBound, outlierIds.size());
    }

    private void plotOutlierResults(List<OutlierAnalysisResult> results) {
        for (OutlierAnalysisResult result : results) {
            plotHistogram(result.getValuesBefore(), result.getColumnName() + " (قبل از حذف داده‌های پرت)");
            plotHistogram(result.getValuesAfter(), result.getColumnName() + " (بعد از حذف داده‌های پرت)");
        }
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
