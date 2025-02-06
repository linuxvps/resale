package com.example.demo.hambastegi;


import com.example.demo.repository.LendingClubRepository;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.moment.Kurtosis;
import org.apache.commons.math3.stat.descriptive.moment.Skewness;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;
import org.apache.commons.math3.stat.inference.TestUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;
import java.util.HashMap;

@Service
public class DataDistributionAnalyzer {

    @Autowired
    private LendingClubRepository lendingClubRepository;

    public Map<String, Object> analyzeColumnDistribution(String columnName) {
        // دریافت داده‌های ستون مورد نظر از دیتابیس
        List<Double> values = lendingClubRepository.findColumnByCondition(columnName,columnName + " is not null");

        // بررسی تعداد داده‌ها
        if (values == null || values.isEmpty()) {
            throw new IllegalArgumentException("داده‌ای برای این ستون یافت نشد: " + columnName);
        }

        // تبدیل لیست به آرایه برای تحلیل آماری
        double[] dataArray = values.stream().mapToDouble(Double::doubleValue).toArray();

        // محاسبه شاخص‌های آماری
        DescriptiveStatistics stats = new DescriptiveStatistics(dataArray);
        double mean = stats.getMean();
        double stdDev = stats.getStandardDeviation();
        double skewness = new Skewness().evaluate(dataArray);
        double kurtosis = new Kurtosis().evaluate(dataArray);
        double min = stats.getMin();
        double max = stats.getMax();

        // تست کولموگروف-اسمیرنوف برای بررسی نرمال بودن داده‌ها
        KolmogorovSmirnovTest ksTest = new KolmogorovSmirnovTest();
        double ksPValue = ksTest.kolmogorovSmirnovTest(StatUtils.normalize(dataArray), dataArray);

        // نتیجه‌گیری: آیا داده‌ها نرمال هستند؟
        boolean isNormal = Math.abs(skewness) < 1.0 && Math.abs(kurtosis - 3) < 1.0 && ksPValue > 0.05;

        // انتخاب روش مناسب برای شناسایی داده‌های پرت
        String recommendedMethod = isNormal ? "Z-Score" : "IQR";

        // ساختن خروجی
        Map<String, Object> results = new HashMap<>();
        results.put("ستون", columnName);
        results.put("میانگین", mean);
        results.put("انحراف معیار", stdDev);
        results.put("چولگی", skewness);
        results.put("کشیدگی", kurtosis);
        results.put("حداقل مقدار", min);
        results.put("حداکثر مقدار", max);
        results.put("نتیجه تست نرمال بودن", isNormal ? "داده‌ها نرمال هستند" : "داده‌ها نرمال نیستند");
        results.put("روش پیشنهادی برای داده‌های پرت", recommendedMethod);

        return results;
    }
}
