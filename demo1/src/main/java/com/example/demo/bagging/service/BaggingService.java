package com.example.demo.bagging.service;

import com.example.demo.repository.parsian.ParsianLoanRepository;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.Filter;

import java.util.*;

@Service
public class BaggingService {
    @Autowired
    private ParsianLoanRepository parsianLoanRepository;

    public void calcBagging() throws Exception {
        Instances data = parsianLoanRepository.createInstance();

        // تحلیل آماری
        analyzeStatistics(data);
        removeUninformativeAttributes(data);

        // // حذف داده‌های پرت بر اساس IQR
        data = removeOutliersUsingIQR(data);


        // استانداردسازی داده‌ها ----- بهتر نشد وقتی گذاشتمیش
        data = preprocessData(data);

        Instances[] splitData = splitDataset(data, 0.7);
        Instances trainData = splitData[0];
        Instances testData = splitData[1];

        Classifier[] baseModels = getBaseModels();

        Vote votingModel = createVotingModel(baseModels);

        Bagging baggingModel = trainBaggingModel(votingModel, trainData);

        evaluateModel(baggingModel, trainData, testData);

    }

    private Map<Double, Integer> getClassDistribution(Instances data) {
        Map<Double, Integer> classCounts = new HashMap<>();
        int classIndex = data.classIndex();

        for (Instance instance : data) {
            double classValue = instance.classValue();
            classCounts.put(classValue, classCounts.getOrDefault(classValue, 0) + 1);
        }

        // نمایش اطلاعات کلاس‌ها
        System.out.println("\n=== Class Balance ===");
        for (Map.Entry<Double, Integer> entry : classCounts.entrySet()) {
            System.out.printf("Class: %.0f | Count: %d\n", entry.getKey(), entry.getValue());
        }

        return classCounts;
    }

    private Instances removeUninformativeAttributes(Instances data) {
        // جذف Attribute برخی متغیرها مقدارهای یکتا بسیار کم دارند
        for (int i = data.numAttributes() - 1; i >= 0; i--) {
            AttributeStats stats = data.attributeStats(i);
            if (stats.distinctCount == 1) { // اگر فقط یک مقدار یکتا دارد، حذف کن
                System.out.println("Removing uninformative attribute: " + data.attribute(i).name());
                data.deleteAttributeAt(i);
            }
        }
        return data;
    }

    private Instances preprocessData(Instances data) throws Exception {
        // استانداردسازی متغیرهای عددی
        Standardize standardize = new Standardize();
        standardize.setInputFormat(data);
        data = Filter.useFilter(data, standardize);
        return data;
    }

    private Instances removeOutliersUsingIQR(Instances data) {
        Instances filteredData = new Instances(data); // کپی از داده‌ها

        for (int i = 0; i < filteredData.numAttributes(); i++) {
            if (filteredData.attribute(i).isNumeric()) {
                List<Double> values = new ArrayList<>();

                // جمع‌آوری مقادیر عددی برای محاسبه چارک‌ها
                for (int j = 0; j < filteredData.numInstances(); j++) {
                    values.add(filteredData.instance(j).value(i));
                }

                // تبدیل لیست به آرایه برای استفاده در Apache Commons Math
                double[] valuesArray = values.stream().mapToDouble(Double::doubleValue).toArray();

                // محاسبه چارک اول (Q1) و چارک سوم (Q3) با Percentile
                Percentile percentile = new Percentile();
                double Q1 = percentile.evaluate(valuesArray, 25);
                double Q3 = percentile.evaluate(valuesArray, 75);
                double IQR = Q3 - Q1;

                // محاسبه محدوده مجاز
                double lowerBound = Q1 - 1.5 * IQR;
                double upperBound = Q3 + 1.5 * IQR;

                List<Integer> outlierIndices = new ArrayList<>();

                // شناسایی داده‌های پرت
                for (int j = 0; j < filteredData.numInstances(); j++) {
                    double value = filteredData.instance(j).value(i);
                    if (value < lowerBound || value > upperBound) {
                        outlierIndices.add(j);
                    }
                }

                // مرتب‌سازی ایندکس‌ها به‌صورت نزولی برای جلوگیری از مشکل IndexOutOfBoundsException
                outlierIndices.sort(Collections.reverseOrder());

                // حذف داده‌های پرت
                for (int index : outlierIndices) {
                    filteredData.delete(index);
                }

                // نمایش اطلاعات آماری پس از حذف داده‌های پرت
                System.out.printf("Attribute: %s | Q1: %.2f | Q3: %.2f | IQR: %.2f | Lower Bound: %.2f | Upper Bound: %.2f | Outliers Removed: %d\n",
                        filteredData.attribute(i).name(), Q1, Q3, IQR, lowerBound, upperBound, outlierIndices.size());
            }
        }

        return filteredData;
    }



    private void analyzeStatistics(Instances data) {
        System.out.println("\n=== Descriptive Statistics ===");
        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attribute(i).isNumeric()) {
                AttributeStats stats = data.attributeStats(i);
                double mean = data.meanOrMode(i);
                double stdDev = stats.numericStats.stdDev;
                double variance = Math.pow(stdDev, 2);
                double min = stats.numericStats.min;
                double max = stats.numericStats.max;
                double range = max - min;  // محاسبه دامنه
                int count = stats.totalCount;
                int missingValues = stats.missingCount;  // تعداد مقادیر گمشده
                int distinctValues = stats.distinctCount;  // تعداد مقادیر یکتا

                System.out.printf("Variable: %s | Count: %d | Mean: %.2f | StdDev: %.2f | Variance: %.2f | Min: %.2f | Max: %.2f | Range: %.2f | Missing Values: %d | Distinct Values: %d %n%s%n",
                        data.attribute(i).name(), count, mean, stdDev, variance, min, max, range, missingValues, distinctValues, "-----------------");

            }
        }
    }


    private Vote createVotingModel(Classifier[] baseModels) {
        Vote votingModel = new Vote();
        votingModel.setClassifiers(baseModels);
//        SelectedTag votingRule = new SelectedTag(Vote.MAJORITY_VOTING_RULE, Vote.TAGS_RULES);
        SelectedTag votingRule = new SelectedTag(Vote.AVERAGE_RULE, Vote.TAGS_RULES);
        votingModel.setCombinationRule(votingRule);

        return votingModel;
    }

    private Classifier[] getBaseModels() throws Exception {
        J48 decisionTree = new J48();
        decisionTree.setOptions(new String[]{"-M", "3"}); // حداکثر عمق 3

        SMO svm = new SMO();
        Logistic logistic = new Logistic();

        return new Classifier[]{decisionTree, svm, logistic};
    }

    /**
     * متد برای تقسیم داده‌ها به آموزش و تست
     */
    private Instances[] splitDataset(Instances data, double trainRatio) {
        data.randomize(new Random(42));
        int trainSize = (int) Math.round(data.numInstances() * trainRatio);
        int testSize = data.numInstances() - trainSize;

        Instances trainData = new Instances(data, 0, trainSize);
        Instances testData = new Instances(data, trainSize, testSize);

        return new Instances[]{trainData, testData};
    }


    /**
     * متد برای ایجاد و آموزش مدل Bagging
     */
    private Bagging trainBaggingModel(Vote votingModel, Instances trainData) throws Exception {
        Bagging baggingModel = new Bagging();
        baggingModel.setClassifier(votingModel);
        baggingModel.setNumIterations(10);
        baggingModel.setBagSizePercent(80);

        baggingModel.buildClassifier(trainData);
        return baggingModel;
    }

    /**
     * متد برای ارزیابی مدل
     */
    private void evaluateModel(Classifier model, Instances trainData, Instances testData) throws Exception {
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(model, testData);

        System.out.println("=== Accuracy ===");
        System.out.printf("Accuracy: %.2f%%\n", eval.pctCorrect());

        System.out.println("\n=== Classification Report ===");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }

}
