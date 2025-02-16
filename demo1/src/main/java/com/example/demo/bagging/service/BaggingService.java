package com.example.demo.bagging.service;

import com.example.demo.repository.parsian.ParsianLoanRepository;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

import java.text.DecimalFormat;
import java.util.*;

@Service
public class BaggingService {

    private static final Logger logger = LoggerFactory.getLogger(BaggingService.class);
    private static final DecimalFormat df = new DecimalFormat("0.00");

    @Autowired
    private ParsianLoanRepository parsianLoanRepository;

    public void calcBagging() throws Exception {
        // بارگذاری داده از منبع
        Instances rawData = loadData();
        // اجرای خط لوله پیش‌پردازش داده
        Instances preprocessedData = prepareData(rawData);
        // تقسیم داده به آموزش و تست
        Instances[] splits = splitDataset(preprocessedData, 0.7);
        Instances trainData = splits[0];
        Instances testData = splits[1];

        // ایجاد مدل‌های پایه و ترکیب آن‌ها با رای‌گیری
        Classifier[] baseModels = getBaseModels();
        Vote votingModel = createVotingModel(baseModels);
        // ساخت و آموزش مدل Bagging
        Bagging baggingModel = trainBaggingModel(votingModel, trainData);
        // ارزیابی مدل نهایی
        evaluateModel(baggingModel, trainData, testData);
    }

    private Instances loadData() {
        return parsianLoanRepository.createInstance();
    }

    private Instances prepareData(Instances data) throws Exception {
        data = statisticalDataAnalysis(data);
        data = removeHighlyCorrelatedFeatures(data, 0.95);
        data = manageOutliers(data);
        return data;
    }

    private Instances removeHighlyCorrelatedFeatures(Instances data, double threshold) {
        // ساخت یک کپی از داده‌های ورودی
        Instances newData = new Instances(data);
        int numAttributes = newData.numAttributes();
        int classIndex = newData.classIndex();
        Set<Integer> attributesToRemove = new HashSet<>();

        // بررسی تمامی جفت ویژگی‌های عددی
        for (int i = 0; i < numAttributes - 1; i++) {
            for (int j = i + 1; j < numAttributes; j++) {
                if (newData.attribute(i).isNumeric() && newData.attribute(j).isNumeric()) {
                    double correlation = calculatePearsonCorrelation(newData, i, j);
                    if (Math.abs(correlation) > threshold) {
                        // اطمینان از عدم حذف ویژگی کلاس
                        if (j != classIndex) {
                            logger.info("🔴 همبستگی بالا بین: {} و {} | مقدار: {} | حذف: {}",
                                    newData.attribute(i).name(),
                                    newData.attribute(j).name(),
                                    df.format(correlation),
                                    newData.attribute(j).name());
                            attributesToRemove.add(j);
                        }
                    }
                }
            }
        }

        // حذف ویژگی‌ها به ترتیب نزولی برای جلوگیری از تغییر ایندکس‌ها
        List<Integer> sortedAttributes = new ArrayList<>(attributesToRemove);
        sortedAttributes.sort(Collections.reverseOrder());
        for (int index : sortedAttributes) {
            if (index < newData.numAttributes() && index != classIndex) {
                logger.info("✅ ویژگی حذف شد: {}", newData.attribute(index).name());
                newData.deleteAttributeAt(index);
            }
        }
        return newData;
    }


    private double calculatePearsonCorrelation(Instances data, int index1, int index2) {
        double[] array1 = new double[data.numInstances()];
        double[] array2 = new double[data.numInstances()];

        for (int i = 0; i < data.numInstances(); i++) {
            array1[i] = data.instance(i).value(index1);
            array2[i] = data.instance(i).value(index2);
        }

        PearsonsCorrelation correlation = new PearsonsCorrelation();
        return correlation.correlation(array1, array2);
    }

    private Instances statisticalDataAnalysis(Instances data) throws Exception {
        // تحلیل اولیه آماری برای آگاهی از وضعیت داده‌ها
        analyzeStatistics(data);
        // حذف ویژگی‌هایی که تنها یک مقدار یکتا دارند
        removeUninformativeAttributes(data);
        // رفع یا حذف مقادیر گمشده
        data = handleMissingValues(data);
        // حذف ویژگی‌های کم‌واریانس یا ثابت
        data = removeLowVarianceAttributes(data);
        // اعمال استانداردسازی در صورت نیاز
        data = standardizeData(data);

        return data;
    }

    private Instances manageOutliers(Instances data) {
        // حذف داده‌های پرت برای هر ویژگی عددی
        data = removeOutliersUsingIQR(data);
        return data;
    }

    private void analyzeStatistics(Instances data) {
        logger.info("=== آمار توصیفی ===");
        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attribute(i).isNumeric()) {
                AttributeStats stats = data.attributeStats(i);
                double mean = data.meanOrMode(i);
                double stdDev = stats.numericStats.stdDev;
                double variance = Math.pow(stdDev, 2);
                double min = stats.numericStats.min;
                double max = stats.numericStats.max;
                double range = max - min;
                int count = stats.totalCount;
                int missing = stats.missingCount;
                int distinct = stats.distinctCount;

                logger.info("متغیر: {} | تعداد: {} | میانگین: {} | انحراف معیار: {} | واریانس: {} | حداقل: {} | حداکثر: {} | دامنه: {} | گمشده: {} | یکتا: {}\n-----------------",
                        data.attribute(i).name(),
                        count,
                        df.format(mean),
                        df.format(stdDev),
                        df.format(variance),
                        df.format(min),
                        df.format(max),
                        df.format(range),
                        missing,
                        distinct);
            }
        }
    }

    private void removeUninformativeAttributes(Instances data) {
        // حذف ویژگی‌هایی که تنها یک مقدار یکتا دارند
        for (int i = data.numAttributes() - 1; i >= 0; i--) {
            AttributeStats stats = data.attributeStats(i);
            if (stats.distinctCount == 1) {
                logger.info("حذف ویژگی بی‌اطلاعی: {}", data.attribute(i).name());
                data.deleteAttributeAt(i);
            }
        }
    }

    private Instances handleMissingValues(Instances data) {
        logMissingValues(data);
        Instances modifiedData = new Instances(data);
        for (int i = 0; i < modifiedData.numAttributes(); i++) {
            AttributeStats stats = modifiedData.attributeStats(i);
            int missingCount = stats.missingCount;
            double missingPerc = (double) missingCount / stats.totalCount * 100;
            if (missingCount > 0) {
                if (missingPerc < 5) {
                    fillMissingWithMean(modifiedData, i);
                } else if (missingPerc < 20) {
                    fillMissingWithMedian(modifiedData, i);
                } else {
                    logger.info("❌ ویژگی: {} | حذف به دلیل {}% مقادیر گمشده",
                            modifiedData.attribute(i).name(), df.format(missingPerc));
                    modifiedData.deleteAttributeAt(i);
                }
            }
        }
        return modifiedData;
    }

    private void logMissingValues(Instances data) {
        logger.info("=== تحلیل مقادیر گمشده ===");
        for (int i = 0; i < data.numAttributes(); i++) {
            AttributeStats stats = data.attributeStats(i);
            int missing = stats.missingCount;
            double perc = (double) missing / stats.totalCount * 100;
            if (missing > 0) {
                logger.info("ویژگی: {} | گمشده: {} ({}%)", data.attribute(i).name(), missing, df.format(perc));
            }
        }
    }

    private void fillMissingWithMean(Instances data, int attributeIndex) {
        double mean = data.meanOrMode(attributeIndex);
        for (int j = 0; j < data.numInstances(); j++) {
            if (data.instance(j).isMissing(attributeIndex)) {
                data.instance(j).setValue(attributeIndex, mean);
            }
        }
        logger.info("✅ ویژگی: {} | جایگزینی مقادیر گمشده با میانگین: {}",
                data.attribute(attributeIndex).name(), df.format(mean));
    }

    private void fillMissingWithMedian(Instances data, int attributeIndex) {
        double median = calculateMedian(data, attributeIndex);
        for (int j = 0; j < data.numInstances(); j++) {
            if (data.instance(j).isMissing(attributeIndex)) {
                data.instance(j).setValue(attributeIndex, median);
            }
        }
        logger.info("🔄 ویژگی: {} | جایگزینی مقادیر گمشده با میانه: {}",
                data.attribute(attributeIndex).name(), df.format(median));
    }

    private double calculateMedian(Instances data, int attributeIndex) {
        List<Double> values = new ArrayList<>();
        for (int i = 0; i < data.numInstances(); i++) {
            if (!data.instance(i).isMissing(attributeIndex)) {
                values.add(data.instance(i).value(attributeIndex));
            }
        }
        Collections.sort(values);
        int size = values.size();
        int mid = size / 2;
        if (size % 2 == 0) {
            return (values.get(mid - 1) + values.get(mid)) / 2.0;
        } else {
            return values.get(mid);
        }
    }

    private Instances removeLowVarianceAttributes(Instances data) {
        Instances processedData = new Instances(data);
        List<Integer> removeIndices = new ArrayList<>();
        for (int i = 0; i < processedData.numAttributes(); i++) {
            if (processedData.attribute(i).isNumeric()) {
                AttributeStats stats = processedData.attributeStats(i);
                double stdDev = stats.numericStats.stdDev;
                double variance = Math.pow(stdDev, 2);
                if (variance < 0.0001 || stdDev == 0) {
                    removeIndices.add(i);
                    logger.info("⚠️ ویژگی: {} | واریانس: {} | حذف (کم‌واریانس یا ثابت)",
                            processedData.attribute(i).name(), df.format(variance));
                }
            }
        }
        Collections.reverse(removeIndices);
        for (int index : removeIndices) {
            processedData.deleteAttributeAt(index);
        }
        return processedData;
    }

    private Instances standardizeData(Instances data) throws Exception {
        Instances processedData = new Instances(data);
        List<Integer> toStandardize = new ArrayList<>();
        for (int i = 0; i < processedData.numAttributes(); i++) {
            if (processedData.attribute(i).isNumeric()) {
                AttributeStats stats = processedData.attributeStats(i);
                double stdDev = stats.numericStats.stdDev;
                double mean = processedData.meanOrMode(i);
                if (stdDev > 1.5 * mean) {
                    toStandardize.add(i);
                    logger.info("✅ ویژگی: {} | انحراف معیار: {} | میانگین: {} | استانداردسازی اعمال شد",
                            processedData.attribute(i).name(), df.format(stdDev), df.format(mean));
                }
            }
        }
        if (!toStandardize.isEmpty()) {
            Standardize standardize = new Standardize();
            standardize.setInputFormat(processedData);
            processedData = Filter.useFilter(processedData, standardize);
        }
        return processedData;
    }

    private Instances removeOutliersUsingIQR(Instances data) {
        Instances filteredData = new Instances(data);
        for (int i = 0; i < filteredData.numAttributes(); i++) {
            if (filteredData.attribute(i).isNumeric()) {
                removeOutliersForAttribute(filteredData, i);
            }
        }
        return filteredData;
    }

    private int removeOutliersForAttribute(Instances data, int attributeIndex) {
        double[] metrics = calculateIQRMetrics(data, attributeIndex);
        double Q1 = metrics[0];
        double Q3 = metrics[1];
        double IQR = metrics[2];
        double lowerBound = metrics[3];
        double upperBound = metrics[4];
        List<Integer> outlierIndices = new ArrayList<>();
        for (int j = 0; j < data.numInstances(); j++) {
            double value = data.instance(j).value(attributeIndex);
            if (value < lowerBound || value > upperBound) {
                outlierIndices.add(j);
            }
        }
        outlierIndices.sort(Collections.reverseOrder());
        int removedCount = outlierIndices.size();
        for (int idx : outlierIndices) {
            data.delete(idx);
        }
        logger.info("ویژگی: {} | Q1: {} | Q3: {} | IQR: {} | حد پایین: {} | حد بالا: {} | داده‌های پرت حذف شده: {}",
                data.attribute(attributeIndex).name(),
                df.format(Q1),
                df.format(Q3),
                df.format(IQR),
                df.format(lowerBound),
                df.format(upperBound),
                removedCount);
        return removedCount;
    }

    private double[] calculateIQRMetrics(Instances data, int attributeIndex) {
        List<Double> values = new ArrayList<>();
        for (int j = 0; j < data.numInstances(); j++) {
            values.add(data.instance(j).value(attributeIndex));
        }
        double[] vals = values.stream().mapToDouble(Double::doubleValue).toArray();
        Percentile percentile = new Percentile();
        double Q1 = percentile.evaluate(vals, 25);
        double Q3 = percentile.evaluate(vals, 75);
        double IQR = Q3 - Q1;
        double lowerBound = Q1 - 1.5 * IQR;
        double upperBound = Q3 + 1.5 * IQR;
        return new double[]{Q1, Q3, IQR, lowerBound, upperBound};
    }

    private Instances[] splitDataset(Instances data, double trainRatio) {
        data.randomize(new Random(42));
        int trainSize = (int) Math.round(data.numInstances() * trainRatio);
        int testSize = data.numInstances() - trainSize;
        Instances trainData = new Instances(data, 0, trainSize);
        Instances testData = new Instances(data, trainSize, testSize);
        return new Instances[]{trainData, testData};
    }

    private Classifier[] getBaseModels() throws Exception {
        J48 decisionTree = new J48();
        decisionTree.setOptions(new String[]{"-M", "3"});
        SMO svm = new SMO();
        Logistic logistic = new Logistic();
        return new Classifier[]{decisionTree, svm, logistic};
    }

    private Vote createVotingModel(Classifier[] baseModels) {
        Vote votingModel = new Vote();
        votingModel.setClassifiers(baseModels);
        SelectedTag votingRule = new SelectedTag(Vote.AVERAGE_RULE, Vote.TAGS_RULES);
        votingModel.setCombinationRule(votingRule);
        return votingModel;
    }

    private Bagging trainBaggingModel(Vote votingModel, Instances trainData) throws Exception {
        Bagging baggingModel = new Bagging();
        baggingModel.setClassifier(votingModel);
        baggingModel.setNumIterations(10);
        baggingModel.setBagSizePercent(80);
        baggingModel.buildClassifier(trainData);
        return baggingModel;
    }

    private void evaluateModel(Classifier model, Instances trainData, Instances testData) throws Exception {
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(model, testData);

        logger.error("=== دقت مدل ===");
        logger.error("دقت: {}%", df.format(eval.pctCorrect()));
        logger.error("=== گزارش طبقه‌بندی ===");
        logger.error("{}", eval.toSummaryString());
        logger.error("{}", eval.toClassDetailsString());
        logger.error("{}", eval.toMatrixString());
    }
}
