package com.example.demo.bagging;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.core.*;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.*;

public class LoanRiskVotingModel {
    public static void main(String[] args) throws Exception {
        // 1. خواندن داده‌های وام (Loan Dataset)
        Instances data = loadDataset();

        // 2. تقسیم داده به آموزش (70%) و آزمون (30%)
        data.randomize(new Random(42));
        int trainSize = (int) Math.round(data.numInstances() * 0.7);
        int testSize = data.numInstances() - trainSize;
        Instances trainData = new Instances(data, 0, trainSize);
        Instances testData = new Instances(data, trainSize, testSize);

        // 3. ایجاد مدل‌های پایه (درخت تصمیم، SVM، رگرسیون لجستیک)
        J48 decisionTree = new J48();
        decisionTree.setOptions(new String[]{"-M", "3"}); // حداکثر عمق 3

        SMO svm = new SMO();
        Logistic logistic = new Logistic();

        // 4. ایجاد مدل Voting بدون Bagging
        Classifier[] baseModels = {decisionTree, svm, logistic};
        Vote votingModel = createVotingModel(baseModels);
        votingModel.buildClassifier(trainData); // آموزش مدل

        // 5. ارزیابی مدل روی داده آزمون
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(votingModel, testData);

        // 6. چاپ نتایج
        System.out.println("=== دقت مدل Voting ===");
        System.out.printf("دقت: %.2f%%\n", eval.pctCorrect());

        System.out.println("\n=== گزارش دسته‌بندی ===");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }

    // تابع ایجاد مدل Voting (بدون Bagging)
    private static Vote createVotingModel(Classifier[] baseModels) {
        Vote votingModel = new Vote();
        votingModel.setClassifiers(baseModels);
        votingModel.setCombinationRule(new weka.core.SelectedTag(Vote.MAJORITY_VOTING_RULE, Vote.TAGS_RULES));

        return votingModel;
    }

    // متد برای خواندن loan.data از منابع و تبدیل آن به Instances
    private static Instances loadDataset() {
        return loadIrisDataset();
    }

    private static Instances loadIrisDataset() {
        String fileName = "static/iris.data";
        List<double[]> features = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        List<String> classNames = Arrays.asList("Iris-setosa", "Iris-versicolor", "Iris-virginica");

        try (InputStream inputStream = Optional.ofNullable(
                        BaggingWithVotingExample.class.getClassLoader().getResourceAsStream(fileName))
                .orElseThrow(() -> new IllegalArgumentException("فایل داده یافت نشد: " + fileName));
             BufferedReader br = new BufferedReader(new InputStreamReader(inputStream))) {

            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length == 5) {
                    features.add(Arrays.stream(parts, 0, 4).mapToDouble(Double::parseDouble).toArray());
                    labels.add(parts[4]);
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("خطا در خواندن فایل داده", e);
        }

        // تعریف ویژگی‌ها
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            attributes.add(new Attribute("feature" + (i + 1)));
        }

        // تعریف کلاس‌ها با استفاده از ArrayList
        attributes.add(new Attribute("class", new ArrayList<>(classNames)));

        // ایجاد مجموعه داده
        Instances dataset = new Instances("IrisDataset", attributes, features.size());
        dataset.setClassIndex(4);

        // اضافه کردن داده‌ها
        for (int i = 0; i < features.size(); i++) {
            double[] instanceValues = Arrays.copyOf(features.get(i), 5);
            instanceValues[4] = classNames.indexOf(labels.get(i));
            dataset.add(new DenseInstance(1.0, instanceValues));
        }

        return dataset;
    }}
