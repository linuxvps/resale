package com.example.demo.bagging.service;

import com.example.demo.repository.parsian.ParsianLoanRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SelectedTag;

import java.util.Random;

@Service
public class BaggingService {
    @Autowired
    private ParsianLoanRepository parsianLoanRepository;

    public void calcBagging() throws Exception {
        Instances data = parsianLoanRepository.createInstance();
// بررسی میانگین، انحراف معیار، حداقل و حداکثر مقادیر
        System.out.println(data.toSummaryString());


        Instances[] splitData = splitDataset(data, 0.7);
        Instances trainData = splitData[0];
        Instances testData = splitData[1];

        Classifier[] baseModels = getBaseModels();

        Vote votingModel = createVotingModel(baseModels);

        Bagging baggingModel = trainBaggingModel(votingModel, trainData);

        evaluateModel(baggingModel, trainData, testData);

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
