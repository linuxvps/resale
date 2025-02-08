package com.example.demo.hambastegi;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;

import java.util.List;

@Data

public class OutlierAnalysisResult {
    private String columnName;
    private List<Double> valuesBefore;
    private List<Double> valuesAfter;
    private double mean;
    private double standardDeviation;
    private double skewness;
    private double kurtosis;
    private double lowerBound;
    private double upperBound;
    private int outliersRemoved;

    public OutlierAnalysisResult(String columnName, List<Double> valuesBefore, List<Double> valuesAfter, double mean, double standardDeviation, double skewness, double kurtosis, double lowerBound, double upperBound, int outliersRemoved) {
        this.columnName = columnName;
        this.valuesBefore = valuesBefore;
        this.valuesAfter = valuesAfter;
        this.mean = mean;
        this.standardDeviation = standardDeviation;
        this.skewness = skewness;
        this.kurtosis = kurtosis;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.outliersRemoved = outliersRemoved;
    }
}
