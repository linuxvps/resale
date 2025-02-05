package com.example.demo.hambastegi;

import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
public class ColumnPair {
    private List<Double> column1Values;
    private String column1Name;
    private List<Double> column2Values;
    private String column2Name;
    private Double pearsonCorrelation;

}