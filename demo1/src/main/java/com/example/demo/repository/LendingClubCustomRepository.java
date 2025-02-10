package com.example.demo.repository;

import com.example.demo.LendingClub;
import com.example.demo.hambastegi.ColumnPair;
import weka.core.Instances;

import java.util.List;

public interface LendingClubCustomRepository {
    List<Double> findColumnByCondition(String columnName, String condition);

    ColumnPair findTwoColumns(String column1, String column2);


    List<Object[]> findRecordsWhereFieldIsNotNull(List<String> fieldNames);
    Instances loadDatasetFromLoan();


}