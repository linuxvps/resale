package com.example.demo.repository;

import com.example.demo.hambastegi.ColumnPair;

import java.util.List;

public interface LendingClubCustomRepository {
    List<Double> findColumnByCondition(String columnName, String condition);

    ColumnPair findTwoColumns(String column1, String column2);


}