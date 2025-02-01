package com.example.demo;

import java.util.List;

public interface LendingClubCustomRepository {
    List<Double> findColumnByCondition(String columnName, String condition);
}