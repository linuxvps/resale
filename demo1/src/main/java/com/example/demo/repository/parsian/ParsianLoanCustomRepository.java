package com.example.demo.repository.parsian;

import com.example.demo.hambastegi.ColumnPair;
import weka.core.Instances;

import java.util.List;

public interface ParsianLoanCustomRepository {

    List<?> findDistinct(String column);

    public Instances createInstance();
}
