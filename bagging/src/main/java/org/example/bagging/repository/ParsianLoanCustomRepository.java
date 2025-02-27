package org.example.bagging.repository;

import weka.core.Instances;

import java.util.List;

public interface ParsianLoanCustomRepository {

    List<?> findDistinct(String column, String condition);

    Instances createInstance();
}
