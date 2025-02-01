package com.example.demo;

import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import jakarta.persistence.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

@Repository
public class LendingClubCustomRepositoryImpl implements LendingClubCustomRepository {

    @PersistenceContext
    private EntityManager entityManager;

    @Override
    public List<Double> findColumnByCondition(String columnName, String condition) {

        String sql = "SELECT " + columnName + " FROM loan WHERE " + condition;
        Query query = entityManager.createNativeQuery(sql);

        // تبدیل لیست خروجی به List<Double>
        List<?> resultList = query.getResultList();
        return resultList.stream()
                .filter(Objects::nonNull) // حذف مقدارهای null
                .map(result -> ((Number) result).doubleValue()) // تبدیل Object به Double
                .collect(Collectors.toList());

    }
}
