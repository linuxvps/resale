package com.example.demo.repository;

import com.example.demo.hambastegi.ColumnPair;
import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import jakarta.persistence.Query;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
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

    // متد جدید برای دریافت همزمان دو ستون و بازگرداندن آن‌ها در قالب ColumnPair
    @Override
    public ColumnPair findTwoColumns(String column1, String column2) {
        String sql = "SELECT " + column1 + ", " + column2 + " FROM loan WHERE " + column1 + " IS NOT NULL AND " + column2 + " IS NOT NULL";
        Query query = entityManager.createNativeQuery(sql);

        List<?> resultList = query.getResultList();
        List<Double> column1List = new ArrayList<>();
        List<Double> column2List = new ArrayList<>();

        for (Object row : resultList) {
            Object[] columns = (Object[]) row;
            if (columns[0] != null && columns[1] != null) {
                column1List.add(((Number) columns[0]).doubleValue());
                column2List.add(((Number) columns[1]).doubleValue());
            }
        }
        ColumnPair pair = new ColumnPair();
        pair.setColumn1Values(column1List);
        pair.setColumn2Values(column2List);
        return pair;
//        return new ColumnPair(column1List, column2List);
    }



}
