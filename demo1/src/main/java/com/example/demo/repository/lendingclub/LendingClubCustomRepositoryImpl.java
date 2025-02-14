package com.example.demo.repository.lendingclub;

import com.example.demo.hambastegi.ColumnPair;
import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import jakarta.persistence.Query;
import org.springframework.stereotype.Repository;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.*;
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



    @Override
    public List<Object[]> findRecordsWhereFieldIsNotNull(List<String> fieldNames) {
        if (fieldNames == null || fieldNames.isEmpty()) {
            throw new IllegalArgumentException("لیست فیلدها نباید خالی باشد.");
        }


        // ایجاد کوئری داینامیک
        String selectedFields = "len_id , " + String.join(", ", fieldNames);
        String whereClause = fieldNames.stream()
                .map(field -> field + " IS NOT NULL")
                .collect(Collectors.joining(" AND "));

        String queryString = "SELECT " + selectedFields + " FROM loan WHERE " + whereClause;

        Query query = entityManager.createNativeQuery(queryString);
        return query.getResultList();
    }

    @Override
    public Instances loadDatasetFromLoan() {
        // دریافت لیست ستون‌های مهم از جدول loan_features
        String featureSql = "SELECT column_name FROM ln.loan_features WHERE importance_level = 4 AND table_name = 'loan'";
        @SuppressWarnings("unchecked")
        List<String> importantColumns = entityManager.createNativeQuery(featureSql).getResultList();

        if (importantColumns.isEmpty()) {
            throw new RuntimeException("هیچ ستونی با اهمیت 4 برای جدول loan پیدا نشد.");
        }

        // اطمینان از اینکه loan_status داخل لیست نیست
        importantColumns.remove("loan_status");

        // اضافه کردن loan_status در آخر لیست
        importantColumns.add("loan_status");

        String columns = String.join(", ", importantColumns);
        String loanSql = "SELECT " + columns + " FROM ln.loan WHERE is_deleted <> 1 OR is_deleted IS NULL AND len_id < 100";
        Query loanQuery = entityManager.createNativeQuery(loanSql);
        List<?> resultList = loanQuery.getResultList();

        // تعریف ویژگی‌ها (Nominal برای loan_status)
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (String col : importantColumns) {
            if ("loan_status".equals(col)) {
                // ایجاد Attribute Nominal برای ستون loan_status
                List<String> loanStatusValues = getUniqueLoanStatusValues(resultList, importantColumns.indexOf("loan_status"));
                attributes.add(new Attribute("loan_status", loanStatusValues));
            } else {
                attributes.add(new Attribute(col)); // ستون‌های عددی به‌صورت Numeric تعریف می‌شوند
            }
        }

        Instances dataset = new Instances("LoanDataset", attributes, resultList.size());
        dataset.setClassIndex(attributes.size() - 1); // تنظیم loan_status به عنوان کلاس پیش‌بینی

        boolean isSingleColumn = importantColumns.size() == 1;
        for (Object row : resultList) {
            double[] values = new double[attributes.size()];

            if (isSingleColumn) {
                Object val = row;
                values[0] = convertToDouble(val);
            } else {
                Object[] rowArray = (Object[]) row;
                for (int i = 0; i < rowArray.length; i++) {
                    if (attributes.get(i).isNominal()) { // بررسی Nominal بودن
                        values[i] = attributes.get(i).indexOfValue(String.valueOf(rowArray[i]));
                    } else {
                        values[i] = convertToDouble(rowArray[i]);
                    }
                }
            }
            dataset.add(new DenseInstance(1.0, values));
        }

        return dataset;
    }

    // متد برای دریافت مقدارهای منحصربه‌فرد loan_status
    private List<String> getUniqueLoanStatusValues(List<?> resultList, int loanStatusIndex) {
        Set<String> uniqueValues = new HashSet<>();
        for (Object row : resultList) {
            Object[] rowArray = (Object[]) row;
            uniqueValues.add(String.valueOf(rowArray[loanStatusIndex]));
        }
        return new ArrayList<>(uniqueValues);
    }

    private double convertToDouble(Object val) {
        if (val == null) {
            return Double.NaN; // مقدار گمشده
        } else if (val instanceof Number) {
            return ((Number) val).doubleValue();
        } else {
            return Double.NaN; // مقدارهای String و غیرعددی به NaN تبدیل می‌شوند (چون باید Nominal باشد)
        }
    }


}
