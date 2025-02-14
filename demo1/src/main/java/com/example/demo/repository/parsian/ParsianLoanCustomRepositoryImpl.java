package com.example.demo.repository.parsian;

import com.example.demo.entity.DynamicClassGenerator;
import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import jakarta.persistence.Query;
import lombok.SneakyThrows;
import org.springframework.stereotype.Repository;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.Utils;

import java.lang.reflect.Field;
import java.util.*;
import java.util.stream.Collectors;

import static org.antlr.v4.runtime.atn.EmptyPredictionContext.Instance;

@Repository
public class ParsianLoanCustomRepositoryImpl implements ParsianLoanCustomRepository {

    @PersistenceContext
    private EntityManager entityManager;


    @Override
    public List<?> findDistinct(String column) {
        String sql = "SELECT distinct " + column + " FROM parsian_loan ";
        Query query = entityManager.createNativeQuery(sql);

        // تبدیل لیست خروجی به List<Double>
        List<?> resultList = query.getResultList();
        return resultList;
    }

    @SneakyThrows
    @Override
    public Instances createInstance() {
        String nominalColumnName = "status";
        List<String> importantColumns = loadImportantColumns();
        putNominalColumnNameAtTheEnd(importantColumns, nominalColumnName);
        String loanSql = buildLoanSql(importantColumns);
        List<?> resultList = getLoanData(loanSql);
        List<Object> dynamicObjects = DynamicClassGenerator.generateDynamicObjects(importantColumns, resultList);
        List<String> nominalStatus = getNominal(nominalColumnName);

        ArrayList<Attribute> attributes = importantColumns.stream()
                .map(Attribute::new)
                .collect(Collectors.toCollection(ArrayList::new));
        attributes.removeIf(attr -> attr.name().equals(nominalColumnName));
        attributes.add(new Attribute(nominalColumnName, nominalStatus));

        Instances dataset = new Instances("LoanDataset", attributes, dynamicObjects.size());
        dataset.setClassIndex(attributes.size() - 1);
        addInstancesToDataset(dynamicObjects, importantColumns, attributes, dataset);
        return dataset;
    }

    private List<String> fillLabel(List<?> resultList) {
        List<String> labels = new ArrayList<>();

        for (Object result : resultList) {
            Object[] row = (Object[]) result;
            int lastIndex = row.length - 1;
            labels.add(row[lastIndex].toString());
        }

        return labels;

    }


    private Map<String, List<?>> fillFeaturesAndLabel(List<?> resultList) {
        List<double[]> features = new ArrayList<>();
        List<String> labels = new ArrayList<>();

        for (Object result : resultList) {
            Object[] row = (Object[]) result;
            int featureCount = row.length - 1;
            double[] featureValues = new double[featureCount];
            for (int i = 0; i < featureCount; i++) {
                featureValues[i] = convertToDouble(row[i]);
            }
            features.add(featureValues);
            labels.add(row[featureCount].toString());
        }


        Map<String, List<?>> output = new HashMap<>();
        output.put("features", features);
        output.put("labels", labels);
        return output;
    }


    private void putNominalColumnNameAtTheEnd(List<String> importantColumns, String nominalColumnName) {
        importantColumns.remove(nominalColumnName);
        importantColumns.add(nominalColumnName);
    }

    private List<String> loadImportantColumns() {
        String featureSql = "SELECT column_name FROM ln.loan_features WHERE importance_level = 4 AND table_name = 'parsian_loan' ";
        @SuppressWarnings("unchecked") List<String> columns = entityManager.createNativeQuery(featureSql).getResultList();
        if (columns.isEmpty()) {
            throw new RuntimeException("هیچ ستونی با اهمیت 4 برای جدول loan پیدا نشد.");
        }
        return columns;
    }

    private String buildLoanSql(List<String> importantColumns) {
        String columns = String.join(", ", importantColumns);
        return "SELECT " + columns + " FROM ln.parsian_loan WHERE id < 100";
    }

    @SuppressWarnings("unchecked")
    private List<?> getLoanData(String loanSql) {
        Query loanQuery = entityManager.createNativeQuery(loanSql);
        return loanQuery.getResultList();
    }


    public void addInstancesToDataset(List<?> resultList, List<String> importantColumns, ArrayList<Attribute> attributes, Instances dataset) {

        for (Object row : resultList) {
            double[] values = new double[attributes.size()];

            if (row instanceof Object[]) {
                // حالت 1: اگر row از نوع Object[] باشد (یعنی از دیتابیس خوانده شده باشد)
                Object[] rowArray = (Object[]) row;
                for (int i = 0; i < rowArray.length; i++) {
                    values[i] = convertValue(rowArray[i], attributes.get(i));
                }
            } else {
                // حالت 2: اگر row یک شیء جاوا باشد (مثلاً ParsianLoan)
                for (int i = 0; i < importantColumns.size(); i++) {
                    Object fieldValue = getFieldValue(row, importantColumns.get(i));
                    values[i] = convertValue(fieldValue, attributes.get(i));
                }
            }

            dataset.add(new DenseInstance(1.0, values));
        }
    }

    // متد کمکی برای تبدیل مقدار به عدد یا مقدار Nominal
    private double convertValue(Object value, Attribute attribute) {
        if (value == null) {
            return Utils.missingValue();
        }
        if (attribute.isNominal()) {
            return attribute.indexOfValue(String.valueOf(value));
        } else {
            return convertToDouble(value);
        }
    }

    // متد کمکی برای تبدیل Object به Double (پشتیبانی از انواع مختلف عددی)
    private double convertToDouble(Object value) {
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        try {
            return Double.parseDouble(value.toString());
        } catch (NumberFormatException e) {
            return Utils.missingValue();
        }
    }

    // متد کمکی برای دریافت مقدار فیلد از یک شیء جاوا با استفاده از Reflection
    private Object getFieldValue(Object obj, String fieldName) {
        try {
            Field field = obj.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.get(obj);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            return null; // مقدار نامعتبر
        }
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


    private ArrayList<String> getNominal(String nominalColumnName) {
        // دریافت مقادیر متمایز از دیتابیس و تبدیل به لیستی از Attribute با استفاده از استریم
        ArrayList<String> collect = findDistinct(nominalColumnName).stream().map(Object::toString).collect(Collectors.toCollection(ArrayList::new));
        return collect;
    }


}
