package org.example.bagging.repository;

import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import jakarta.persistence.Query;
import lombok.SneakyThrows;
import org.example.bagging.entity.DynamicClassGenerator;
import org.springframework.stereotype.Repository;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.Utils;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Repository
public class ParsianLoanCustomRepositoryImpl implements ParsianLoanCustomRepository {

    @PersistenceContext
    private EntityManager entityManager;

    /**
     * متد دریافت مقادیر متمایز یک ستون مشخص از جدول parsian_loan
     *
     * @param column نام ستونی که باید مقادیر یکتا از آن گرفته شود.
     * @return لیستی از مقادیر متمایز
     */
    @Override
    public List<?> findDistinct(final String column, String condition) {
        final String sql = "SELECT DISTINCT " + column + " FROM parsian_loan " + condition;
        Query query = entityManager.createNativeQuery(sql);
        return query.getResultList();
    }

    /**
     * متد ساخت دیتاست Instances برای عملیات یادگیری ماشین (استفاده در Weka)
     *
     * @return مجموعه Instances که حاوی رکوردها و ویژگی‌ها است.
     */
    @SneakyThrows
    @Override
    public Instances createInstance() {
        final String nominalColumnName = "status";
        final String condition = " where id < 1000";
        List<String> importantColumns = loadImportantColumns();
        moveNominalColumnToEnd(importantColumns, nominalColumnName);

        importantColumns = importantColumns.stream().filter(s -> !s.contains("date")).toList();
        ArrayList<Attribute> attributes = importantColumns.stream()
                .filter(column -> !column.equals(nominalColumnName))
                .map(Attribute::new)
                .collect(Collectors.toCollection(ArrayList::new));
        final List<String> nominalStatus = loadNominalValues(nominalColumnName, condition);
        attributes.add(new Attribute(nominalColumnName, nominalStatus));


        final String loanSql = buildLoanSql(importantColumns, condition);
        final List<?> resultList = getLoanData(loanSql);
        final List<Object> dynamicObjects = DynamicClassGenerator.generateDynamicObjects(importantColumns, resultList);

        // ایجاد دیتاست و تنظیم اندیس کلاس (ستون هدف)
        Instances dataset = new Instances("LoanDataset", attributes, dynamicObjects.size());
        dataset.setClassIndex(attributes.size() - 1);

        // افزودن رکوردها به دیتاست
        addInstancesToDataset(dynamicObjects, importantColumns, attributes, dataset);

        return dataset;
    }

    /**
     * متد کمکی برای جابه‌جایی ستون Nominal به انتهای لیست
     */
    private void moveNominalColumnToEnd(final List<String> importantColumns, final String nominalColumnName) {
        importantColumns.remove(nominalColumnName);
        importantColumns.add(nominalColumnName);
    }

    /**
     * متد بارگذاری ستون‌های با اهمیت سطح ۴ از جدول مورد نظر
     *
     * @return لیستی از ستون‌های مهم
     */
    private List<String> loadImportantColumns() {
        final String featureSql =
                "SELECT column_name FROM ln.loan_features WHERE importance_level in (4,3) AND table_name = 'parsian_loan'";
        @SuppressWarnings("unchecked")
        List<String> columns = entityManager.createNativeQuery(featureSql).getResultList();
        if (columns.isEmpty()) {
            throw new RuntimeException("هیچ ستونی با اهمیت 4 برای جدول parsian_loan پیدا نشد.");
        }
        return columns;
    }

    /**
     * متد کمک‌کننده برای ساخت Query اصلی جهت دریافت داده‌های وام
     */
    private String buildLoanSql(final List<String> importantColumns, String condition) {
        String columns = String.join(", ", importantColumns);
        return "SELECT " + columns + " FROM ln.parsian_loan " + condition;
    }

    /**
     * اجرای Query و برگرداندن نتیجه خام
     */
    @SuppressWarnings("unchecked")
    private List<?> getLoanData(final String loanSql) {
        Query loanQuery = entityManager.createNativeQuery(loanSql);
        return loanQuery.getResultList();
    }

    /**
     * افزودن رکوردهای موجود در resultList به دیتاست Weka
     *
     * @param resultList       لیست اشیا (چه آرایه Object[] و چه اشیا داینامیک)
     * @param importantColumns لیست ستون‌های مهم
     * @param attributes       لیست Attributeها
     * @param dataset          شیء Instances هدف برای درج داده
     */
    private void addInstancesToDataset(final List<?> resultList,
                                       final List<String> importantColumns,
                                       final ArrayList<Attribute> attributes,
                                       final Instances dataset) {

        for (Object row : resultList) {
            double[] values = mapRowToValues(row, attributes, importantColumns);
            dataset.add(new DenseInstance(1.0, values));
        }
    }

    private double[] mapRowToValues(final Object row,
                                    final List<Attribute> attributes,
                                    final List<String> importantColumns) {

        double[] values = new double[attributes.size()];

        if (row instanceof Object[]) {
            // اگر row از دیتابیس در قالب آرایه Object[] آمده باشد
            Object[] rowArray = (Object[]) row;
            for (int i = 0; i < rowArray.length; i++) {
                values[i] = convertValue(rowArray[i], attributes.get(i));
            }
        } else {
            // اگر row یک شیء جاوا باشد و ستون‌ها را با Reflection بخوانیم
            for (int i = 0; i < importantColumns.size(); i++) {
                Object fieldValue = getFieldValue(row, importantColumns.get(i));
                values[i] = convertValue(fieldValue, attributes.get(i));
            }
        }

        return values;
    }


    /**
     * متد تبدیل مقدار شیء دریافتی به مقدار عددی یا Nominal (در صورت نامعتبر بودن، مقدار missingValue ثبت می‌شود)
     */
    private double convertValue(final Object value, final Attribute attribute) {
        if (value == null) {
            return Utils.missingValue();
        }
        if (attribute.isNominal()) {
            return attribute.indexOfValue(String.valueOf(value));
        } else {
            return convertToDouble(value);
        }
    }

    /**
     * متد کمک‌کننده برای تبدیل مقدار Object به double
     */
    private double convertToDouble(final Object value) {
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        try {
            return Double.parseDouble(value.toString());
        } catch (NumberFormatException e) {
            return Utils.missingValue();
        }
    }

    /**
     * متد Reflection برای گرفتن مقدار فیلد به شکل دینامیک از یک شیء
     */
    private Object getFieldValue(final Object obj, final String fieldName) {
        try {
            Field field = obj.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.get(obj);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            return null;
        }
    }

    /**
     * دریافت لیستی از مقادیر یکتای ستون Nominal به صورت رشته
     */
    private ArrayList<String> loadNominalValues(final String nominalColumnName, final String condition) {
        return findDistinct(nominalColumnName, condition)
                .stream()
                .map(Object::toString)
                .collect(Collectors.toCollection(ArrayList::new));
    }
}
