package com.example.demo.entity;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import javassist.*;

public class DynamicClassGenerator {

    // متدی که هم کلاس داینامیک ایجاد می‌کند و هم اشیاء را از روی resultList می‌سازد
    public static List<Object> generateDynamicObjects(List<String> columns, List<?> resultList) throws Exception {
        // ایجاد کلاس داینامیک با توجه به لیست ستون‌ها
        Class<?> dynamicClass = generateDynamicLoanDataClass(columns);
        Constructor<?> constructor = dynamicClass.getDeclaredConstructor();
        List<Object> objectList = new ArrayList<>();

        // نگاشت هر ردیف از نتیجه به یک شیء از کلاس داینامیک
        for (Object row : resultList) {
            // ایجاد نمونه جدید از کلاس داینامیک
            Object instance = constructor.newInstance();

            // اطمینان از اینکه هر ردیف به صورت آرایه‌ای از Object است
            if (row instanceof Object[]) {
                Object[] rowArray = (Object[]) row;
                for (int i = 0; i < columns.size(); i++) {
                    String column = columns.get(i);
                    // ایجاد نام متد setter به صورت set + اولین حرف بزرگ + باقی رشته
                    String setterName = "set" + Character.toUpperCase(column.charAt(0)) + column.substring(1);
                    Method setterMethod = dynamicClass.getMethod(setterName, Object.class);
                    setterMethod.invoke(instance, rowArray[i]);
                }
            }
            objectList.add(instance);
        }
        return objectList;
    }

    // متدی که یک کلاس داینامیک با پراپرتی‌های داده شده ایجاد می‌کند
    public static Class<?> generateDynamicLoanDataClass(List<String> columns) throws Exception {
        ClassPool pool = ClassPool.getDefault();
        CtClass ctClass = pool.makeClass("com.example.demo.entity.DynamicLoanData");

        // اضافه کردن یک constructor پیش‌فرض
        CtConstructor constructor = new CtConstructor(new CtClass[]{}, ctClass);
        constructor.setBody("{}");
        ctClass.addConstructor(constructor);

        for (String column : columns) {
            // ایجاد فیلد خصوصی از نوع Object با نام ستون
            CtField ctField = new CtField(pool.get("java.lang.Object"), column, ctClass);
            ctField.setModifiers(Modifier.PRIVATE);
            ctClass.addField(ctField);

            // ایجاد متد getter با نام get + اولین حرف بزرگ + باقی رشته
            String getterName = "get" + Character.toUpperCase(column.charAt(0)) + column.substring(1);
            CtMethod getterMethod = CtNewMethod.getter(getterName, ctField);
            ctClass.addMethod(getterMethod);

            // ایجاد متد setter با نام set + اولین حرف بزرگ + باقی رشته
            String setterName = "set" + Character.toUpperCase(column.charAt(0)) + column.substring(1);
            CtMethod setterMethod = CtNewMethod.setter(setterName, ctField);
            ctClass.addMethod(setterMethod);
        }

        // تغییر: استفاده از ClassLoader برای سازگاری با جاوا 9 به بعد
        return ctClass.toClass(Thread.currentThread().getContextClassLoader(), null);
    }
}
