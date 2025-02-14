package com.example.demo.test;

import java.util.Date;

public class Employee {
    private int age;
    private Date hireDate;
    private double experience;
    private String salary;

    public Employee(int age, Date hireDate, double experience, String salary) {
        this.age = age;
        this.hireDate = hireDate;
        this.experience = experience;
        this.salary = salary;
    }

    public int getAge() { return age; }
    public Date getHireDate() { return hireDate; }
    public double getExperience() { return experience; }
    public String getSalary() { return salary; }
}
