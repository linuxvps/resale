package com.example.demo.test;

import java.util.Date;
import java.util.List;

public class main {
    Employee employee1 =
            new Employee(20,new Date(120, 0, 1),3,"low");
    Employee employee2 =
            new Employee(30,new Date(110, 0, 1),8,"medium");
    Employee employee3 =
            new Employee(40,new Date(115, 0, 1),4,"medium");
    Employee employee4 =
            new Employee(50,new Date(100, 0, 1),9,"high");

    List<Employee> employees = List.of(employee1 ,employee2,employee3,employee4);

}
