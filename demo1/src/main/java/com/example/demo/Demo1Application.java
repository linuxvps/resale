package com.example.demo;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Map;

@SpringBootApplication
public class Demo1Application implements CommandLineRunner {

    @Autowired
    private CreditRiskService creditRiskService;


    public static void main(String[] args) {
        SpringApplication.run(Demo1Application.class, args);
    }

    @Override
    public void run(String... args) throws Exception {
        String csvFolder = "C:/Users/nima/Desktop/lendingClub/file/55";
        List<Map<String, Object>> maps = creditRiskService.calcStatForList();
        String time = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss").format(new Date());
        String fileName = "CreditRisk_Report_" + time + ".docx";
        creditRiskService.createWordReport(maps,fileName);

    }
}
